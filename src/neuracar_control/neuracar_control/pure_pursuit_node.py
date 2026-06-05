#!/usr/bin/env python3
"""
=======================================================================
 Pure Pursuit Controller — Neuracar
 Proyecto: Neuracar / Smart Mobility
 Adaptado de: Pure Pursuit QCar (Marmanja / Iván Valdez del Toro)
-----------------------------------------------------------------------
 Implementa Pure Pursuit con lookahead dinámico para seguimiento de
 trayectoria CSV. Misma lógica que el QCar pero con:
   - Topics Neuracar (/neuracar/odometry, /neuracar/user_command, etc.)
   - Rutas estructuradas en neuracar_control/data/
   - Parada por obstáculo (/neuracar/lidar/obstacle_alert)
   - Análisis cuantitativo + gráficas al terminar (igual que Stanley)
   - Normalización rad→[-1,1] para el ESP32-A

 Ley de control Pure Pursuit (Ackermann):
   α  = atan2(ty − yr, tx − xr) − θ    (ángulo al punto objetivo)
   δ  = atan2(2·L·sin(α), Lf)           (ángulo de dirección)
   Lf = k·v + Lfc                        (lookahead dinámico)

 Suscribe:
   /neuracar/odometry            (nav_msgs/Odometry)
   /neuracar/velocity            (geometry_msgs/TwistStamped)
   /neuracar/lidar/obstacle_alert (std_msgs/Bool)

 Publica:
   /neuracar/user_command  (geometry_msgs/Vector3Stamped)
     vector.x = throttle ∈ [-1,1]
     vector.y = steering ∈ [-1,1]  (+1=derecha, −1=izquierda)

 Parámetros ROS2:
   run_name      (str)   — CSV en trajectories/  (obligatorio)
   wheelbase     (float) — distancia entre ejes [m]   [default: 0.256]
   lookahead     (float) — Lfc base [m]               [default: 0.20]
   k_gain        (float) — ganancia lookahead dinámico [default: 0.5]
   speed         (float) — throttle crucero            [default: 0.55]
   speed_curve   (float) — throttle en curva           [default: 0.55]
   min_throttle  (float) — mínimo físico del motor     [default: 0.45]
   max_steer     (float) — ángulo máx. servo [rad]     [default: 0.5]
   goal_radius   (float) — radio de meta [m]           [default: 0.05]
   loop          (bool)  — repetir trayectoria         [default: False]
   max_loops     (int)   — vueltas máximas (0=∞)       [default: 1]

 Uso:
   ros2 run neuracar_control pure_pursuit_node \
     --ros-args -p run_name:=vuelta_04 -p speed:=0.55
=======================================================================
"""

import csv
import math
import os
import time
from datetime import datetime
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


# ── Tipos ────────────────────────────────────────────────────────────────────
Waypoint = Tuple[float, float, float]   # x, y, theta


# ── Utilidades de ruta ────────────────────────────────────────────────────────
def resolve_data_dir() -> str:
    pkg_src = os.path.expanduser(
        "~/Workspaces/Neuracar/src/neuracar_control")
    if os.path.isdir(pkg_src):
        d = os.path.join(pkg_src, "data", "trajectories")
        os.makedirs(d, exist_ok=True)
        return d
    try:
        from ament_index_python.packages import get_package_share_directory
        get_package_share_directory("neuracar_control")
        d = os.path.join(os.path.expanduser("~/Workspaces/Neuracar/src"),
                         "neuracar_control", "data", "trajectories")
        os.makedirs(d, exist_ok=True)
        return d
    except Exception:
        pass
    d = os.path.expanduser(
        "~/Workspaces/Neuracar/src/neuracar_control/data/trajectories")
    os.makedirs(d, exist_ok=True)
    return d


def resolve_runs_dir() -> str:
    pkg_src = os.path.expanduser(
        "~/Workspaces/Neuracar/src/neuracar_control")
    if os.path.isdir(pkg_src):
        d = os.path.join(pkg_src, "data", "runs")
        os.makedirs(d, exist_ok=True)
        return d
    d = os.path.expanduser(
        "~/Workspaces/Neuracar/src/neuracar_control/data/runs")
    os.makedirs(d, exist_ok=True)
    return d


def load_csv(path: str) -> List[Waypoint]:
    pts = []
    with open(path) as f:
        for row in csv.DictReader(f):
            pts.append((float(row['x']), float(row['y']), float(row['theta'])))
    return pts


def dist2d(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def normalize(angle: float) -> float:
    while angle >  math.pi: angle -= 2 * math.pi
    while angle <= -math.pi: angle += 2 * math.pi
    return angle


def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
    return math.atan2(siny, cosy)


# ── Análisis (igual que Stanley) ──────────────────────────────────────────────
def compute_analysis(ref, real, run_name, runs_dir, elapsed,
                     obstacle_stops, loops_done):
    if not real:
        return

    rows = []
    cte_list = []
    for rx, ry, ryaw, rt in real:
        min_d = float('inf')
        best_ref = ref[0]
        for wp in ref:
            d = dist2d((rx, ry), (wp[0], wp[1]))
            if d < min_d:
                min_d = d
                best_ref = wp
        wx, wy, wtheta = best_ref
        dx = rx - wx; dy = ry - wy
        cte = -math.sin(wtheta) * dx + math.cos(wtheta) * dy
        cte_list.append(cte)
        rows.append({'time_s': round(rt, 3),
                     'ref_x': round(wx, 4), 'ref_y': round(wy, 4),
                     'ref_theta': round(wtheta, 4),
                     'real_x': round(rx, 4), 'real_y': round(ry, 4),
                     'real_theta': round(ryaw, 4),
                     'cte_m': round(cte, 4),
                     'dist_to_ref': round(min_d, 4)})

    n = len(cte_list)
    cte_abs    = [abs(c) for c in cte_list]
    rms_cte    = math.sqrt(sum(c**2 for c in cte_list) / n)
    max_cte    = max(cte_abs)
    mean_cte   = sum(cte_list) / n
    pct_u5     = sum(1 for c in cte_abs if c < 0.05) / n * 100
    pct_u10    = sum(1 for c in cte_abs if c < 0.10) / n * 100

    if rms_cte < 0.05:   quality = 'EXCELENTE — seguimiento muy preciso (RMS < 5 cm)'
    elif rms_cte < 0.10: quality = 'BUENO — ligeras desviaciones, aceptable para pista'
    elif rms_cte < 0.20: quality = 'REGULAR — revisar ganancias (k_gain, lookahead)'
    else:                quality = 'DEFICIENTE — revisar odometría, velocidad o ganancias'

    peak_note = (f'Pico alto ({max_cte:.3f} m) — revisar curvas o inicio de ruta'
                 if max_cte > 0.30 else
                 f'Pico moderado ({max_cte:.3f} m) — normal en curvas cerradas'
                 if max_cte > 0.15 else
                 f'Picos pequeños ({max_cte:.3f} m) — comportamiento estable')

    if abs(mean_cte) > 0.03:
        side = 'derecha' if mean_cte > 0 else 'izquierda'
        bias_note = (f'Sesgo hacia la {side} ({mean_cte:+.3f} m) — '
                     f'ajustar lookahead o revisar calibración IMU')
    else:
        bias_note = f'Sin sesgo apreciable (mean CTE = {mean_cte:+.3f} m)'

    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f'{run_name}_pp_{ts}'

    # CSV
    csv_out = os.path.join(runs_dir, f'{stem}_analysis.csv')
    with open(csv_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    # Reporte texto
    sep = '=' * 56
    txt_out = os.path.join(runs_dir, f'{stem}_report.txt')
    with open(txt_out, 'w') as f:
        f.write(f'{sep}\n PURE PURSUIT — REPORTE DE ANÁLISIS\n Neuracar\n{sep}\n')
        f.write(f' Prueba        : {run_name}\n')
        f.write(f' Fecha         : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f' Duración      : {elapsed:.1f} s\n')
        f.write(f' Vueltas       : {loops_done}\n')
        f.write(f' Paradas lidar : {obstacle_stops}\n')
        f.write(f' Puntos real   : {n}\n')
        f.write(f' Waypoints ref : {len(ref)}\n{sep}\n')
        f.write(f' MÉTRICAS CUANTITATIVAS\n{sep}\n')
        f.write(f'  RMS CTE        : {rms_cte:.4f} m\n')
        f.write(f'  CTE máximo     : {max_cte:.4f} m\n')
        f.write(f'  CTE medio      : {mean_cte:+.4f} m\n')
        f.write(f'  % pts < 5 cm   : {pct_u5:.1f}%\n')
        f.write(f'  % pts < 10 cm  : {pct_u10:.1f}%\n{sep}\n')
        f.write(f' ANÁLISIS CUALITATIVO\n{sep}\n')
        f.write(f'  Calidad general : {quality}\n')
        f.write(f'  Picos           : {peak_note}\n')
        f.write(f'  Sesgo lateral   : {bias_note}\n')
        if obstacle_stops > 0:
            f.write(f'  Obstáculos      : {obstacle_stops} parada(s)\n')
        f.write(f'{sep}\n')

    # Gráficas
    try:
        times    = [r['time_s']  for r in rows]
        ref_x    = [r['ref_x']   for r in rows]
        ref_y    = [r['ref_y']   for r in rows]
        real_x   = [r['real_x']  for r in rows]
        real_y   = [r['real_y']  for r in rows]

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle(
            f'Pure Pursuit — {run_name}\n'
            f'RMS={rms_cte:.3f}m  max={max_cte:.3f}m  '
            f'<5cm={pct_u5:.0f}%  <10cm={pct_u10:.0f}%', fontsize=11)

        ax = axes[0, 0]
        ax.plot(ref_x, ref_y, '-r', lw=2, label='Referencia')
        ax.plot(real_x, real_y, '-b', lw=1.5, label='Real')
        if real_x:
            ax.scatter(real_x[0],  real_y[0],  s=100, c='green',
                       marker='o', zorder=5, label='Inicio')
            ax.scatter(real_x[-1], real_y[-1], s=100, c='red',
                       marker='x', zorder=5, label='Fin')
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
        ax.set_title('Trayectoria XY')
        ax.legend(fontsize=8); ax.grid(True); ax.set_aspect('equal')

        ax = axes[0, 1]
        ax.plot(times, cte_list, '-g', lw=1.5, label='CTE')
        ax.axhline(y=0,        color='k', lw=0.8, ls='--')
        ax.axhline(y=mean_cte, color='r', lw=1,   ls=':',
                   label=f'Media {mean_cte:+.3f}m')
        ax.axhline(y= 0.05, color='orange', lw=0.7, ls=':')
        ax.axhline(y=-0.05, color='orange', lw=0.7, ls=':', label='±5 cm')
        ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('CTE [m]')
        ax.set_title('Error lateral (CTE con signo)')
        ax.legend(fontsize=8); ax.grid(True)

        ax = axes[1, 0]
        ax.plot(times, cte_abs, '-m', lw=1.5, label='|CTE|')
        ax.axhline(y=rms_cte, color='r', lw=1.2, ls='--',
                   label=f'RMS {rms_cte:.3f}m')
        ax.axhline(y=0.05, color='orange', lw=0.7, ls=':', label='5 cm')
        ax.axhline(y=0.10, color='red',    lw=0.7, ls=':', label='10 cm')
        ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('|CTE| [m]')
        ax.set_title('Error lateral absoluto')
        ax.legend(fontsize=8); ax.grid(True)

        ax = axes[1, 1]
        ax.hist(cte_abs, bins=30, color='steelblue',
                edgecolor='white', alpha=0.8)
        ax.axvline(x=rms_cte, color='r',      lw=1.5, ls='--',
                   label=f'RMS {rms_cte:.3f}m')
        ax.axvline(x=0.05,    color='orange',  lw=1,   ls=':',
                   label='5 cm')
        ax.axvline(x=0.10,    color='red',     lw=1,   ls=':',
                   label='10 cm')
        ax.set_xlabel('|CTE| [m]'); ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución del error lateral')
        ax.legend(fontsize=8); ax.grid(True)

        plt.tight_layout()
        png_out = os.path.join(runs_dir, f'{stem}_plot.png')
        fig.savefig(png_out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Gráfica    → {png_out}')
    except Exception as exc:
        print(f'  [WARN] Gráfica no generada: {exc}')

    sep = '=' * 56
    print(f'\n{sep}')
    print(f' ANÁLISIS PURE PURSUIT — {run_name}')
    print(f'{sep}')
    print(f'  RMS CTE     : {rms_cte:.4f} m')
    print(f'  CTE máximo  : {max_cte:.4f} m')
    print(f'  CTE medio   : {mean_cte:+.4f} m')
    print(f'  < 5 cm      : {pct_u5:.1f}%')
    print(f'  < 10 cm     : {pct_u10:.1f}%')
    print(f'  Calidad     : {quality}')
    print(f'  Picos       : {peak_note}')
    print(f'  Sesgo       : {bias_note}')
    print(f'  CSV         → {csv_out}')
    print(f'  Reporte     → {txt_out}')
    print(f'{sep}\n')


# ── Nodo principal ────────────────────────────────────────────────────────────
class PurePursuitNode(Node):

    def __init__(self):
        super().__init__('pure_pursuit_node')

        # ── Parámetros ─────────────────────────────────────────────
        self.declare_parameter('run_name',     '')
        self.declare_parameter('wheelbase',    0.256)
        self.declare_parameter('lookahead',    0.20)
        self.declare_parameter('k_gain',       0.5)
        self.declare_parameter('speed',        0.55)
        self.declare_parameter('speed_curve',  0.55)
        self.declare_parameter('min_throttle', 0.45)
        self.declare_parameter('max_steer',    0.5)
        self.declare_parameter('goal_radius',  0.05)
        self.declare_parameter('loop',         False)
        self.declare_parameter('max_loops',    1)

        run_name = self.get_parameter('run_name').value
        if not run_name:
            self.get_logger().fatal('Parámetro run_name vacío.')
            raise RuntimeError('run_name requerido')

        self._L            = self.get_parameter('wheelbase').value
        self._Lfc          = self.get_parameter('lookahead').value
        self._k_gain       = self.get_parameter('k_gain').value
        self._speed        = self.get_parameter('speed').value
        self._speed_curve  = self.get_parameter('speed_curve').value
        self._min_throttle = self.get_parameter('min_throttle').value
        self._max_steer    = self.get_parameter('max_steer').value
        self._goal_r       = self.get_parameter('goal_radius').value
        self._loop         = self.get_parameter('loop').value
        self._max_loops    = self.get_parameter('max_loops').value

        # ── Rutas ─────────────────────────────────────────────────
        self._data_dir = resolve_data_dir()
        self._runs_dir = resolve_runs_dir()
        csv_path = os.path.join(self._data_dir, f'{run_name}.csv')
        if not os.path.exists(csv_path):
            self.get_logger().fatal(f'CSV no encontrado: {csv_path}')
            raise FileNotFoundError(csv_path)

        self._run_name  = run_name
        self._waypoints = load_csv(csv_path)
        self._n         = len(self._waypoints)
        if self._n < 2:
            raise RuntimeError('CSV necesita al menos 2 waypoints')

        # ── Estado del vehículo ────────────────────────────────────
        self._x    = 0.0
        self._y    = 0.0
        self._yaw  = 0.0
        self._v    = 0.0

        # ── Estado del controlador ─────────────────────────────────
        self._nearest_idx  = 0   # índice del waypoint más cercano
        self._obstacle     = False
        self._obs_stops    = 0
        self._done         = False
        self._loops        = 0
        self._start_time   = time.time()

        # ── Registro de trayectoria real ───────────────────────────
        self._real_track: List[Tuple[float, float, float, float]] = []
        self.create_timer(0.1, self._record_pose)

        # ── Publisher ──────────────────────────────────────────────
        self._pub = self.create_publisher(
            Vector3Stamped, '/neuracar/user_command', 10)

        # ── Subscribers ────────────────────────────────────────────
        self.create_subscription(
            Odometry, '/neuracar/odometry', self._odom_cb, 10)
        self.create_subscription(
            TwistStamped, '/neuracar/velocity', self._vel_cb, 10)
        self.create_subscription(
            Bool, '/neuracar/lidar/obstacle_alert', self._obs_cb, 10)

        # ── Timer de control 20 Hz ─────────────────────────────────
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info('=' * 52)
        self.get_logger().info(' PURE PURSUIT — Neuracar')
        self.get_logger().info('=' * 52)
        self.get_logger().info(
            f'  Trayectoria: {self._n} waypoints — {csv_path}')
        self.get_logger().info(
            f'  L={self._L}m  Lfc={self._Lfc}m  k_gain={self._k_gain}')
        self.get_logger().info(
            f'  speed={self._speed}  min_throttle={self._min_throttle}')
        self.get_logger().info(
            f'  max_steer={self._max_steer}rad  goal_radius={self._goal_r}m')
        self.get_logger().info(
            f'  trajectories: {self._data_dir}')
        self.get_logger().info(
            f'  resultados:   {self._runs_dir}')

    # ── Callbacks ──────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        self._x   = msg.pose.pose.position.x
        self._y   = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

    def _vel_cb(self, msg: TwistStamped):
        self._v = msg.twist.linear.x

    def _obs_cb(self, msg: Bool):
        was = self._obstacle
        self._obstacle = msg.data
        if self._obstacle and not was:
            self._obs_stops += 1
            self.get_logger().warn(
                f'¡Obstáculo! Parada #{self._obs_stops}')
        elif not self._obstacle and was:
            self.get_logger().info('Obstáculo despejado — reanudando')

    def _record_pose(self):
        if not self._done:
            t = time.time() - self._start_time
            self._real_track.append((self._x, self._y, self._yaw, t))

    # ── Búsqueda del waypoint más cercano ─────────────────────────
    def _find_nearest(self) -> int:
        """
        Busca el waypoint más cercano en una ventana amplia hacia adelante.
        Igual que el Stanley corregido — cubre 4 segundos de movimiento.
        """
        if self._n > 10:
            d_sample = dist2d(
                (self._waypoints[0][0], self._waypoints[0][1]),
                (self._waypoints[9][0], self._waypoints[9][1])
            ) / 9.0
            wp_per_sec = abs(self._v) / d_sample if d_sample > 1e-4 and self._v != 0 else 30
        else:
            wp_per_sec = 30

        look = max(150, int(wp_per_sec * 4))
        start = max(0,       self._nearest_idx - 3)
        end   = min(self._n, self._nearest_idx + look)

        best, min_d = self._nearest_idx, float('inf')
        for i in range(start, end):
            d = dist2d((self._x, self._y),
                       (self._waypoints[i][0], self._waypoints[i][1]))
            if d < min_d:
                min_d, best = d, i
        return best

    # ── Búsqueda del punto objetivo con lookahead dinámico ─────────
    def _find_target(self, nearest: int) -> Tuple[float, float, int]:
        """
        Desde el waypoint más cercano, busca el primero que esté
        a distancia ≥ Lf (lookahead dinámico).
        Retorna (tx, ty, target_idx).
        """
        v   = max(abs(self._v), 0.0)
        Lf  = max(self._k_gain * v + self._Lfc, self._Lfc)

        target = nearest
        while target < self._n - 1:
            d = dist2d((self._x, self._y),
                       (self._waypoints[target][0], self._waypoints[target][1]))
            if d >= Lf:
                break
            target += 1

        tx, ty, _ = self._waypoints[target]
        return tx, ty, target, Lf

    # ── Loop de control ────────────────────────────────────────────
    def _control_loop(self):
        if self._obstacle:
            self._publish(0.0, 0.0)
            return

        if self._done:
            self._publish(0.0, 0.0)
            return

        # ── Verificar meta ────────────────────────────────────────
        ex, ey, _ = self._waypoints[-1]
        if dist2d((self._x, self._y), (ex, ey)) < self._goal_r:
            self._loops += 1
            if self._loop and (self._max_loops == 0
                               or self._loops < self._max_loops):
                self._nearest_idx = 0
                self.get_logger().info(
                    f'Vuelta {self._loops} completada — reiniciando')
            else:
                self._done = True
                self.get_logger().info(
                    f'¡Meta alcanzada! Vueltas: {self._loops}')
                self._publish(0.0, 0.0)
                return

        # ── Encontrar nearest y target ────────────────────────────
        nearest = self._find_nearest()
        self._nearest_idx = nearest
        tx, ty, target_idx, Lf = self._find_target(nearest)

        # ── Pure Pursuit — ángulo α ───────────────────────────────
        alpha = math.atan2(ty - self._y, tx - self._x) - self._yaw
        alpha = normalize(alpha)

        # ── Ley de control Ackermann ──────────────────────────────
        Lf_safe  = max(Lf, 1e-3)
        steer_rad = math.atan2(2.0 * self._L * math.sin(alpha), Lf_safe)

        # Normalizar rad → [-1,1] para ESP32-A
        steering_norm = steer_rad / self._max_steer
        steering_norm = max(-1.0, min(1.0, steering_norm))

        # Velocidad adaptativa
        raw_speed = self._speed_curve if abs(steering_norm) > 0.3 else self._speed
        speed = max(raw_speed, self._min_throttle)

        self.get_logger().info(
            f'wp={target_idx}/{self._n}  Lf={Lf:.2f}m | '
            f'α={math.degrees(alpha):+.1f}° | '
            f'δ={steer_rad:+.3f}rad norm={steering_norm:+.3f} | '
            f'throttle={speed:.2f}',
            throttle_duration_sec=0.3)

        self._publish(speed, steering_norm)

    def _publish(self, speed: float, steering: float):
        try:
            msg = Vector3Stamped()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.vector.x        = float(speed)
            msg.vector.y        = float(steering)
            self._pub.publish(msg)
        except Exception:
            pass

    def finalize(self):
        elapsed = time.time() - self._start_time
        compute_analysis(
            ref            = self._waypoints,
            real           = self._real_track,
            run_name       = self._run_name,
            runs_dir       = self._runs_dir,
            elapsed        = elapsed,
            obstacle_stops = self._obs_stops,
            loops_done     = self._loops,
        )


# ── main ──────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    try:
        node = PurePursuitNode()
    except (RuntimeError, FileNotFoundError) as e:
        print(f'[ERROR] {e}')
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'[WARN] spin terminó con excepción: {exc}')
    finally:
        node._publish(0.0, 0.0)
        print('\nGuardando análisis de la prueba...')
        node.finalize()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print('Pure Pursuit detenido.')


if __name__ == '__main__':
    main()
    