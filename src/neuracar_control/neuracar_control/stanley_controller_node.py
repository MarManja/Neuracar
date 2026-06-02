#!/usr/bin/env python3
"""
=======================================================================
 Stanley Controller — Neuracar  (trayectoria CSV)
 Proyecto: Neuracar / Smart Mobility
-----------------------------------------------------------------------
 Implementa el controlador Stanley para seguimiento de trayectoria.
 Combina error de heading + error lateral (cross-track error) para
 calcular el ángulo de dirección.

   - Rutas estructuradas en neuracar_control/data/
   - Parada de emergencia por obstáculo (/neuracar/lidar/obstacle_alert)
   - Registro de trayectoria real vs referencia
   - Análisis cuantitativo y cualitativo al finalizar
   - Guardado automático de análisis en CSV + reporte de texto

 Ley de control Stanley:
   δ(t) = ψ_e(t) + arctan( k · e(t) / (v(t) + k_s) )

   donde:
     ψ_e = error de heading entre el vehículo y el segmento de trayectoria
     e   = cross-track error (distancia lateral al waypoint más cercano)
     v   = velocidad lineal del vehículo
     k   = ganancia Stanley (tunable)
     k_s = softening constant (evita división por cero a v≈0)

 Suscribe:
   /neuracar/odometry  (nav_msgs/Odometry)   — pose (x, y, θ)
   /neuracar/velocity  (geometry_msgs/TwistStamped) — velocidad v
   /neuracar/lidar/obstacle_alert (std_msgs/Bool) — parada de emergencia

 Publica:
   /neuracar/user_command  (geometry_msgs/Vector3Stamped)
       twist.linear.x   = throttle [m/s]  (velocidad objetivo) [-1, 1]
       twist.angular.y  = steering [rad]  (ángulo de dirección) [-1, 1]

 Parámetros ROS2:
    run_name      (str)   — nombre del CSV en neuracar_control/data/
                            SIN extensión  (obligatorio)
   k             (float) — ganancia Stanley          [default: 0.5]
   k_soft        (float) — softening constant        [default: 0.5]
   speed         (float) — velocidad de crucero m/s  [default: 0.3]
   speed_curve   (float) — velocidad en curva m/s    [default: 0.2]
   max_steer     (float) — límite de steering rad    [default: 0.5]
   lookahead_idx (int)   — waypoints hacia adelante  [default: 3]
   goal_radius   (m)     — radio para detectar meta  [default: 0.3]
   loop          (bool)  — repetir trayectoria       [default: True]
   max_loops     (int)   — vueltas máximas (0=inf)     [default: 1]

 Uso:
   ros2 run neuracar stanley_controller \
     --ros-args -p run_name:=vuelta_01 -p k:=0.5
=======================================================================
"""
 
import csv
import math
import os
import time
from datetime import datetime
from typing import List, Tuple, Optional
 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
 
 
# ── Tipos ───────────────────────────────────────────────────────────────────
Waypoint = Tuple[float, float, float]   # x, y, theta
 
 
# ── Utilidades ──────────────────────────────────────────────────────────────
def resolve_data_dir() -> str:
    """
    Ruta canónica: ~/Workspaces/Neuracar/src/neuracar_control/data/
    Si no existe, la crea con mkdir -p.
    Como fallback (entorno sin ament o fuera del workspace) intenta
    ament_index antes de caer en la ruta hardcoded.
    """
    canonical = os.path.expanduser(
        "~/Workspaces/Neuracar/src/neuracar_control/data"
    )
    pkg_src = os.path.expanduser(
        "~/Workspaces/Neuracar/src/neuracar_control"
    )
    if os.path.isdir(pkg_src):
        os.makedirs(canonical, exist_ok=True)
        return canonical
 
    try:
        from ament_index_python.packages import get_package_share_directory
        get_package_share_directory("neuracar_control")
        src_data = os.path.join(
            os.path.expanduser("~/Workspaces/Neuracar/src"),
            "neuracar_control", "data"
        )
        os.makedirs(src_data, exist_ok=True)
        return src_data
    except Exception:
        pass
 
    os.makedirs(canonical, exist_ok=True)
    return canonical
 
 
def load_csv(path: str) -> List[Waypoint]:
    pts = []
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            pts.append((float(row['x']), float(row['y']), float(row['theta'])))
    return pts
 
 
def normalize(angle: float) -> float:
    while angle > math.pi:  angle -= 2 * math.pi
    while angle <= -math.pi: angle += 2 * math.pi
    return angle
 
 
def dist2d(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])
 
 
def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
    return math.atan2(siny, cosy)
 
 
# ── Análisis ─────────────────────────────────────────────────────────────────
def compute_analysis(ref: List[Waypoint],
                     real: List[Tuple[float, float, float, float]],
                     run_name: str,
                     data_dir: str,
                     elapsed: float,
                     obstacle_stops: int,
                     loops_done: int) -> None:
    """
    Calcula métricas cuantitativas y guarda:
      - <run_name>_analysis.csv  — puntos referencia + real + error
      - <run_name>_report.txt   — reporte legible con análisis cualitativo
    """
    if not real:
        return
 
    # ── Métricas por punto ────────────────────────────────────────
    # Para cada punto real, encontrar el waypoint de referencia más cercano
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
 
        # Cross-track error con signo
        wx, wy, wtheta = best_ref
        # Error lateral: proyección perpendicular al heading de referencia
        dx = rx - wx
        dy = ry - wy
        cte = -math.sin(wtheta) * dx + math.cos(wtheta) * dy
        cte_list.append(cte)
 
        rows.append({
            'time_s':      round(rt, 3),
            'ref_x':       round(wx, 4),
            'ref_y':       round(wy, 4),
            'ref_theta':   round(wtheta, 4),
            'real_x':      round(rx, 4),
            'real_y':      round(ry, 4),
            'real_theta':  round(ryaw, 4),
            'cte_m':       round(cte, 4),
            'dist_to_ref': round(min_d, 4),
        })
 
    cte_arr = cte_list
    n       = len(cte_arr)
    cte_abs = [abs(c) for c in cte_arr]
    rms_cte = math.sqrt(sum(c**2 for c in cte_arr) / n)
    max_cte = max(cte_abs)
    mean_cte = sum(cte_arr) / n
    pct_under5 = sum(1 for c in cte_abs if c < 0.05) / n * 100
    pct_under10 = sum(1 for c in cte_abs if c < 0.10) / n * 100
 
    # ── Guardar CSV de análisis ───────────────────────────────────
    csv_out = os.path.join(data_dir, f'{run_name}_analysis.csv')
    with open(csv_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
 
    # ── Análisis cualitativo ──────────────────────────────────────
    if rms_cte < 0.05:
        quality = 'EXCELENTE — seguimiento muy preciso (RMS < 5 cm)'
    elif rms_cte < 0.10:
        quality = 'BUENO — ligeras desviaciones, aceptable para pista'
    elif rms_cte < 0.20:
        quality = 'REGULAR — revisar ganancias Stanley (k, k_soft)'
    else:
        quality = 'DEFICIENTE — revisar odometría, velocidad o ganancias'
 
    if max_cte > 0.30:
        peak_note = f'Pico alto ({max_cte:.3f} m) — revisar curvas o inicio de ruta'
    elif max_cte > 0.15:
        peak_note = f'Pico moderado ({max_cte:.3f} m) — normal en curvas cerradas'
    else:
        peak_note = f'Picos pequeños ({max_cte:.3f} m) — comportamiento estable'
 
    bias_note = ''
    if abs(mean_cte) > 0.03:
        side = 'derecha' if mean_cte > 0 else 'izquierda'
        bias_note = (f'Sesgo sistemático hacia la {side} ({mean_cte:+.3f} m) — '
                     f'ajustar target_x_ratio o revisar calibración IMU')
    else:
        bias_note = f'Sin sesgo apreciable (mean CTE = {mean_cte:+.3f} m)'
 
    # ── Guardar reporte de texto ──────────────────────────────────
    txt_out = os.path.join(data_dir, f'{run_name}_report.txt')
    sep = '=' * 56
    with open(txt_out, 'w') as f:
        f.write(f'{sep}\n')
        f.write(f' STANLEY CONTROLLER — REPORTE DE ANÁLISIS\n')
        f.write(f' Neuracar / Smart Mobility\n')
        f.write(f'{sep}\n')
        f.write(f' Prueba        : {run_name}\n')
        f.write(f' Fecha         : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f' Duración      : {elapsed:.1f} s\n')
        f.write(f' Vueltas       : {loops_done}\n')
        f.write(f' Paradas lidar : {obstacle_stops}\n')
        f.write(f' Puntos real   : {n}\n')
        f.write(f' Waypoints ref : {len(ref)}\n')
        f.write(f'{sep}\n')
        f.write(f' MÉTRICAS CUANTITATIVAS\n')
        f.write(f'{sep}\n')
        f.write(f'  RMS CTE        : {rms_cte:.4f} m\n')
        f.write(f'  CTE máximo     : {max_cte:.4f} m\n')
        f.write(f'  CTE medio      : {mean_cte:+.4f} m\n')
        f.write(f'  % pts < 5 cm   : {pct_under5:.1f}%\n')
        f.write(f'  % pts < 10 cm  : {pct_under10:.1f}%\n')
        f.write(f'{sep}\n')
        f.write(f' ANÁLISIS CUALITATIVO\n')
        f.write(f'{sep}\n')
        f.write(f'  Calidad general : {quality}\n')
        f.write(f'  Picos           : {peak_note}\n')
        f.write(f'  Sesgo lateral   : {bias_note}\n')
        if obstacle_stops > 0:
            f.write(f'  Obstáculos      : {obstacle_stops} parada(s) — '
                    f'verificar clearance del detector\n')
        f.write(f'{sep}\n')
        f.write(f' ARCHIVOS\n')
        f.write(f'{sep}\n')
        f.write(f'  CSV análisis : {csv_out}\n')
        f.write(f'  Reporte txt  : {txt_out}\n')
        f.write(f'{sep}\n')
 
    # ── Imprimir en terminal ──────────────────────────────────────
    print(f'\n{sep}')
    print(f' ANÁLISIS — {run_name}')
    print(f'{sep}')
    print(f'  RMS CTE     : {rms_cte:.4f} m')
    print(f'  CTE máximo  : {max_cte:.4f} m')
    print(f'  CTE medio   : {mean_cte:+.4f} m')
    print(f'  < 5 cm      : {pct_under5:.1f}%')
    print(f'  < 10 cm     : {pct_under10:.1f}%')
    print(f'  Calidad     : {quality}')
    print(f'  Picos       : {peak_note}')
    print(f'  Sesgo       : {bias_note}')
    print(f'  CSV         → {csv_out}')
    print(f'  Reporte     → {txt_out}')
    print(f'{sep}\n')
 
 
# ── Nodo principal ────────────────────────────────────────────────────────────
class StanleyController(Node):
 
    def __init__(self):
        super().__init__('stanley_controller')
 
        # ── Parámetros ─────────────────────────────────────────────
        self.declare_parameter('run_name',    '')
        self.declare_parameter('k',           0.5)
        self.declare_parameter('k_soft',      0.5)
        self.declare_parameter('speed',       0.3)
        self.declare_parameter('speed_curve', 0.2)
        self.declare_parameter('max_steer',   0.5)
        self.declare_parameter('lookahead',   3)
        self.declare_parameter('goal_radius', 0.3)
        self.declare_parameter('loop',        False)
        self.declare_parameter('max_loops',   1)
 
        run_name = self.get_parameter('run_name').value
        if not run_name:
            self.get_logger().fatal('Parámetro run_name vacío. Usa -p run_name:=vuelta_01')
            raise RuntimeError('run_name requerido')
 
        self._k           = self.get_parameter('k').value
        self._k_soft      = self.get_parameter('k_soft').value
        self._speed       = self.get_parameter('speed').value
        self._speed_curve = self.get_parameter('speed_curve').value
        self._max_steer   = self.get_parameter('max_steer').value
        self._look        = self.get_parameter('lookahead').value
        self._goal_r      = self.get_parameter('goal_radius').value
        self._loop        = self.get_parameter('loop').value
        self._max_loops   = self.get_parameter('max_loops').value
 
        # ── Rutas estructuradas ────────────────────────────────────
        self._data_dir = resolve_data_dir()
        csv_path = os.path.join(self._data_dir, f'{run_name}.csv')
 
        if not os.path.exists(csv_path):
            self.get_logger().fatal(f'CSV no encontrado: {csv_path}')
            raise FileNotFoundError(csv_path)
 
        self._run_name = run_name
        self._waypoints = load_csv(csv_path)
        self._n = len(self._waypoints)
        if self._n < 2:
            raise RuntimeError('CSV necesita al menos 2 waypoints')
 
        self.get_logger().info(f'Trayectoria: {self._n} waypoints — {csv_path}')
 
        # ── Estado del controlador ─────────────────────────────────
        self._x       = 0.0
        self._y       = 0.0
        self._yaw     = 0.0
        self._v       = 0.0
        self._idx     = 0
        self._done    = False
        self._loops   = 0
        self._obstacle = False
        self._obs_paused = False
        self._obstacle_stops = 0
        self._start_time = time.time()
 
        # ── Registro de trayectoria real ───────────────────────────
        # Cada entrada: (x, y, yaw, tiempo_relativo)
        self._real_track: List[Tuple[float, float, float, float]] = []
        self._track_timer = self.create_timer(0.1, self._record_pose)
 
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
        self.get_logger().info(' STANLEY CONTROLLER — Neuracar')
        self.get_logger().info('=' * 52)
        self.get_logger().info(f'  k={self._k}  k_soft={self._k_soft}')
        self.get_logger().info(
            f'  speed={self._speed} m/s | curve={self._speed_curve} m/s')
        self.get_logger().info(
            f'  max_steer={math.degrees(self._max_steer):.1f}°')
        self.get_logger().info(
            f'  loop={self._loop}  max_loops={self._max_loops}')
        self.get_logger().info(
            f'  data_dir={self._data_dir}')
 
    # ── Callbacks ──────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        self._x   = msg.pose.pose.position.x
        self._y   = msg.pose.pose.position.y
        q         = msg.pose.pose.orientation
        self._yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
 
    def _vel_cb(self, msg: TwistStamped):
        self._v = msg.twist.linear.x
 
    def _obs_cb(self, msg: Bool):
        was_obstacle = self._obstacle
        self._obstacle = msg.data
        if self._obstacle and not was_obstacle:
            self._obstacle_stops += 1
            self.get_logger().warn(
                f'¡Obstáculo detectado! Parada #{self._obstacle_stops}')
        elif not self._obstacle and was_obstacle:
            self.get_logger().info('Obstáculo despejado — reanudando')
 
    def _record_pose(self):
        """Guarda pose actual cada 100 ms para el análisis post-prueba."""
        if self._done:
            return
        t = time.time() - self._start_time
        self._real_track.append((self._x, self._y, self._yaw, t))
 
    # ── Waypoint más cercano ───────────────────────────────────────
    def _nearest(self) -> int:
        best, min_d = self._idx, float('inf')
        for i in range(max(0, self._idx - 5),
                       min(self._n, self._idx + 50)):
            d = dist2d((self._x, self._y),
                       (self._waypoints[i][0], self._waypoints[i][1]))
            if d < min_d:
                min_d, best = d, i
        return best
 
    # ── Loop de control ────────────────────────────────────────────
    def _control_loop(self):
        # ── Parada por obstáculo ───────────────────────────────────
        if self._obstacle:
            if not self._obs_paused:
                self._obs_paused = True
            self._publish(0.0, 0.0)
            return
        self._obs_paused = False
 
        if self._done:
            self._publish(0.0, 0.0)
            return
 
        # ── Waypoint objetivo ──────────────────────────────────────
        nearest    = self._nearest()
        target_idx = min(nearest + self._look, self._n - 1)
        tx, ty, _  = self._waypoints[target_idx]
 
        # ── Verificar llegada al final ─────────────────────────────
        ex, ey, _ = self._waypoints[-1]
        if dist2d((self._x, self._y), (ex, ey)) < self._goal_r:
            self._loops += 1
            if self._loop and (self._max_loops == 0
                               or self._loops < self._max_loops):
                self._idx = 0
                self.get_logger().info(
                    f'Vuelta {self._loops} completada — reiniciando')
            else:
                self._done = True
                self.get_logger().info(
                    f'¡Meta alcanzada! Vueltas: {self._loops}')
                self._publish(0.0, 0.0)
                return
 
        self._idx = nearest
 
        # ── Error de heading ψ_e ──────────────────────────────────
        if target_idx < self._n - 1:
            nx, ny, _ = self._waypoints[target_idx + 1]
            path_hdg  = math.atan2(ny - ty, nx - tx)
        else:
            path_hdg  = self._waypoints[target_idx][2]
 
        psi_e = normalize(path_hdg - self._yaw)
 
        # ── Cross-track error e ───────────────────────────────────
        dx = tx - self._x
        dy = ty - self._y
        e  = -math.sin(self._yaw) * dx + math.cos(self._yaw) * dy
 
        # ── Ley Stanley ───────────────────────────────────────────
        v_eff    = abs(self._v) + self._k_soft
        cte_term = math.atan2(self._k * e, v_eff)
        steering = normalize(psi_e + cte_term)
        steering = max(-self._max_steer, min(steering, self._max_steer))
 
        speed = (self._speed_curve
                 if abs(steering) > 0.15 else self._speed)
 
        self.get_logger().info(
            f'wp={target_idx}/{self._n} | '
            f'ψ_e={math.degrees(psi_e):+.1f}° | '
            f'e={e:+.3f}m | '
            f'δ={math.degrees(steering):+.1f}° | '
            f'v={speed:.2f}m/s',
            throttle_duration_sec=0.3)
 
        self._publish(speed, steering)
 
    def _publish(self, speed: float, steering: float):
        msg = Vector3Stamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.vector.x        = float(speed)
        msg.vector.y        = float(steering)
        self._pub.publish(msg)
 
    def finalize(self):
        """Llama al análisis al terminar."""
        elapsed = time.time() - self._start_time
        compute_analysis(
            ref            = self._waypoints,
            real           = self._real_track,
            run_name       = self._run_name,
            data_dir       = self._data_dir,
            elapsed        = elapsed,
            obstacle_stops = self._obstacle_stops,
            loops_done     = self._loops,
        )
 
 
# ── main ──────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    try:
        node = StanleyController()
    except (RuntimeError, FileNotFoundError) as e:
        print(f'[ERROR] {e}')
        rclpy.shutdown()
        return
 
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish(0.0, 0.0)
        node.finalize()
        node.destroy_node()
        rclpy.shutdown()
        print('Stanley Controller detenido.')
 
 
if __name__ == '__main__':
    main()