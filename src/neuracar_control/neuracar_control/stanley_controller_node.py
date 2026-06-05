# stanley_controller_node.py — Neuracar v2.0
# Cambio v2.0: publica velocidad [m/s] en /neuracar/cmd_velocity
# y steering norm [-1,1] en /neuracar/cmd_steering en lugar de
# user_command directo. El velocity_pid_node convierte m/s → throttle.
# El resto de la lógica (Stanley, análisis, CSV) es idéntica a v1.0.

# ── PATCH sobre stanley_controller_node.py v1.0 ──────────────────
# Solo cambian:
#   1. El publisher (user_command → cmd_velocity + cmd_steering)
#   2. _publish()
#   3. Parámetro speed/speed_curve ahora son m/s reales (ya lo eran,
#      pero ahora el PID los respeta físicamente)
#
# Para aplicar: reemplazar este archivo por el original y aplicar
# los cambios marcados con # ← CAMBIO v2.0

import csv
import math
import os
import time
from datetime import datetime
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32   # ← CAMBIO v2.0: añadir Float32


Waypoint = Tuple[float, float, float]


def _base_data_dir() -> str:
    pkg_src = os.path.expanduser("~/Workspaces/Neuracar/src/neuracar_control")
    if os.path.isdir(pkg_src):
        return os.path.join(pkg_src, "data")
    try:
        from ament_index_python.packages import get_package_share_directory
        get_package_share_directory("neuracar_control")
        return os.path.join(os.path.expanduser("~/Workspaces/Neuracar/src"),
                            "neuracar_control", "data")
    except Exception:
        pass
    return os.path.expanduser("~/Workspaces/Neuracar/src/neuracar_control/data")


def resolve_data_dir() -> str:
    d = os.path.join(_base_data_dir(), "trajectories")
    os.makedirs(d, exist_ok=True)
    return d


def resolve_runs_dir() -> str:
    d = os.path.join(_base_data_dir(), "runs")
    os.makedirs(d, exist_ok=True)
    return d


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
    cosy = 1.0 - 2.0 * (qy**2 + qz**2)
    return math.atan2(siny, cosy)


def compute_analysis(ref, real, run_name, runs_dir, elapsed,
                     obstacle_stops, loops_done):
    if not real:
        return
    rows, cte_list = [], []
    for rx, ry, ryaw, rt in real:
        min_d, best_ref = float('inf'), ref[0]
        for wp in ref:
            d = dist2d((rx, ry), (wp[0], wp[1]))
            if d < min_d:
                min_d, best_ref = d, wp
        wx, wy, wtheta = best_ref
        dx, dy = rx - wx, ry - wy
        cte = -math.sin(wtheta) * dx + math.cos(wtheta) * dy
        cte_list.append(cte)
        rows.append({'time_s': round(rt, 3),
                     'ref_x': round(wx, 4), 'ref_y': round(wy, 4),
                     'ref_theta': round(wtheta, 4),
                     'real_x': round(rx, 4), 'real_y': round(ry, 4),
                     'real_theta': round(ryaw, 4),
                     'cte_m': round(cte, 4), 'dist_to_ref': round(min_d, 4)})

    n = len(cte_list)
    cte_abs = [abs(c) for c in cte_list]
    rms_cte = math.sqrt(sum(c**2 for c in cte_list) / n)
    max_cte = max(cte_abs)
    pct5  = sum(1 for c in cte_abs if c < 0.05) / n * 100
    pct10 = sum(1 for c in cte_abs if c < 0.10) / n * 100

    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f'{run_name}_{ts}'
    csv_out = os.path.join(runs_dir, f'{stem}_analysis.csv')
    with open(csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    quality = ('EXCELENTE' if rms_cte < 0.05 else
               'BUENO'     if rms_cte < 0.10 else
               'REGULAR'   if rms_cte < 0.20 else 'DEFICIENTE')

    report = os.path.join(runs_dir, f'{stem}_report.txt')
    with open(report, 'w') as f:
        f.write(f'Stanley Controller — Reporte de Prueba\n{"="*50}\n')
        f.write(f'Trayectoria : {run_name}\n')
        f.write(f'Fecha       : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Duración    : {elapsed:.1f} s\n')
        f.write(f'Vueltas     : {loops_done}\n')
        f.write(f'Paradas obs.: {obstacle_stops}\n\n')
        f.write(f'RMS CTE     : {rms_cte:.4f} m  → {quality}\n')
        f.write(f'CTE máximo  : {max_cte:.4f} m\n')
        f.write(f'% < 5 cm    : {pct5:.1f}%\n')
        f.write(f'% < 10 cm   : {pct10:.1f}%\n')
    print(f'Análisis guardado: {csv_out}')
    print(f'Reporte guardado : {report}')


class StanleyController(Node):

    def __init__(self):
        super().__init__('stanley_controller')

        self.declare_parameter('run_name',     '')
        self.declare_parameter('k',            0.5)
        self.declare_parameter('k_soft',       0.5)
        self.declare_parameter('speed',        0.3)    # m/s crucero
        self.declare_parameter('speed_curve',  0.2)    # m/s en curva
        self.declare_parameter('min_throttle', 0.0)    # m/s mínimo (el PID aplica min físico)
        self.declare_parameter('max_steer',    0.5)
        self.declare_parameter('lookahead',    3)
        self.declare_parameter('goal_radius',  0.3)
        self.declare_parameter('loop',         False)
        self.declare_parameter('max_loops',    1)

        self._run_name    = self.get_parameter('run_name').value
        self._k           = self.get_parameter('k').value
        self._k_soft      = self.get_parameter('k_soft').value
        self._speed       = self.get_parameter('speed').value
        self._speed_curve = self.get_parameter('speed_curve').value
        self._min_speed   = self.get_parameter('min_throttle').value
        self._max_steer   = self.get_parameter('max_steer').value
        self._look        = self.get_parameter('lookahead').value
        self._goal_r      = self.get_parameter('goal_radius').value
        self._loop        = self.get_parameter('loop').value
        self._max_loops   = self.get_parameter('max_loops').value

        if not self._run_name:
            raise RuntimeError('Parámetro run_name obligatorio.')

        self._data_dir = resolve_data_dir()
        self._runs_dir = resolve_runs_dir()
        csv_path = os.path.join(self._data_dir, f'{self._run_name}.csv')
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f'CSV no encontrado: {csv_path}')

        self._waypoints = load_csv(csv_path)
        self._n = len(self._waypoints)
        if self._n < 2:
            raise RuntimeError('CSV necesita al menos 2 waypoints')

        self._x = self._y = self._yaw = self._v = 0.0
        self._idx = 0
        self._obstacle = False
        self._obs_paused = False
        self._obstacle_stops = 0
        self._done = False
        self._loops = 0
        self._start_time = time.time()
        self._real_track = []

        # ── Publishers — ← CAMBIO v2.0 ──────────────────────────────
        # Ya NO publica en /neuracar/user_command directamente.
        # Publica velocidad m/s y steering norm por separado.
        # El velocity_pid_node los convierte a user_command con PID.
        self._pub_vel = self.create_publisher(
            Float32, '/neuracar/cmd_velocity', 10)   # ← CAMBIO v2.0
        self._pub_str = self.create_publisher(
            Float32, '/neuracar/cmd_steering', 10)   # ← CAMBIO v2.0

        # ── Subscribers ──────────────────────────────────────────────
        self.create_subscription(Odometry, '/neuracar/odometry', self._odom_cb, 10)
        self.create_subscription(TwistStamped, '/neuracar/velocity', self._vel_cb, 10)
        self.create_subscription(Bool, '/neuracar/lidar/obstacle_alert', self._obs_cb, 10)

        self.create_timer(0.1, self._record_pose)
        self.create_timer(0.05, self._control_loop)  # 20 Hz

        self.get_logger().info('=' * 52)
        self.get_logger().info(' STANLEY CONTROLLER v2.0 — con PID velocidad')
        self.get_logger().info('=' * 52)
        self.get_logger().info(f'  run={self._run_name}  {self._n} waypoints')
        self.get_logger().info(
            f'  k={self._k}  k_soft={self._k_soft}  max_steer={self._max_steer}rad')
        self.get_logger().info(
            f'  speed={self._speed} m/s  speed_curve={self._speed_curve} m/s')
        self.get_logger().info(
            '  → El PID de velocidad compensa la batería NiMH automáticamente')

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
            self._obstacle_stops += 1
            self.get_logger().warn(f'¡Obstáculo! Parada #{self._obstacle_stops}')
        elif not self._obstacle and was:
            self.get_logger().info('Obstáculo despejado — reanudando')

    def _record_pose(self):
        if not self._done:
            t = time.time() - self._start_time
            self._real_track.append((self._x, self._y, self._yaw, t))

    def _nearest(self) -> int:
        if self._n > 10:
            d_sample = dist2d((self._waypoints[0][0], self._waypoints[0][1]),
                               (self._waypoints[9][0], self._waypoints[9][1])) / 9.0
            v_max = max(abs(self._v), self._speed)
            wp_per_sec = v_max / d_sample if d_sample > 1e-4 else 50
        else:
            wp_per_sec = 50
        look_ahead   = max(600, int(wp_per_sec * 6))
        search_start = max(0, self._idx - 3)
        search_end   = min(self._n, self._idx + look_ahead)
        best, min_d  = self._idx, float('inf')
        for i in range(search_start, search_end):
            d = dist2d((self._x, self._y),
                       (self._waypoints[i][0], self._waypoints[i][1]))
            if d < min_d:
                min_d, best = d, i
        return best

    def _control_loop(self):
        if self._obstacle:
            self._obs_paused = True
            self._publish(0.0, 0.0)
            return
        self._obs_paused = False

        if self._done:
            self._publish(0.0, 0.0)
            return

        nearest    = self._nearest()
        target_idx = min(nearest + self._look, self._n - 1)
        tx, ty, _  = self._waypoints[target_idx]

        ex, ey, _ = self._waypoints[-1]
        if dist2d((self._x, self._y), (ex, ey)) < self._goal_r:
            self._loops += 1
            if self._loop and (self._max_loops == 0 or self._loops < self._max_loops):
                self._idx = 0
                self.get_logger().info(f'Vuelta {self._loops} — reiniciando')
            else:
                self._done = True
                self.get_logger().info(f'¡Meta! Vueltas: {self._loops}')
                self._publish(0.0, 0.0)
                return

        self._idx = nearest

        if target_idx < self._n - 1:
            nx, ny, _ = self._waypoints[target_idx + 1]
            path_hdg  = math.atan2(ny - ty, nx - tx)
        else:
            path_hdg = self._waypoints[target_idx][2]

        psi_e = normalize(path_hdg - self._yaw)
        dx = tx - self._x; dy = ty - self._y
        e  = -math.sin(self._yaw) * dx + math.cos(self._yaw) * dy

        v_eff     = abs(self._v) + self._k_soft
        cte_term  = math.atan2(self._k * e, v_eff)
        steer_rad = normalize(psi_e + cte_term)

        steering_norm = max(-1.0, min(1.0, steer_rad / self._max_steer))

        # Velocidad deseada en m/s — el PID la mantiene con batería descargada
        speed_ms = self._speed_curve if abs(steering_norm) > 0.3 else self._speed
        speed_ms = max(speed_ms, self._min_speed)

        self.get_logger().info(
            f'wp={target_idx}/{self._n} | psi_e={math.degrees(psi_e):+.1f}° | '
            f'e={e:+.3f}m | steer={steer_rad:+.3f}rad({steering_norm:+.3f}) | '
            f'v_sp={speed_ms:.3f}m/s',
            throttle_duration_sec=0.3)

        self._publish(speed_ms, steering_norm)

    def _publish(self, speed_ms: float, steering: float):
        # ← CAMBIO v2.0: publicar en dos topics separados para el PID
        try:
            v_msg = Float32(); v_msg.data = float(speed_ms)
            self._pub_vel.publish(v_msg)
            s_msg = Float32(); s_msg.data = float(steering)
            self._pub_str.publish(s_msg)
        except Exception:
            pass

    def finalize(self):
        elapsed = time.time() - self._start_time
        compute_analysis(ref=self._waypoints, real=self._real_track,
                         run_name=self._run_name, runs_dir=self._runs_dir,
                         elapsed=elapsed, obstacle_stops=self._obstacle_stops,
                         loops_done=self._loops)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = StanleyController()
    except (RuntimeError, FileNotFoundError) as e:
        print(f'[ERROR] {e}'); rclpy.shutdown(); return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'[WARN] {exc}')
    finally:
        node._publish(0.0, 0.0)
        print('\nGuardando análisis...')
        node.finalize()
        try: node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass
        print('Stanley Controller detenido.')


if __name__ == '__main__':
    main()