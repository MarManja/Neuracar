"""
pure_pursuit_node.py — NeuraCar
══════════════════════════════════════════════════════════════════
Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026

Pure Pursuit geometric path-tracking controller. Reads a pre-recorded
CSV of (x, y, theta) waypoints and computes steering commands at 20 Hz
using a fixed lookahead distance. Saves post-run analysis (CSV, TXT,
PNG) automatically on Ctrl+C.

  delta = atan2(2 * L * sin(alpha), Lf)

Subscriptions:
  /neuracar/odometry              nav_msgs/Odometry
  /neuracar/velocity              geometry_msgs/TwistStamped
  /neuracar/lidar/obstacle_alert  std_msgs/Bool

Publications:
  /neuracar/cmd_velocity  std_msgs/Float32   [m/s]
  /neuracar/cmd_steering  std_msgs/Float32   [-1, 1]
  /neuracar/path_reference nav_msgs/Path     reference waypoints
  /neuracar/path_real      nav_msgs/Path     measured trajectory

Parameters:
  run_name             (str,   vuelta_05_5cm): CSV trajectory (no extension)
  lookahead            (float, 0.40):  Lookahead distance Lf [m]
  wheelbase            (float, 0.256): Vehicle wheelbase L [m]
  speed                (float, 0.50):  Straight speed [m/s]
  speed_curve          (float, 0.55):  Curve speed [m/s]
  curve_steer_threshold(float, 0.35):  |steering| threshold for curve mode
  max_steer            (float, 0.50):  Max steering angle [rad]
  steering_sign        (float, -1.0):  Servo inversion correction
  nearest_back_steps   (int,   0):     Local search backward window
  nearest_fwd_steps    (int,   35):    Local search forward window
  monotonic_index      (bool,  true):  Prevent backward trajectory jumps
  loop                 (bool,  false): Repeat trajectory at goal
  max_loops            (int,   1):     Max laps (0 = infinite)
  goal_radius          (float, 0.25):  Goal detection radius [m]
  stop_on_obstacle     (bool,  false): true=end run, false=pause/resume
══════════════════════════════════════════════════════════════════
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
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float32

Waypoint = Tuple[float, float, float]
RealPoint = Tuple[float, float, float, float]


def resolve_data_dir() -> str:
    d = os.path.join(
        os.path.expanduser("~/Workspaces/Neuracar/src/neuracar_control"),
        "data", "trajectories"
    )
    os.makedirs(d, exist_ok=True)
    return d


def resolve_runs_dir() -> str:
    d = os.path.join(
        os.path.expanduser("~/Workspaces/Neuracar/src/neuracar_control"),
        "data", "runs_pure_pursuit"
    )
    os.makedirs(d, exist_ok=True)
    return d


def load_csv(path: str) -> List[Waypoint]:
    pts: List[Waypoint] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            pts.append((float(row['x']), float(row['y']), float(row['theta'])))
    return pts


def normalize(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def dist2d(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def yaw_from_quat(qx, qy, qz, qw) -> float:
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy ** 2 + qz ** 2)
    )


def cumulative_distances(ref: List[Waypoint]) -> List[float]:
    s = [0.0]
    for i in range(1, len(ref)):
        s.append(s[-1] + dist2d(ref[i - 1], ref[i]))
    return s


def compute_analysis(ref: List[Waypoint], real: List[RealPoint], run_name: str,
                     runs_dir: str, elapsed: float, obstacle_stops: int,
                     loops_done: int, finished_by_obstacle: bool) -> None:
    if not real:
        print('[WARN] No hay trayectoria real para analizar.')
        return

    rows = []
    cte_list = []

    for rx, ry, ryaw, rt in real:
        min_d = float('inf')
        best = ref[0]
        for wp in ref:
            d = dist2d((rx, ry), (wp[0], wp[1]))
            if d < min_d:
                min_d = d
                best = wp

        wx, wy, wt = best
        cte = -math.sin(wt) * (rx - wx) + math.cos(wt) * (ry - wy)
        cte_list.append(cte)
        rows.append({
            'time_s': round(rt, 3),
            'ref_x': round(wx, 4),
            'ref_y': round(wy, 4),
            'real_x': round(rx, 4),
            'real_y': round(ry, 4),
            'cte_m': round(cte, 4),
        })

    n = len(cte_list)
    cte_abs = [abs(c) for c in cte_list]
    rms = math.sqrt(sum(c * c for c in cte_list) / n)
    quality = ('EXCELENTE' if rms < 0.05 else
               'BUENO' if rms < 0.10 else
               'REGULAR' if rms < 0.20 else 'DEFICIENTE')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f'{run_name}_{ts}'

    csv_out = os.path.join(runs_dir, f'{stem}_analysis.csv')
    with open(csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    rpt = os.path.join(runs_dir, f'{stem}_report.txt')
    with open(rpt, 'w') as f:
        f.write(f'Pure Pursuit — Reporte\n{"=" * 40}\n')
        f.write(f'Trayectoria : {run_name}\n')
        f.write(f'Duración    : {elapsed:.1f} s\n')
        f.write(f'Vueltas     : {loops_done}\n')
        f.write(f'Paradas obs.: {obstacle_stops}\n')
        f.write(f'Terminó por obstáculo: {"sí" if finished_by_obstacle else "no"}\n\n')
        f.write(f'RMS CTE     : {rms:.4f} m  → {quality}\n')
        f.write(f'CTE máximo  : {max(cte_abs):.4f} m\n')
        f.write(f'% < 5 cm    : {sum(1 for c in cte_abs if c < 0.05) / n * 100:.1f}%\n')
        f.write(f'% < 10 cm   : {sum(1 for c in cte_abs if c < 0.10) / n * 100:.1f}%\n')

    png_out = os.path.join(runs_dir, f'{stem}_trayectoria.png')
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#0D1117')

        ax = axes[0]
        ax.set_facecolor('#161B22')
        ax.tick_params(colors='#8B949E')
        ax.xaxis.label.set_color('#8B949E')
        ax.yaxis.label.set_color('#8B949E')
        ax.title.set_color('#E6EDF3')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363D')

        ref_x = [wp[0] for wp in ref]
        ref_y = [wp[1] for wp in ref]
        real_x = [r[0] for r in real]
        real_y = [r[1] for r in real]

        ax.plot(ref_x, ref_y, '--', color='#3FB950', linewidth=1.5, label='Referencia')
        ax.plot(real_x, real_y, '-', color='#4C9EF0', linewidth=1.5, label='Real')
        ax.plot(ref_x[0], ref_y[0], 'o', color='#3FB950', markersize=8)
        ax.plot(ref_x[-1], ref_y[-1], 's', color='#3FB950', markersize=8)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'Trayectoria — {run_name}')
        ax.legend(facecolor='#161B22', labelcolor='#E6EDF3', edgecolor='#30363D')
        ax.set_aspect('equal')
        ax.grid(True, color='#30363D', alpha=0.5)

        ax2 = axes[1]
        ax2.set_facecolor('#161B22')
        ax2.tick_params(colors='#8B949E')
        ax2.xaxis.label.set_color('#8B949E')
        ax2.yaxis.label.set_color('#8B949E')
        ax2.title.set_color('#E6EDF3')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#30363D')

        times = [r[3] for r in real]
        ax2.plot(times, cte_list, color='#E3B341', linewidth=1)
        ax2.axhline(y=0.05, color='#3FB950', linestyle='--', alpha=0.6, label='±5 cm')
        ax2.axhline(y=-0.05, color='#3FB950', linestyle='--', alpha=0.6)
        ax2.axhline(y=0.10, color='#FF4444', linestyle='--', alpha=0.4, label='±10 cm')
        ax2.axhline(y=-0.10, color='#FF4444', linestyle='--', alpha=0.4)
        ax2.axhline(y=0.0, color='#30363D', linestyle='-', alpha=0.8)
        ax2.set_xlabel('Tiempo [s]')
        ax2.set_ylabel('CTE [m]')
        ax2.set_title(f'Error lateral — RMS={rms:.4f}m ({quality})')
        ax2.legend(facecolor='#161B22', labelcolor='#E6EDF3', edgecolor='#30363D')
        ax2.grid(True, color='#30363D', alpha=0.5)

        plt.tight_layout()
        plt.savefig(png_out, dpi=120, facecolor='#0D1117', bbox_inches='tight')
        plt.close(fig)
        print(f'PNG guardado: {png_out}')
    except Exception as e:
        print(f'[WARN] No se pudo generar PNG: {e}')

    print(f'Análisis guardado: {runs_dir}/{stem}')


class PurePursuitNode(Node):

    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.declare_parameter('run_name', 'vuelta_05_5cm')
        self.declare_parameter('wheelbase', 0.256)
        self.declare_parameter('lookahead', 0.40)
        self.declare_parameter('speed', 0.50)
        self.declare_parameter('speed_curve', 0.55)
        self.declare_parameter('min_straight_speed', 0.45)
        self.declare_parameter('min_curve_speed', 0.55)
        self.declare_parameter('curve_steer_threshold', 0.35)
        self.declare_parameter('max_steer', 0.5)
        self.declare_parameter('steering_sign', -1.0)
        self.declare_parameter('steering_cmd_gain', 1.0)
        self.declare_parameter('goal_radius', 0.25)
        self.declare_parameter('loop', False)
        self.declare_parameter('max_loops', 1)
        self.declare_parameter('nearest_back_steps', 0)
        self.declare_parameter('nearest_fwd_steps', 35)
        self.declare_parameter('monotonic_index', True)
        self.declare_parameter('stop_on_obstacle', False)

        run = str(self.get_parameter('run_name').value)
        if not run:
            raise RuntimeError('Parámetro run_name obligatorio.')

        self._L = float(self.get_parameter('wheelbase').value)
        self._Lf = float(self.get_parameter('lookahead').value)
        self._speed = float(self.get_parameter('speed').value)
        self._speed_c = float(self.get_parameter('speed_curve').value)
        self._min_straight = float(self.get_parameter('min_straight_speed').value)
        self._min_curve = float(self.get_parameter('min_curve_speed').value)
        self._curve_thr = float(self.get_parameter('curve_steer_threshold').value)
        self._max_steer = float(self.get_parameter('max_steer').value)
        raw_sign = float(self.get_parameter('steering_sign').value)
        self._steering_sign = -1.0 if raw_sign < 0.0 else 1.0
        self._steering_cmd_gain = float(self.get_parameter('steering_cmd_gain').value)
        self._goal_r = float(self.get_parameter('goal_radius').value)
        self._loop = bool(self.get_parameter('loop').value)
        self._max_loops = int(self.get_parameter('max_loops').value)
        self._back_steps = int(self.get_parameter('nearest_back_steps').value)
        self._fwd_steps = int(self.get_parameter('nearest_fwd_steps').value)
        self._monotonic_index = bool(self.get_parameter('monotonic_index').value)
        self._stop_on_obstacle = bool(self.get_parameter('stop_on_obstacle').value)
        self._run_name = run

        self._data_dir = resolve_data_dir()
        self._runs_dir = resolve_runs_dir()
        csv_path = os.path.join(self._data_dir, f'{run}.csv')
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f'CSV no encontrado: {csv_path}')

        self._waypoints = load_csv(csv_path)
        self._n = len(self._waypoints)
        if self._n < 2:
            raise RuntimeError('CSV necesita al menos 2 waypoints.')

        self._s = cumulative_distances(self._waypoints)
        self._path_len = self._s[-1]

        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._v = 0.0
        self._nearest_idx = 0
        self._obstacle = False
        self._obs_stops = 0
        self._done = False
        self._finished_by_obstacle = False
        self._loops = 0
        self._start_time = time.time()
        self._finish_elapsed = None
        self._real_track: List[RealPoint] = []
        self._analysis_saved = False
        self._ref_published = False

        self._pub_vel = self.create_publisher(Float32, '/neuracar/cmd_velocity', 10)
        self._pub_str = self.create_publisher(Float32, '/neuracar/cmd_steering', 10)
        self._pub_ref = self.create_publisher(Path, '/neuracar/path_reference', 10)
        self._pub_real = self.create_publisher(Path, '/neuracar/path_real', 10)

        self.create_subscription(Odometry, '/neuracar/odometry', self._odom_cb, 10)
        self.create_subscription(TwistStamped, '/neuracar/velocity', self._vel_cb, 10)
        self.create_subscription(Bool, '/neuracar/lidar/obstacle_alert', self._obs_cb, 10)

        self.create_timer(0.10, self._record_pose)
        self.create_timer(0.05, self._control_loop)
        self.create_timer(0.50, self._publish_paths)
        self.create_timer(1.00, self._publish_ref_periodic)

        self.get_logger().info('=' * 60)
        self.get_logger().info(' PURE PURSUIT v2.5 — baseline estático probado')
        self.get_logger().info('=' * 60)
        self.get_logger().info(
            f'  run={self._run_name} | {self._n} wp | longitud={self._path_len:.2f} m')
        self.get_logger().info(
            f'  Lf fijo={self._Lf:.2f} m | speed={self._speed:.2f} | curve={self._speed_c:.2f}')
        self.get_logger().info(
            f'  min_recta={self._min_straight:.2f} | min_curva={self._min_curve:.2f} | '
            f'curve_thr={self._curve_thr:.2f}')
        self.get_logger().info(
            f'  búsqueda local: -{self._back_steps} / +{self._fwd_steps} wp | '
            f'monotonic={self._monotonic_index}')
        self.get_logger().info(
            f'  steering_sign={self._steering_sign:+.0f} | '
            f'steering_cmd_gain={self._steering_cmd_gain:.2f}')
        self.get_logger().info(
            f'  LiDAR: {"termina prueba" if self._stop_on_obstacle else "pausa y reanuda"}')

    def _odom_cb(self, msg: Odometry):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

    def _vel_cb(self, msg: TwistStamped):
        self._v = msg.twist.linear.x

    def _obs_cb(self, msg: Bool):
        was = self._obstacle
        self._obstacle = bool(msg.data)

        if self._obstacle and not was:
            self._obs_stops += 1
            self.get_logger().warn(f'¡Obstáculo! #{self._obs_stops}')

            if self._stop_on_obstacle and not self._done:
                self._finished_by_obstacle = True
                self._done = True
                self._finish_elapsed = time.time() - self._start_time
                self._publish(0.0, 0.0)
                self.get_logger().warn(
                    'Prueba terminada por LiDAR. Usa Ctrl+C para guardar análisis.')

        elif not self._obstacle and was:
            if self._stop_on_obstacle:
                self.get_logger().info(
                    'Obstáculo despejado, pero la prueba ya fue terminada por seguridad.')
            else:
                self.get_logger().info('Despejado — reanudando')

    def _record_pose(self):
        if self._done:
            return
        self._real_track.append((self._x, self._y, self._yaw,
                                 time.time() - self._start_time))

    def _publish_ref_periodic(self):

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'odom'
        for x, y, theta in self._waypoints:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.0
            ps.pose.orientation.z = math.sin(theta / 2.0)
            ps.pose.orientation.w = math.cos(theta / 2.0)
            path.poses.append(ps)

        self._pub_ref.publish(path)
        self._ref_published = True

    def _publish_paths(self):
        if not self._real_track:
            return
        now = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp = now
        path.header.frame_id = 'odom'
        for x, y, yaw, _ in self._real_track:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.0
            ps.pose.orientation.z = math.sin(yaw / 2.0)
            ps.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(ps)
        self._pub_real.publish(path)

    def _find_nearest_local(self) -> int:
        start = max(0, self._nearest_idx - self._back_steps)
        end = min(self._n, self._nearest_idx + self._fwd_steps + 1)

        best = self._nearest_idx
        min_d = float('inf')
        for i in range(start, end):
            d = dist2d((self._x, self._y),
                       (self._waypoints[i][0], self._waypoints[i][1]))
            if d < min_d:
                min_d = d
                best = i

        if self._monotonic_index and best < self._nearest_idx:
            best = self._nearest_idx

        return best

    def _find_target_by_arclength(self, nearest: int):
        target_s = self._s[nearest] + self._Lf

        if self._loop and self._path_len > 1e-6:
            target_s = target_s % self._path_len
            search_start = 0 if target_s < self._s[nearest] else nearest
        else:
            target_s = min(target_s, self._s[-1])
            search_start = nearest

        t = search_start
        while t < self._n - 1 and self._s[t] < target_s:
            t += 1

        return self._waypoints[t][0], self._waypoints[t][1], t

    def _speed_for_steering(self, steer_cmd: float) -> float:
        if abs(steer_cmd) >= self._curve_thr:
            return max(self._speed_c, self._min_curve)
        return max(self._speed, self._min_straight)

    def _control_loop(self):
        if self._obstacle:
            self._publish(0.0, 0.0)
            return

        if self._done:
            self._publish(0.0, 0.0)
            return

        ex, ey, _ = self._waypoints[-1]
        near_goal = dist2d((self._x, self._y), (ex, ey)) < self._goal_r
        idx_near_end = self._nearest_idx >= max(0, self._n - 8)
        if near_goal and idx_near_end:
            self._loops += 1
            if self._loop and (self._max_loops == 0 or self._loops < self._max_loops):
                self._nearest_idx = 0
                self.get_logger().info(f'Vuelta {self._loops} — reiniciando')
            else:
                self._done = True
                self._finish_elapsed = time.time() - self._start_time
                self.get_logger().info(f'¡Meta! Vueltas: {self._loops}')
                self._publish(0.0, 0.0)
                return

        nearest = self._find_nearest_local()
        self._nearest_idx = nearest
        tx, ty, tidx = self._find_target_by_arclength(nearest)

        alpha = normalize(math.atan2(ty - self._y, tx - self._x) - self._yaw)
        steer_rad = math.atan2(
            2.0 * self._L * math.sin(alpha),
            max(self._Lf, 1e-3)
        )
        steer_raw = clamp(steer_rad / self._max_steer, -1.0, 1.0)
        steer_cmd = clamp(
            self._steering_sign * self._steering_cmd_gain * steer_raw,
            -1.0, 1.0
        )
        speed_ms = self._speed_for_steering(steer_cmd)

        self.get_logger().info(
            f'wp={nearest}->{tidx}/{self._n} Lf={self._Lf:.2f}m | '
            f'alpha={math.degrees(alpha):+.1f}° | '
            f'delta={steer_rad:+.3f}rad(raw={steer_raw:+.3f}, cmd={steer_cmd:+.3f}) | '
            f'v_sp={speed_ms:.3f}m/s',
            throttle_duration_sec=0.3
        )

        self._publish(speed_ms, steer_cmd)

    def _publish(self, speed_ms: float, steering: float):
        try:
            v = Float32()
            v.data = float(speed_ms)
            self._pub_vel.publish(v)

            s = Float32()
            s.data = float(steering)
            self._pub_str.publish(s)
        except Exception:
            pass

    def finalize(self):
        if self._analysis_saved:
            return
        self._analysis_saved = True
        elapsed = (self._finish_elapsed if self._finish_elapsed is not None
                   else time.time() - self._start_time)
        compute_analysis(
            ref=self._waypoints,
            real=self._real_track,
            run_name=self._run_name,
            runs_dir=self._runs_dir,
            elapsed=elapsed,
            obstacle_stops=self._obs_stops,
            loops_done=self._loops,
            finished_by_obstacle=self._finished_by_obstacle,
        )


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
        print(f'[WARN] {exc}')
    finally:
        try:
            node._publish(0.0, 0.0)
        except Exception:
            pass
        print('\nGuardando análisis...')
        try:
            node.finalize()
        except Exception as exc:
            print(f'[WARN] No se pudo guardar análisis: {exc}')
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