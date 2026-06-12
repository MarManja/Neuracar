"""
stanley_controller_node.py — NeuraCar
══════════════════════════════════════════════════════════════════
Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026

Stanley path-tracking controller. Minimizes combined heading error
and cross-track error from the front axle center. Saves post-run
analysis (CSV, TXT, PNG) automatically on Ctrl+C.

  delta = heading_error + atan2(k * cte, |v| + k_soft)

Subscriptions:
  /neuracar/odometry              nav_msgs/Odometry
  /neuracar/velocity              geometry_msgs/TwistStamped
  /neuracar/lidar/obstacle_alert  std_msgs/Bool

Publications:
  /neuracar/cmd_velocity   std_msgs/Float32   [m/s]
  /neuracar/cmd_steering   std_msgs/Float32   [-1, 1]
  /neuracar/path_reference nav_msgs/Path      reference waypoints
  /neuracar/path_real      nav_msgs/Path      measured trajectory

Parameters:
  run_name             (str,   vuelta_05_5cm): CSV trajectory (no extension)
  k                    (float, 0.80):  Cross-track error gain
  k_soft               (float, 0.50):  Low-speed softening factor
  heading_lookahead    (float, 0.30):  Path heading preview distance [m]
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
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
from geometry_msgs.msg import TwistStamped, PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float32

Waypoint = Tuple[float, float, float]

def _base_data_dir() -> str:
    pkg_src = os.path.expanduser('~/Workspaces/Neuracar/src/neuracar_control')
    if os.path.isdir(pkg_src):
        return os.path.join(pkg_src, 'data')

    try:
        from ament_index_python.packages import get_package_share_directory
        get_package_share_directory('neuracar_control')
        return os.path.join(
            os.path.expanduser('~/Workspaces/Neuracar/src'),
            'neuracar_control', 'data'
        )
    except Exception:
        pass

    return os.path.expanduser('~/Workspaces/Neuracar/src/neuracar_control/data')

def resolve_data_dir() -> str:
    d = os.path.join(_base_data_dir(), 'trajectories')
    os.makedirs(d, exist_ok=True)
    return d

def resolve_runs_dir() -> str:
    d = os.path.join(_base_data_dir(), 'runs_stanley')
    os.makedirs(d, exist_ok=True)
    return d

def load_csv(path: str) -> List[Waypoint]:
    pts: List[Waypoint] = []
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            pts.append((float(row['x']), float(row['y']), float(row['theta'])))
    return pts

def normalize(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a

def dist2d(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])

def yaw_from_quat(qx, qy, qz, qw) -> float:
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy ** 2 + qz ** 2)
    )

def cumulative_lengths(wps: List[Waypoint]) -> List[float]:
    s = [0.0]
    for i in range(1, len(wps)):
        s.append(s[-1] + dist2d((wps[i-1][0], wps[i-1][1]),
                                (wps[i][0], wps[i][1])))
    return s

def path_heading(wps: List[Waypoint], idx: int, heading_idx: int) -> float:
    idx = max(0, min(idx, len(wps) - 1))
    heading_idx = max(0, min(heading_idx, len(wps) - 1))

    if heading_idx != idx:
        x0, y0, _ = wps[idx]
        x1, y1, _ = wps[heading_idx]
        if dist2d((x0, y0), (x1, y1)) > 1e-6:
            return math.atan2(y1 - y0, x1 - x0)

    if idx < len(wps) - 1:
        x0, y0, _ = wps[idx]
        x1, y1, _ = wps[idx + 1]
        return math.atan2(y1 - y0, x1 - x0)
    return wps[idx][2]

def advance_by_distance(s_map: List[float], start_idx: int, ds: float) -> int:
    target_s = s_map[start_idx] + max(0.0, ds)
    i = start_idx
    while i < len(s_map) - 1 and s_map[i] < target_s:
        i += 1
    return i

def compute_analysis(ref: List[Waypoint], real, run_name: str, runs_dir: str,
                     elapsed: float, obstacle_stops: int, loops_done: int,
                     ended_by_obstacle: bool):
    if not real:
        return

    rows = []
    cte_list = []

    for rx, ry, ryaw, rt in real:
        min_d, best = float('inf'), ref[0]
        for wp in ref:
            d = dist2d((rx, ry), (wp[0], wp[1]))
            if d < min_d:
                min_d, best = d, wp

        wx, wy, wt = best
        dx, dy = rx - wx, ry - wy
        cte = -math.sin(wt) * dx + math.cos(wt) * dy
        cte_list.append(cte)

        rows.append({
            'time_s': round(rt, 3),
            'ref_x': round(wx, 4),
            'ref_y': round(wy, 4),
            'ref_theta': round(wt, 4),
            'real_x': round(rx, 4),
            'real_y': round(ry, 4),
            'real_theta': round(ryaw, 4),
            'cte_m': round(cte, 4),
            'dist_to_ref': round(min_d, 4),
        })

    n = len(cte_list)
    cte_abs = [abs(c) for c in cte_list]
    rms = math.sqrt(sum(c * c for c in cte_list) / n)
    max_cte = max(cte_abs)
    pct5 = sum(1 for c in cte_abs if c < 0.05) / n * 100.0
    pct10 = sum(1 for c in cte_abs if c < 0.10) / n * 100.0

    quality = ('EXCELENTE' if rms < 0.05 else
               'BUENO'     if rms < 0.10 else
               'REGULAR'   if rms < 0.20 else 'DEFICIENTE')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f'{run_name}_{ts}_stanley'

    # CSV
    csv_out = os.path.join(runs_dir, f'{stem}_analysis.csv')
    with open(csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    rpt = os.path.join(runs_dir, f'{stem}_report.txt')
    with open(rpt, 'w') as f:
        f.write(f'Stanley Controller — Reporte\n{"="*40}\n')
        f.write(f'Trayectoria : {run_name}\n')
        f.write(f'Duración    : {elapsed:.1f} s\n')
        f.write(f'Vueltas     : {loops_done}\n')
        f.write(f'Paradas obs.: {obstacle_stops}\n')
        f.write(f'Terminó por obstáculo: {"sí" if ended_by_obstacle else "no"}\n\n')
        f.write(f'RMS CTE     : {rms:.4f} m  → {quality}\n')
        f.write(f'CTE máximo  : {max_cte:.4f} m\n')
        f.write(f'% < 5 cm    : {pct5:.1f}%\n')
        f.write(f'% < 10 cm   : {pct10:.1f}%\n')

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

        ref_x = [p[0] for p in ref]
        ref_y = [p[1] for p in ref]
        real_x = [p[0] for p in real]
        real_y = [p[1] for p in real]

        ax.plot(ref_x, ref_y, '--', color='#3FB950', linewidth=1.5,
                label='Referencia')
        ax.plot(real_x, real_y, '-', color='#4C9EF0', linewidth=1.5,
                label='Real')
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

        times = [p[3] for p in real[:len(cte_list)]]
        ax2.plot(times, cte_list, color='#E3B341', linewidth=1)
        ax2.axhline(0.05, color='#3FB950', linestyle='--', alpha=0.6, label='±5 cm')
        ax2.axhline(-0.05, color='#3FB950', linestyle='--', alpha=0.6)
        ax2.axhline(0.10, color='#FF4444', linestyle='--', alpha=0.4, label='±10 cm')
        ax2.axhline(-0.10, color='#FF4444', linestyle='--', alpha=0.4)
        ax2.axhline(0.0, color='#30363D', linestyle='-', alpha=0.8)
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

class StanleyControllerNode(Node):

    def __init__(self):
        super().__init__('stanley_controller')

        self.declare_parameter('run_name', 'vuelta_05_5cm')

        # Geometría / Stanley
        self.declare_parameter('wheelbase', 0.256)
        self.declare_parameter('front_axle_offset', 0.0)
        self.declare_parameter('k', 0.80)
        self.declare_parameter('k_soft', 0.50)
        self.declare_parameter('cte_sign', 1.0)
        self.declare_parameter('heading_lookahead', 0.30)  
        self.declare_parameter('max_steer', 0.50)         

        self.declare_parameter('speed', 0.50)
        self.declare_parameter('speed_curve', 0.55)
        self.declare_parameter('min_straight_speed', 0.45)
        self.declare_parameter('min_curve_speed', 0.55)
        self.declare_parameter('curve_steer_threshold', 0.35)

        self.declare_parameter('nearest_back_steps', 0)
        self.declare_parameter('nearest_fwd_steps', 35)
        self.declare_parameter('monotonic_index', True)

        self.declare_parameter('steering_sign', -1.0)
        self.declare_parameter('steering_cmd_gain', 1.0)

        self.declare_parameter('goal_radius', 0.25)
        self.declare_parameter('loop', False)
        self.declare_parameter('max_loops', 1)

        self.declare_parameter('stop_on_obstacle', False)

        self._run_name = str(self.get_parameter('run_name').value)
        self._L = float(self.get_parameter('wheelbase').value)
        self._front_offset = float(self.get_parameter('front_axle_offset').value)
        self._k = float(self.get_parameter('k').value)
        self._k_soft = float(self.get_parameter('k_soft').value)
        self._cte_sign = float(self.get_parameter('cte_sign').value)
        self._heading_lookahead = float(self.get_parameter('heading_lookahead').value)
        self._max_steer = float(self.get_parameter('max_steer').value)

        self._speed = float(self.get_parameter('speed').value)
        self._speed_curve = float(self.get_parameter('speed_curve').value)
        self._min_straight = float(self.get_parameter('min_straight_speed').value)
        self._min_curve = float(self.get_parameter('min_curve_speed').value)
        self._curve_thr = float(self.get_parameter('curve_steer_threshold').value)

        self._back_steps = int(self.get_parameter('nearest_back_steps').value)
        self._fwd_steps = int(self.get_parameter('nearest_fwd_steps').value)
        self._monotonic = bool(self.get_parameter('monotonic_index').value)

        self._steering_sign = float(self.get_parameter('steering_sign').value)
        self._steering_gain = float(self.get_parameter('steering_cmd_gain').value)

        self._goal_r = float(self.get_parameter('goal_radius').value)
        self._loop = bool(self.get_parameter('loop').value)
        self._max_loops = int(self.get_parameter('max_loops').value)
        self._stop_on_obstacle = bool(self.get_parameter('stop_on_obstacle').value)

        if not self._run_name:
            raise RuntimeError('run_name no puede estar vacío.')

        self._data_dir = resolve_data_dir()
        self._runs_dir = resolve_runs_dir()
        csv_path = os.path.join(self._data_dir, f'{self._run_name}.csv')
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f'CSV no encontrado: {csv_path}')

        self._waypoints = load_csv(csv_path)
        self._n = len(self._waypoints)
        if self._n < 2:
            raise RuntimeError('CSV necesita al menos 2 waypoints.')

        self._s_map = cumulative_lengths(self._waypoints)
        self._path_len = self._s_map[-1]

        # Estado
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._v = 0.0
        self._nearest_idx = 0
        self._obstacle = False
        self._obs_stops = 0
        self._ended_by_obstacle = False
        self._done = False
        self._loops = 0
        self._start_time = time.time()
        self._real_track = []
        self._ref_published = False

        # Publishers
        self._pub_vel = self.create_publisher(Float32, '/neuracar/cmd_velocity', 10)
        self._pub_str = self.create_publisher(Float32, '/neuracar/cmd_steering', 10)
        qos_path_ref = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._pub_ref = self.create_publisher(Path, '/neuracar/path_reference', qos_path_ref)
        self._pub_real = self.create_publisher(Path, '/neuracar/path_real', 10)

        self._pub_user = self.create_publisher(Vector3Stamped, '/neuracar/user_command', 10)

        # Subscribers
        self.create_subscription(Odometry, '/neuracar/odometry', self._odom_cb, 10)
        self.create_subscription(TwistStamped, '/neuracar/velocity', self._vel_cb, 10)
        self.create_subscription(Bool, '/neuracar/lidar/obstacle_alert', self._obs_cb, 10)

        self.create_timer(0.10, self._record_pose)
        self.create_timer(0.05, self._control_loop)    # 20 Hz
        self.create_timer(0.50, self._publish_real_path)
        self.create_timer(1.00, self._publish_ref_once)

        self.get_logger().info('=' * 60)
        self.get_logger().info(' STANLEY CONTROLLER ')
        self.get_logger().info('=' * 60)
        self.get_logger().info(
            f'  run={self._run_name} | {self._n} wp | longitud={self._path_len:.2f} m')
        self.get_logger().info(
            f'  k={self._k:.2f} k_soft={self._k_soft:.2f} '
            f'heading_L={self._heading_lookahead:.2f} m max_steer={self._max_steer:.2f} rad')
        self.get_logger().info(
            f'  speed={self._speed:.2f} curve={self._speed_curve:.2f} '
            f'min_recta={self._min_straight:.2f} min_curva={self._min_curve:.2f}')
        self.get_logger().info(
            f'  búsqueda local: -{self._back_steps} / +{self._fwd_steps} wp | '
            f'monotonic={self._monotonic}')
        self.get_logger().info(
            f'  steering_sign={self._steering_sign:+.1f} gain={self._steering_gain:.2f} '
            f'cte_sign={self._cte_sign:+.1f}')
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
            self._publish(0.0, 0.0)
            if self._stop_on_obstacle:
                self._ended_by_obstacle = True
                self._done = True
                self.get_logger().warn(
                    'Prueba terminada por LiDAR. Usa Ctrl+C para guardar análisis.')

        elif not self._obstacle and was:
            if self._ended_by_obstacle:
                self.get_logger().info(
                    'Obstáculo despejado, pero la prueba ya fue terminada.')
            else:
                self.get_logger().info('Obstáculo despejado — reanudando')

    def _record_pose(self):
        if not self._done:
            self._real_track.append((
                self._x, self._y, self._yaw,
                time.time() - self._start_time
            ))

    def _publish_ref_once(self):

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

    def _publish_real_path(self):
        if not self._real_track:
            return
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
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

    def _control_point(self):

        fx = self._x + self._front_offset * math.cos(self._yaw)
        fy = self._y + self._front_offset * math.sin(self._yaw)
        return fx, fy

    def _find_nearest_local(self, px: float, py: float) -> int:
        start = max(0, self._nearest_idx - self._back_steps)
        end = min(self._n, self._nearest_idx + self._fwd_steps + 1)

        best = self._nearest_idx
        min_d = float('inf')
        for i in range(start, end):
            d = dist2d((px, py), (self._waypoints[i][0], self._waypoints[i][1]))
            if d < min_d:
                min_d = d
                best = i

        if self._monotonic:
            best = max(best, self._nearest_idx)

        return best

    def _control_loop(self):
        if self._obstacle:
            self._publish(0.0, 0.0)
            return

        if self._done:
            self._publish(0.0, 0.0)
            return

        # Meta
        ex, ey, _ = self._waypoints[-1]
        if dist2d((self._x, self._y), (ex, ey)) < self._goal_r:
            self._loops += 1
            if self._loop and (self._max_loops == 0 or self._loops < self._max_loops):
                self._nearest_idx = 0
                self.get_logger().info(f'Vuelta {self._loops} — reiniciando')
            else:
                self._done = True
                self.get_logger().info(f'¡Meta! Vueltas: {self._loops}')
                self._publish(0.0, 0.0)
                return

        px, py = self._control_point()
        nearest = self._find_nearest_local(px, py)
        self._nearest_idx = nearest

        heading_idx = advance_by_distance(
            self._s_map, nearest, self._heading_lookahead)
        wx, wy, _ = self._waypoints[nearest]
        path_hdg = path_heading(self._waypoints, nearest, heading_idx)

        psi_e = normalize(path_hdg - self._yaw)

        dx = px - wx
        dy = py - wy
        cte = (-math.sin(path_hdg) * dx + math.cos(path_hdg) * dy) * self._cte_sign

        v_eff = abs(self._v) + self._k_soft
        cte_term = math.atan2(self._k * cte, max(v_eff, 1e-3))
        steer_rad = normalize(psi_e - cte_term)

        steer_internal = max(-1.0, min(1.0, steer_rad / self._max_steer))
        steer_cmd = self._steering_sign * self._steering_gain * steer_internal
        steer_cmd = max(-1.0, min(1.0, steer_cmd))

        if abs(steer_cmd) >= self._curve_thr:
            speed_ms = max(self._speed_curve, self._min_curve)
        else:
            speed_ms = max(self._speed, self._min_straight)

        self.get_logger().info(
            f'wp={nearest}->{heading_idx}/{self._n} | '
            f'ψe={math.degrees(psi_e):+.1f}° | '
            f'cte={cte:+.3f}m | cte_term={math.degrees(cte_term):+.1f}° | '
            f'δ={steer_rad:+.3f}rad int={steer_internal:+.3f} cmd={steer_cmd:+.3f} | '
            f'v_sp={speed_ms:.3f}m/s',
            throttle_duration_sec=0.3
        )

        self._publish(speed_ms, steer_cmd)

    def _publish(self, speed_ms: float, steering: float):
        try:
            v_msg = Float32()
            v_msg.data = float(speed_ms)
            self._pub_vel.publish(v_msg)

            s_msg = Float32()
            s_msg.data = float(steering)
            self._pub_str.publish(s_msg)
        except Exception:
            pass

    def _publish_user_zero(self):

        try:
            msg = Vector3Stamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.vector.x = 0.0  # throttle
            msg.vector.y = 0.0  # steering
            msg.vector.z = 0.0
            self._pub_user.publish(msg)
        except Exception:
            pass

    def hard_stop(self, duration_s: float = 0.8, rate_hz: float = 25.0):

        period = 1.0 / max(rate_hz, 1.0)
        end_t = time.time() + max(duration_s, 0.1)
        while time.time() < end_t:
            self._publish(0.0, 0.0)
            self._publish_user_zero()
            try:
                rclpy.spin_once(self, timeout_sec=0.0)
            except Exception:
                pass
            time.sleep(period)

    def finalize(self):
        compute_analysis(
            ref=self._waypoints,
            real=self._real_track,
            run_name=self._run_name,
            runs_dir=self._runs_dir,
            elapsed=time.time() - self._start_time,
            obstacle_stops=self._obs_stops,
            loops_done=self._loops,
            ended_by_obstacle=self._ended_by_obstacle,
        )


def main(args=None):
    rclpy.init(args=args)
    try:
        node = StanleyControllerNode()
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
            node.hard_stop()
        except Exception:
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
        print('Stanley Controller detenido.')


if __name__ == '__main__':
    main()