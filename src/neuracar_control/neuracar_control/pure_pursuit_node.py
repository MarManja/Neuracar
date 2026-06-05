#!/usr/bin/env python3
"""
pure_pursuit_node.py — Neuracar v2.0
Cambio v2.0: publica en /neuracar/cmd_velocity [m/s] y
/neuracar/cmd_steering [-1,1] en lugar de user_command directo.
El velocity_pid_node convierte m/s → throttle compensando batería NiMH.
Lógica Pure Pursuit idéntica a v1.0.
"""

import csv, math, os, time
from datetime import datetime
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32   # ← CAMBIO v2.0

Waypoint = Tuple[float, float, float]


def resolve_data_dir() -> str:
    d = os.path.join(os.path.expanduser(
        "~/Workspaces/Neuracar/src/neuracar_control"), "data", "trajectories")
    os.makedirs(d, exist_ok=True); return d


def resolve_runs_dir() -> str:
    d = os.path.join(os.path.expanduser(
        "~/Workspaces/Neuracar/src/neuracar_control"), "data", "runs")
    os.makedirs(d, exist_ok=True); return d


def load_csv(path):
    pts = []
    with open(path) as f:
        for row in csv.DictReader(f):
            pts.append((float(row['x']), float(row['y']), float(row['theta'])))
    return pts


def normalize(a):
    while a > math.pi:  a -= 2*math.pi
    while a <= -math.pi: a += 2*math.pi
    return a


def dist2d(p, q): return math.hypot(p[0]-q[0], p[1]-q[1])


def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2))


def compute_analysis(ref, real, run_name, runs_dir, elapsed, obstacle_stops, loops_done):
    if not real: return
    rows, cte_list = [], []
    for rx, ry, ryaw, rt in real:
        min_d, best = float('inf'), ref[0]
        for wp in ref:
            d = dist2d((rx,ry),(wp[0],wp[1]))
            if d < min_d: min_d, best = d, wp
        wx, wy, wt = best
        cte = -math.sin(wt)*(rx-wx) + math.cos(wt)*(ry-wy)
        cte_list.append(cte)
        rows.append({'time_s':round(rt,3),'ref_x':round(wx,4),'ref_y':round(wy,4),
                     'real_x':round(rx,4),'real_y':round(ry,4),'cte_m':round(cte,4)})
    n = len(cte_list); cte_abs = [abs(c) for c in cte_list]
    rms = math.sqrt(sum(c**2 for c in cte_list)/n)
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f'{run_name}_{ts}'
    with open(os.path.join(runs_dir, f'{stem}_analysis.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    quality = ('EXCELENTE' if rms<0.05 else 'BUENO' if rms<0.10 else
               'REGULAR' if rms<0.20 else 'DEFICIENTE')
    with open(os.path.join(runs_dir, f'{stem}_report.txt'), 'w') as f:
        f.write(f'Pure Pursuit — Reporte\n{"="*40}\n')
        f.write(f'Trayectoria: {run_name}\nDuración: {elapsed:.1f}s\n')
        f.write(f'RMS CTE: {rms:.4f}m → {quality}\n')
        f.write(f'CTE máx: {max(cte_abs):.4f}m\n')
    print(f'Análisis guardado: {runs_dir}/{stem}')


class PurePursuitNode(Node):

    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.declare_parameter('run_name',     '')
        self.declare_parameter('wheelbase',    0.256)
        self.declare_parameter('lookahead',    0.20)
        self.declare_parameter('k_gain',       0.5)
        self.declare_parameter('speed',        0.3)    # m/s crucero ← CAMBIO v2.0
        self.declare_parameter('speed_curve',  0.2)    # m/s en curva ← CAMBIO v2.0
        self.declare_parameter('min_throttle', 0.0)    # m/s mínimo
        self.declare_parameter('max_steer',    0.5)
        self.declare_parameter('goal_radius',  0.05)
        self.declare_parameter('loop',         False)
        self.declare_parameter('max_loops',    1)

        run = self.get_parameter('run_name').value
        if not run: raise RuntimeError('Parámetro run_name obligatorio.')

        self._L         = self.get_parameter('wheelbase').value
        self._Lfc       = self.get_parameter('lookahead').value
        self._k_gain    = self.get_parameter('k_gain').value
        self._speed     = self.get_parameter('speed').value
        self._speed_c   = self.get_parameter('speed_curve').value
        self._min_spd   = self.get_parameter('min_throttle').value
        self._max_steer = self.get_parameter('max_steer').value
        self._goal_r    = self.get_parameter('goal_radius').value
        self._loop      = self.get_parameter('loop').value
        self._max_loops = self.get_parameter('max_loops').value
        self._run_name  = run

        self._data_dir = resolve_data_dir()
        self._runs_dir = resolve_runs_dir()
        csv_path = os.path.join(self._data_dir, f'{run}.csv')
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f'CSV no encontrado: {csv_path}')

        self._waypoints = load_csv(csv_path)
        self._n = len(self._waypoints)
        if self._n < 2: raise RuntimeError('CSV necesita al menos 2 waypoints')

        self._x = self._y = self._yaw = self._v = 0.0
        self._nearest_idx = 0
        self._obstacle = False
        self._obs_stops = 0
        self._done = False
        self._loops = 0
        self._start_time = time.time()
        self._real_track = []

        # ── Publishers — ← CAMBIO v2.0 ──────────────────────────────
        self._pub_vel = self.create_publisher(Float32, '/neuracar/cmd_velocity', 10)
        self._pub_str = self.create_publisher(Float32, '/neuracar/cmd_steering', 10)

        self.create_subscription(Odometry,     '/neuracar/odometry',             self._odom_cb, 10)
        self.create_subscription(TwistStamped, '/neuracar/velocity',             self._vel_cb,  10)
        self.create_subscription(Bool,         '/neuracar/lidar/obstacle_alert', self._obs_cb,  10)

        self.create_timer(0.1,  self._record_pose)
        self.create_timer(0.05, self._control_loop)  # 20 Hz

        self.get_logger().info('=' * 52)
        self.get_logger().info(' PURE PURSUIT v2.0 — con PID velocidad')
        self.get_logger().info('=' * 52)
        self.get_logger().info(f'  {self._n} wp | speed={self._speed}m/s | curve={self._speed_c}m/s')
        self.get_logger().info('  → PID compensa batería NiMH automáticamente')

    def _odom_cb(self, msg):
        self._x = msg.pose.pose.position.x; self._y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

    def _vel_cb(self, msg): self._v = msg.twist.linear.x

    def _obs_cb(self, msg):
        was = self._obstacle; self._obstacle = msg.data
        if self._obstacle and not was:
            self._obs_stops += 1
            self.get_logger().warn(f'¡Obstáculo! #{self._obs_stops}')
        elif not self._obstacle and was:
            self.get_logger().info('Despejado — reanudando')

    def _record_pose(self):
        if not self._done:
            self._real_track.append((self._x, self._y, self._yaw,
                                     time.time() - self._start_time))

    def _find_nearest(self):
        d_sample = (dist2d((self._waypoints[0][0], self._waypoints[0][1]),
                           (self._waypoints[9][0], self._waypoints[9][1])) / 9.0
                    if self._n > 10 else 0.01)
        v_max = max(abs(self._v), self._speed)
        wps   = v_max / d_sample if d_sample > 1e-4 else 50
        look  = max(600, int(wps * 6))
        start = max(0, self._nearest_idx - 3)
        end   = min(self._n, self._nearest_idx + look)
        best, min_d = self._nearest_idx, float('inf')
        for i in range(start, end):
            d = dist2d((self._x,self._y),(self._waypoints[i][0],self._waypoints[i][1]))
            if d < min_d: min_d, best = d, i
        return best

    def _find_target(self, nearest):
        Lf = max(self._k_gain * max(abs(self._v), 0.0) + self._Lfc, self._Lfc)
        t = nearest
        while t < self._n - 1:
            if dist2d((self._x,self._y),(self._waypoints[t][0],self._waypoints[t][1])) >= Lf:
                break
            t += 1
        return self._waypoints[t][0], self._waypoints[t][1], t, Lf

    def _control_loop(self):
        if self._obstacle: self._publish(0.0, 0.0); return
        if self._done:     self._publish(0.0, 0.0); return

        ex, ey, _ = self._waypoints[-1]
        if dist2d((self._x,self._y),(ex,ey)) < self._goal_r:
            self._loops += 1
            if self._loop and (self._max_loops == 0 or self._loops < self._max_loops):
                self._nearest_idx = 0
                self.get_logger().info(f'Vuelta {self._loops} — reiniciando')
            else:
                self._done = True
                self.get_logger().info(f'¡Meta! Vueltas: {self._loops}')
                self._publish(0.0, 0.0); return

        nearest = self._find_nearest()
        self._nearest_idx = nearest
        tx, ty, tidx, Lf = self._find_target(nearest)

        alpha     = normalize(math.atan2(ty-self._y, tx-self._x) - self._yaw)
        steer_rad = math.atan2(2.0*self._L*math.sin(alpha), max(Lf, 1e-3))
        steer_n   = max(-1.0, min(1.0, steer_rad / self._max_steer))

        # Velocidad en m/s — el PID la mantiene con batería descargada
        speed_ms = max(self._speed_c if abs(steer_n) > 0.3 else self._speed,
                       self._min_spd)

        self.get_logger().info(
            f'wp={tidx}/{self._n} Lf={Lf:.2f}m | '
            f'α={math.degrees(alpha):+.1f}° | δ={steer_rad:+.3f}rad({steer_n:+.3f}) | '
            f'v_sp={speed_ms:.3f}m/s',
            throttle_duration_sec=0.3)

        self._publish(speed_ms, steer_n)

    def _publish(self, speed_ms: float, steering: float):
        # ← CAMBIO v2.0
        try:
            v = Float32(); v.data = float(speed_ms); self._pub_vel.publish(v)
            s = Float32(); s.data = float(steering);  self._pub_str.publish(s)
        except Exception:
            pass

    def finalize(self):
        compute_analysis(ref=self._waypoints, real=self._real_track,
                         run_name=self._run_name, runs_dir=self._runs_dir,
                         elapsed=time.time()-self._start_time,
                         obstacle_stops=self._obs_stops, loops_done=self._loops)


def main(args=None):
    rclpy.init(args=args)
    try: node = PurePursuitNode()
    except (RuntimeError, FileNotFoundError) as e:
        print(f'[ERROR] {e}'); rclpy.shutdown(); return
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    except Exception as exc: print(f'[WARN] {exc}')
    finally:
        node._publish(0.0, 0.0)
        print('\nGuardando análisis...')
        node.finalize()
        try: node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass
        print('Pure Pursuit detenido.')


if __name__ == '__main__':
    main()