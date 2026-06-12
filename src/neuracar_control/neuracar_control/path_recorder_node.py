"""
path_recorder_node.py — NeuraCar
══════════════════════════════════════════════════════════════════
Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026

Records reference trajectories from odometry. Stores all incoming
points in memory and applies optional downsampling only at save time
(Ctrl+C), ensuring the vehicle can remain stationary at the start
without losing the initial waypoint.

Saves to:
  ~/Workspaces/Neuracar/src/neuracar_control/data/trajectories/
  Columns: x [m], y [m], theta [rad]

Subscriptions:
  /neuracar/odometry    nav_msgs/Odometry

Parameters:
  run_name        (str,   auto):  Output filename without extension.
                                  Default: track_YYYYMMDD_HHMMSS
  downsample_dist (float, 0.05):  Minimum distance between saved
                                  waypoints [m]. 0.0 = save all.
                                  0.05 recommended for Pure Pursuit
                                  0.02 recommended for Stanley
══════════════════════════════════════════════════════════════════
"""
import csv
import math
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def resolve_data_dir() -> str:
  
    pkg_src = os.path.expanduser(
        "~/Workspaces/Neuracar/src/neuracar_control"
    )
    if os.path.isdir(pkg_src):
        base = os.path.join(pkg_src, "data", "trajectories")
        os.makedirs(base, exist_ok=True)  
        return base

    try:
        from ament_index_python.packages import get_package_share_directory
        get_package_share_directory("neuracar_control")
        base = os.path.join(
            os.path.expanduser("~/Workspaces/Neuracar/src"),
            "neuracar_control", "data", "trajectories"
        )
        os.makedirs(base, exist_ok=True)  
        return base
    except Exception:
        pass

    fallback = os.path.expanduser(
        "~/Workspaces/Neuracar/src/neuracar_control/data/trajectories"
    )
    os.makedirs(fallback, exist_ok=True)  
    return fallback


def yaw_from_quaternion(qx, qy, qz, qw) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
    return math.atan2(siny, cosy)


def dist2d(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def downsample(points, min_dist: float):
    if min_dist <= 0.0 or not points:
        return points
    out = [points[0]]
    for p in points[1:]:
        if dist2d((p[0], p[1]), (out[-1][0], out[-1][1])) >= min_dist:
            out.append(p)
    return out


class PathRecorder(Node):

    def __init__(self):
        super().__init__('path_recorder')

        self.declare_parameter('run_name',        '')
        self.declare_parameter('downsample_dist', 0.05)  

        run_name = self.get_parameter('run_name').value
        if not run_name:
            run_name = 'track_' + datetime.now().strftime('%Y%m%d_%H%M%S')

        self._ds_dist = self.get_parameter('downsample_dist').value

        self._data_dir = resolve_data_dir()
        self._outfile  = os.path.join(self._data_dir, f'{run_name}.csv')

        if os.path.exists(self._outfile):
            suffix = datetime.now().strftime('_%H%M%S')
            self._outfile = os.path.join(
                self._data_dir, f'{run_name}{suffix}.csv')

        self._points = [] 

        # ── Subscriber ─────────────────────────────────────────────
        self.create_subscription(
            Odometry, '/neuracar/odometry', self._odom_cb, 10)

        ds_str = (f'{self._ds_dist*100:.0f} cm'
                  if self._ds_dist > 0 else 'sin filtro (guarda todo)')
        self.get_logger().info('=' * 52)
        self.get_logger().info(' PATH RECORDER — Neuracar')
        self.get_logger().info('=' * 52)
        self.get_logger().info(f'  Archivo       : {self._outfile}')
        self.get_logger().info(f'  Downsample    : {ds_str}')
        self.get_logger().info('  Graba desde el primer mensaje de odometría.')
        self.get_logger().info('  Puedes arrancar quieto — no se perderá el punto inicial.')
        self.get_logger().info('  Ctrl+C para guardar.')

    def _odom_cb(self, msg: Odometry):
        """Guarda TODOS los puntos — sin filtro de distancia."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._points.append((x, y, theta))

        if len(self._points) % 500 == 0:
            self.get_logger().info(
                f'  {len(self._points):5d} pts | '
                f'pos=({x:.2f}, {y:.2f}) θ={math.degrees(theta):.1f}°'
            )

    # ──────────────────────────────────────────────────────────────
    def save(self):
        if not self._points:
            self.get_logger().warn('No se grabaron puntos — CSV no creado.')
            return

        raw_n = len(self._points)

        pts_to_save = downsample(self._points, self._ds_dist)
        saved_n = len(pts_to_save)

        if saved_n > 1:
            total_len = sum(
                dist2d((pts_to_save[i][0], pts_to_save[i][1]),
                       (pts_to_save[i+1][0], pts_to_save[i+1][1]))
                for i in range(saved_n - 1)
            )
            mean_dist_cm = total_len / (saved_n - 1) * 100
        else:
            total_len = 0.0
            mean_dist_cm = 0.0

        try:
            with open(self._outfile, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'theta'])
                writer.writerows(pts_to_save)

            self.get_logger().info('=' * 52)
            self.get_logger().info(f'✓ CSV guardado: {self._outfile}')
            self.get_logger().info(f'  Puntos grabados (raw): {raw_n}')
            self.get_logger().info(f'  Puntos en CSV:         {saved_n}')
            self.get_logger().info(
                f'  Longitud total:        {total_len:.2f} m')
            self.get_logger().info(
                f'  Distancia media/wp:    {mean_dist_cm:.1f} cm')
            if mean_dist_cm > 0:
                wp_per_sec_55 = 55.0 / mean_dist_cm  
                self.get_logger().info(
                    f'  wp/s a speed=0.55:     {wp_per_sec_55:.0f}  '
                    f'→ lookahead recomendado ≥ {int(wp_per_sec_55*3)}')
            self.get_logger().info('=' * 52)

        except Exception as e:
            self.get_logger().error(f'Error guardando CSV: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PathRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()