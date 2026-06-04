#!/usr/bin/env python3
"""
=======================================================================
 Path Recorder Node — Neuracar
 Proyecto: Neuracar / Smart Mobility
-----------------------------------------------------------------------
 Graba la trayectoria del robot desde /neuracar/odometry y guarda
 los waypoints en un CSV dentro de neuracar_control/data/.

 La ruta de salida se resuelve automáticamente usando ament_index
 para localizar el paquete neuracar_control en el workspace activo.
 Si el paquete no se encuentra, cae en ~/neuracar_ws/trajectories/.

 Suscribe:
   /neuracar/odometry  (nav_msgs/Odometry)

 Salida:
   <neuracar_control>/data/<run_name>.csv
   Columnas: x, y, theta   [m, m, rad]

 Parámetros ROS2:
   run_name  (str)   — nombre base del archivo, sin extensión
                        default: 'track_YYYYMMDD_HHMMSS'
   min_dist  (float) — distancia mínima entre waypoints [m]  default: 0.02

 Uso:
   ros2 run neuracar_control path_recorder \
     --ros-args -p run_name:=vuelta_01
=======================================================================
"""

import csv
import math
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


MIN_DIST_DEFAULT = 0.02   # 2 cm


def resolve_data_dir() -> str:
    """
    Carpeta de trayectorias de referencia:
      ~/Workspaces/Neuracar/src/neuracar_control/data/trajectories/

    Separada de runs/ donde el Stanley guarda los análisis de prueba.
    Si no existe, la crea con mkdir -p (equivalente).
    """
    pkg_src = os.path.expanduser(
        "~/Workspaces/Neuracar/src/neuracar_control"
    )
    if os.path.isdir(pkg_src):
        base = os.path.join(pkg_src, "data", "trajectories")
        os.makedirs(base, exist_ok=True)  # equivale a mkdir -p
        return base

    try:
        from ament_index_python.packages import get_package_share_directory
        get_package_share_directory("neuracar_control")
        base = os.path.join(
            os.path.expanduser("~/Workspaces/Neuracar/src"),
            "neuracar_control", "data", "trajectories"
        )
        os.makedirs(base, exist_ok=True)  # equivale a mkdir -p
        return base
    except Exception:
        pass

    fallback = os.path.expanduser(
        "~/Workspaces/Neuracar/src/neuracar_control/data/trajectories"
    )
    os.makedirs(fallback, exist_ok=True)  # equivale a mkdir -p
    return fallback


def dist2d(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def yaw_from_quaternion(qx, qy, qz, qw) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
    return math.atan2(siny, cosy)


class PathRecorder(Node):

    def __init__(self):
        super().__init__('path_recorder')

        # ── Parámetros ─────────────────────────────────────────────
        self.declare_parameter('run_name', '')
        self.declare_parameter('min_dist', MIN_DIST_DEFAULT)

        run_name = self.get_parameter('run_name').value
        if not run_name:
            run_name = 'track_' + datetime.now().strftime('%Y%m%d_%H%M%S')

        self._min_dist = self.get_parameter('min_dist').value

        # ── Ruta de salida estructurada ────────────────────────────
        self._data_dir = resolve_data_dir()
        self._outfile  = os.path.join(self._data_dir, f'{run_name}.csv')

        # Evitar sobreescribir un archivo existente
        if os.path.exists(self._outfile):
            base = os.path.join(self._data_dir, run_name)
            suffix = datetime.now().strftime('_%H%M%S')
            self._outfile = f'{base}{suffix}.csv'

        # ── Estado ─────────────────────────────────────────────────
        self._points     = []
        self._last_point = None

        # ── Subscriber ─────────────────────────────────────────────
        self.create_subscription(
            Odometry, '/neuracar/odometry', self._odom_cb, 10)

        self.get_logger().info('=' * 52)
        self.get_logger().info(' PATH RECORDER — Neuracar')
        self.get_logger().info('=' * 52)
        self.get_logger().info(f'  Archivo de salida : {self._outfile}')
        self.get_logger().info(f'  Dist. mínima      : {self._min_dist} m')
        self.get_logger().info('  Conduce el vehículo. Ctrl+C para guardar.')

    # ──────────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        if (self._last_point is None
                or dist2d((x, y), self._last_point) >= self._min_dist):
            self._points.append((x, y, theta))
            self._last_point = (x, y)

            if len(self._points) % 50 == 0:
                self.get_logger().info(
                    f'  Puntos: {len(self._points):4d} | '
                    f'pos=({x:.2f}, {y:.2f}) '
                    f'θ={math.degrees(theta):.1f}°'
                )

    # ──────────────────────────────────────────────────────────────
    def save(self):
        if not self._points:
            self.get_logger().warn('No se grabaron puntos — CSV no creado.')
            return

        try:
            with open(self._outfile, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'theta'])
                writer.writerows(self._points)

            self.get_logger().info(
                f'✓ {len(self._points)} waypoints → {self._outfile}')
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