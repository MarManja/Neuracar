#!/usr/bin/env python3
"""
=======================================================================
 Path Recorder Node — Neuracar
 Proyecto: Neuracar / Smart Mobility
-----------------------------------------------------------------------
 Graba la trayectoria del robot suscribiéndose a /neuracar/odometry
 y guarda los waypoints en un CSV listo para usar con el
 Stanley Controller.

 Suscribe:
   /neuracar/odometry  (nav_msgs/Odometry)

 Salida CSV:
   waypoints_YYYYMMDD_HHMMSS.csv
   Columnas: x, y, theta   (metros, metros, radianes)

 Uso:
   ros2 run neuracar neuracar_path_recorder
   # Conduce manualmente el vehículo por la pista
   # Ctrl+C → guarda el CSV automáticamente
=======================================================================
"""

import csv
import math
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


# ── Distancia mínima entre waypoints consecutivos (m) ──────────────
# Evita guardar miles de puntos casi idénticos en curvas lentas.
MIN_DIST = 0.02   # 2 cm


def dist(p, q):
    """Distancia euclidiana 2D."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def yaw_from_quaternion(qx, qy, qz, qw):
    """Extrae yaw (rad) del cuaternión ZYX."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
    return math.atan2(siny, cosy)


class PathRecorder(Node):

    def __init__(self):
        super().__init__('path_recorder')

        # ── Parámetros ─────────────────────────────────────────────
        self.declare_parameter('outfile', '')
        self.declare_parameter('min_dist', MIN_DIST)

        self._min_dist = self.get_parameter('min_dist').value
        outfile = self.get_parameter('outfile').value
        if not outfile:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            outfile = os.path.join(os.getcwd(), f'waypoints_{stamp}.csv')
        self._outfile = outfile

        self._points = []          # lista de (x, y, theta)
        self._last_point = None    # último punto guardado

        # ── Subscriber ─────────────────────────────────────────────
        self.create_subscription(
            Odometry, '/neuracar/odometry', self._odom_cb, 10)

        self.get_logger().info('=== Path Recorder iniciado ===')
        self.get_logger().info(f'  Suscrito a: /neuracar/odometry')
        self.get_logger().info(f'  Archivo de salida: {self._outfile}')
        self.get_logger().info(f'  Distancia mínima entre puntos: {self._min_dist} m')
        self.get_logger().info('  Conduce el vehículo y presiona Ctrl+C para guardar.')

    # ──────────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        # Solo guarda si se movió lo suficiente
        if self._last_point is None or dist((x, y), self._last_point) >= self._min_dist:
            self._points.append((x, y, theta))
            self._last_point = (x, y)

            if len(self._points) % 50 == 0:
                self.get_logger().info(
                    f'  Puntos grabados: {len(self._points)} | '
                    f'pos=({x:.2f}, {y:.2f}) θ={math.degrees(theta):.1f}°'
                )

    # ──────────────────────────────────────────────────────────────
    def save(self):
        if not self._points:
            self.get_logger().warn('No se grabaron puntos.')
            return
        try:
            with open(self._outfile, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'theta'])
                writer.writerows(self._points)
            self.get_logger().info(
                f'✓ Guardados {len(self._points)} waypoints en: {self._outfile}')
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