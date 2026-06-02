#!/usr/bin/env python3
"""
LiDAR Visualizer Node — Neuracar
====================================
Visualiza el escaneo del RPLidar A3M1 en tiempo real con gráfica polar.
Útil para verificar montaje, detectar offset angular y debug general.

Adaptado de: lidar_kalman_node_amh19.py (AMH19 — QCar Smart Mobility)
Cambios respecto al original:
  - /qcar/scan → /scan
  - Nombre de clase y nodo actualizados
  - QoS BEST_EFFORT mantenido (correcto para LiDAR)
  - Añadido parámetro scan_radius_max

IMPORTANTE — Jetson sin pantalla:
  Este nodo usa matplotlib con pyplot. Para correrlo en la Jetson
  necesitas reenviar el display a tu laptop:
    ssh -X usuario@ip-jetson
    ros2 run neuracar_perception lidar_visualizer_node
  O bien correlo directamente en tu laptop si comparten ROS_DOMAIN_ID.

Suscribe:
  /scan  (sensor_msgs/LaserScan)  RPLidar A3M1

No publica tópicos — solo visualización local.
"""

import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy,
                        QoSProfile, ReliabilityPolicy)
from sensor_msgs.msg import LaserScan


class LidarVisualizerNode(Node):

    def __init__(self):
        super().__init__('lidar_visualizer_node')

        # ── Parámetros ─────────────────────────────────────────────
        self.declare_parameter('scan_radius_max', 6.0)   # m — límite del plot
        self.declare_parameter('update_hz',       10.0)  # Hz del plot
        self._r_max = self.get_parameter('scan_radius_max').value
        hz          = self.get_parameter('update_hz').value

        # ── QoS para LiDAR — BEST_EFFORT es correcto ───────────────
        qos_lidar = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Subscriber ─────────────────────────────────────────────
        self.create_subscription(
            LaserScan, '/scan', self._lidar_callback, qos_lidar)

        # ── Timer para actualizar el plot ──────────────────────────
        self.create_timer(1.0 / hz, self._timer_callback)
        self._latest_scan = None

        # ── Matplotlib polar plot ──────────────────────────────────
        plt.ion()
        self._fig, self._ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self._ax.set_title('Neuracar — RPLidar A3M1', va='bottom')
        self._ax.set_rmax(self._r_max)
        self._ax.set_theta_zero_location('N')   # 0° = arriba = frente del carro
        self._ax.set_theta_direction(-1)         # CW = igual que LiDAR físico

        self.get_logger().info('LiDAR visualizer iniciado — escuchando /scan')
        self.get_logger().info(
            'Si no aparece la ventana, asegúrate de tener DISPLAY configurado.')

    # ──────────────────────────────────────────────────────────────────
    #  Callback: guarda el último escaneo
    # ──────────────────────────────────────────────────────────────────
    def _lidar_callback(self, msg: LaserScan):
        self._latest_scan = msg

    # ──────────────────────────────────────────────────────────────────
    #  Timer: actualiza el plot a la tasa configurada
    # ──────────────────────────────────────────────────────────────────
    def _timer_callback(self):
        if self._latest_scan is None:
            return
        self._plot_scan(self._latest_scan)

    # ──────────────────────────────────────────────────────────────────
    #  Renderizar gráfica polar
    # ──────────────────────────────────────────────────────────────────
    def _plot_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=float)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        ranges = np.clip(ranges, msg.range_min, msg.range_max)
        valid  = np.isfinite(ranges) & (ranges > msg.range_min)

        if not np.any(valid):
            self.get_logger().warn('Sin puntos válidos en el escaneo')
            return

        self._ax.clear()
        self._ax.scatter(angles[valid], ranges[valid], s=4, c='#EF9F27')
        self._ax.set_theta_zero_location('N')
        self._ax.set_theta_direction(-1)
        self._ax.set_rmax(self._r_max)
        self._ax.set_title('Neuracar — RPLidar A3M1', va='bottom')

        # Resaltar zona frontal de detección de obstáculos (±30°)
        theta_front = np.linspace(-np.radians(30), np.radians(30), 50)
        self._ax.fill_between(
            theta_front, 0, 0.35,
            alpha=0.15, color='red', label='zona alerta 35cm'
        )

        plt.pause(0.001)

        d_min  = np.min(ranges[valid])
        d_mean = np.mean(ranges[valid])
        self.get_logger().info(
            f'Min: {d_min:.2f} m  |  Media: {d_mean:.2f} m  |  '
            f'Puntos válidos: {np.sum(valid)}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()