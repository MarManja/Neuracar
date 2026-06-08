#!/usr/bin/env python3
"""
Obstacle Detector Node — Neuracar
===================================
Detecta obstáculos frontales Y traseros con LiDAR.
Ventana angular dinámica ajustada según velocidad del vehículo.

Publica:
  /neuracar/lidar/obstacle_alert       (Bool)  — obstáculo al frente
  /neuracar/lidar/obstacle_alert_rear  (Bool)  — obstáculo atrás

Controladores:
  throttle > 0 bloqueado si obstacle_alert       (frente)
  throttle < 0 bloqueado si obstacle_alert_rear  (atrás)

Parámetros clave:
  lidar_front_offset_rad  float  math.pi  — Neuracar cable atrás: frente = π rad
  lidar_rear_offset_rad   float  0.0      — trasero = 0 rad (cable)
  Si cambias el montaje del LiDAR solo ajusta lidar_front_offset_rad;
  el trasero es siempre front + π (mod 2π).
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


class ObstacleDetectorNode(Node):

    def __init__(self):
        super().__init__('obstacle_detector_node')

        # ── Parámetros ─────────────────────────────────────────────
        self.declare_parameter('distance_threshold',    0.80) # 0.35
        self.declare_parameter('angle_range_low_deg',   22.5)
        self.declare_parameter('angle_range_high_deg',  30.0)
        self.declare_parameter('velocity_threshold',    1.0)
        # Neuracar (cable atrás): frente del vehículo = π en frame raw del LiDAR
        # QCar (cable derecha):   frente = -π/2 (-90°)
        # Ajusta solo este valor si cambias el montaje físico.
        self.declare_parameter('lidar_front_offset_rad', math.pi)
        self.declare_parameter('lidar_rear_offset_rad',  0.0)
        self.declare_parameter('debug_mode', False)

        self._dist_thr  = self.get_parameter('distance_threshold').value
        self._ang_low   = self.get_parameter('angle_range_low_deg').value
        self._ang_high  = self.get_parameter('angle_range_high_deg').value
        self._vel_thr   = self.get_parameter('velocity_threshold').value
        self._front     = self.get_parameter('lidar_front_offset_rad').value
        self._rear      = self.get_parameter('lidar_rear_offset_rad').value
        self._debug     = self.get_parameter('debug_mode').value

        self._angle_range = self._ang_low
        self._velocity    = 0.0

        # ── QoS para LiDAR ─────────────────────────────────────────
        qos_lidar = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Publishers ─────────────────────────────────────────────
        self._pub_front = self.create_publisher(
            Bool, '/neuracar/lidar/obstacle_alert', 10)
        self._pub_rear  = self.create_publisher(
            Bool, '/neuracar/lidar/obstacle_alert_rear', 10)

        # ── Subscribers ────────────────────────────────────────────
        self.create_subscription(
            LaserScan, '/scan', self._lidar_callback, qos_lidar)
        self.create_subscription(
            TwistStamped, '/neuracar/velocity', self._velocity_callback, 10)

        self.get_logger().info(
            f'Obstacle detector — frente={math.degrees(self._front):.0f}°  '
            f'trasero={math.degrees(self._rear):.0f}°  '
            f'umbral={self._dist_thr}m')

    # ── Velocity ────────────────────────────────────────────────────
    def _velocity_callback(self, msg: TwistStamped):
        self._velocity    = abs(msg.twist.linear.x)
        self._angle_range = (
            self._ang_high if self._velocity > self._vel_thr else self._ang_low)

    # ── Detección en una ventana angular ────────────────────────────
    def _check_window(self, ranges, msg, center_offset_rad):
        """
        Retorna (detected: bool, min_dist: float) para una ventana
        centrada en center_offset_rad con ancho ±angle_range.
        Usa ángulos relativos con wrap-around correcto — funciona
        aunque el offset quede fuera de [angle_min, angle_max].
        """
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # Ángulo de cada rayo relativo al centro de la ventana, wrap a (−π, π]
        rel      = (angles - center_offset_rad + np.pi) % (2 * np.pi) - np.pi
        half_rad = np.radians(self._angle_range)
        window   = np.abs(rel) <= half_rad

        valid     = (np.isfinite(ranges) &
                     (ranges > msg.range_min) &
                     (ranges < msg.range_max))
        in_window = window & valid
        close     = ranges < self._dist_thr

        detected = bool(np.any(in_window & close))
        min_dist = (float(np.min(ranges[in_window]))
                    if np.any(in_window) else float('inf'))
        return detected, min_dist

    # ── LiDAR callback ──────────────────────────────────────────────
    def _lidar_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)

        # Debug: obstáculo más cercano en todo el escaneo
        if self._debug:
            valid = (ranges > msg.range_min) & np.isfinite(ranges)
            if np.any(valid):
                idx = np.argmin(np.where(valid, ranges, np.inf))
                ang = math.degrees(msg.angle_min + idx * msg.angle_increment)
                self.get_logger().info(
                    f'Más cercano: {ranges[idx]:.2f}m a {ang:.1f}°')

        # Detección frontal y trasera
        front_detected, front_dist = self._check_window(ranges, msg, self._front)
        rear_detected,  rear_dist  = self._check_window(ranges, msg, self._rear)

        # Publicar alertas
        front_msg = Bool(); front_msg.data = front_detected
        rear_msg  = Bool(); rear_msg.data  = rear_detected
        self._pub_front.publish(front_msg)
        self._pub_rear.publish(rear_msg)

        if front_detected:
            self.get_logger().warn(
                f'FRENTE: obstáculo a {front_dist:.2f}m '
                f'(vel={self._velocity:.2f}m/s, cono=±{self._angle_range:.0f}°)',
                throttle_duration_sec=0.5)
        if rear_detected:
            self.get_logger().warn(
                f'TRASERO: obstáculo a {rear_dist:.2f}m',
                throttle_duration_sec=0.5)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()