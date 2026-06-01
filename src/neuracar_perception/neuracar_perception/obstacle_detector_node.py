#!/usr/bin/env python3
"""
Obstacle Detector Node — Neuracar
===================================
Detecta obstáculos frontales con LiDAR y ventana angular dinámica
que se ajusta según la velocidad del vehículo.

Adaptado de: qcar_lidar_alert_2.py (AMH19 — QCar Smart Mobility)
Cambios respecto al original:
  - /qcar/scan              → /scan
  - /qcar/velocity          → /neuracar/velocity  (TwistStamped, no Vector3Stamped)
  - /qcar/obstacle_alert    → /neuracar/lidar/obstacle_alert
  - front_angle_offset 4.71 → 0.0 rad
      El QCar monta el LiDAR mirando hacia atrás (270°).
      El RPLidar A2M12 con inverted=False tiene 0° al frente.
      Si tu LiDAR está rotado, ajusta LIDAR_FRONT_OFFSET_RAD en los params.
  - Velocidad: msg.vector.x → msg.twist.linear.x  (TwistStamped)
  - Todos los parámetros son declarados en ROS2 param server

Suscribe:
  /scan                    (sensor_msgs/LaserScan)      RPLidar A2M12
  /neuracar/velocity       (geometry_msgs/TwistStamped) odometry_node

Publica:
  /neuracar/lidar/obstacle_alert  (std_msgs/Bool)
"""

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

        # ── Parámetros ROS2 ────────────────────────────────────────
        self.declare_parameter('distance_threshold',   0.35)   # m
        self.declare_parameter('angle_range_low_deg',  22.5)   # deg (<= vel_threshold)
        self.declare_parameter('angle_range_high_deg', 30.0)   # deg (>  vel_threshold)
        self.declare_parameter('velocity_threshold',   1.0)    # m/s
        # Offset del LiDAR respecto al frente del vehículo en radianes.
        # RPLidar con inverted=False → 0.0 (frente = 0°)
        # Si el LiDAR está rotado 180° → math.pi
        # QCar original = 4.71 rad (270°, miraba hacia atrás)
        self.declare_parameter('lidar_front_offset_rad', 0.0)
        self.declare_parameter('debug_mode', True)

        self._dist_thr  = self.get_parameter('distance_threshold').value
        self._ang_low   = self.get_parameter('angle_range_low_deg').value
        self._ang_high  = self.get_parameter('angle_range_high_deg').value
        self._vel_thr   = self.get_parameter('velocity_threshold').value
        self._offset    = self.get_parameter('lidar_front_offset_rad').value
        self._debug     = self.get_parameter('debug_mode').value

        self._angle_range = self._ang_low   # ventana activa
        self._velocity    = 0.0             # m/s

        # ── QoS para LiDAR ─────────────────────────────────────────
        qos_lidar = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Publisher ──────────────────────────────────────────────
        self._alert_pub = self.create_publisher(
            Bool, '/neuracar/lidar/obstacle_alert', 10)

        # ── Subscribers ────────────────────────────────────────────
        self.create_subscription(
            LaserScan, '/scan', self._lidar_callback, qos_lidar)

        # /neuracar/velocity es TwistStamped (publicado por odometry_node)
        self.create_subscription(
            TwistStamped, '/neuracar/velocity', self._velocity_callback, 10)

        self.get_logger().info('Obstacle detector node iniciado')
        self.get_logger().info(f'  Umbral distancia : {self._dist_thr} m')
        self.get_logger().info(f'  Ventana lenta    : ±{self._ang_low}°')
        self.get_logger().info(f'  Ventana rápida   : ±{self._ang_high}°')
        self.get_logger().info(f'  Offset LiDAR     : {np.degrees(self._offset):.1f}°')

    # ──────────────────────────────────────────────────────────────────
    #  Velocity callback — TwistStamped (no Vector3Stamped como en QCar)
    # ──────────────────────────────────────────────────────────────────
    def _velocity_callback(self, msg: TwistStamped):
        self._velocity = abs(msg.twist.linear.x)

        self._angle_range = (
            self._ang_high if self._velocity > self._vel_thr else self._ang_low
        )

        if self._debug:
            self.get_logger().info(
                f'Velocidad: {self._velocity:.2f} m/s → ventana ±{self._angle_range}°'
            )

    # ──────────────────────────────────────────────────────────────────
    #  LiDAR callback
    # ──────────────────────────────────────────────────────────────────
    def _lidar_callback(self, msg: LaserScan):
        ranges         = np.array(msg.ranges)
        angle_min      = msg.angle_min
        angle_increment = msg.angle_increment

        # Debug: obstáculo más cercano en todo el escaneo
        if self._debug:
            valid = (ranges > msg.range_min) & np.isfinite(ranges)
            if np.any(valid):
                idx_min = np.argmin(np.where(valid, ranges, np.inf))
                ang_min = angle_min + idx_min * angle_increment
                self.get_logger().info(
                    f'Más cercano: {ranges[idx_min]:.2f} m '
                    f'a {np.degrees(ang_min):.1f}°'
                )

        # Índice central de la ventana frontal
        center_idx = int((self._offset - angle_min) / angle_increment)
        half_idx   = int(np.radians(self._angle_range) / angle_increment)

        start_idx = max(0, center_idx - half_idx)
        end_idx   = min(len(ranges) - 1, center_idx + half_idx)

        # Buscar obstáculos en la ventana
        obstacle_detected = False
        min_dist          = float('inf')

        for i in range(start_idx, end_idx + 1):
            d = ranges[i]
            if msg.range_min < d < self._dist_thr:
                obstacle_detected = True
                min_dist = min(min_dist, d)

        # Publicar alerta
        alert          = Bool()
        alert.data     = obstacle_detected
        self._alert_pub.publish(alert)

        if obstacle_detected:
            self.get_logger().warn(
                f'¡Obstáculo a {min_dist:.2f} m!  '
                f'(vel={self._velocity:.2f} m/s, cono=±{self._angle_range}°)'
            )


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