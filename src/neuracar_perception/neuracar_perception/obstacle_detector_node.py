#!/usr/bin/env python3
"""
obstacle_detector_node.py — NeuraCar
══════════════════════════════════════════════════════════════════
Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026

Detects front and rear obstacles using the RPLiDAR A3M1.
Dynamic angular window adjusts based on vehicle speed.

LiDAR is mounted with cable facing rear:
  Front window centered at π rad
  Rear window  centered at 0 rad

Subscriptions:
  /scan                  sensor_msgs/LaserScan
  /neuracar/velocity     geometry_msgs/TwistStamped

Publications:
  /neuracar/lidar/obstacle_alert       std_msgs/Bool  front obstacle
  /neuracar/lidar/obstacle_alert_rear  std_msgs/Bool  rear obstacle

Parameters:
  distance_threshold     (float, 0.80):  Stop distance [m]
  angle_range_low_deg    (float, 22.5):  Half-cone at low speed [deg]
  angle_range_high_deg   (float, 30.0):  Half-cone at high speed [deg]
  velocity_threshold     (float, 1.0):   Speed to switch cone width [m/s]
  lidar_front_offset_rad (float, pi):    Front window center [rad]
  lidar_rear_offset_rad  (float, 0.0):   Rear window center [rad]
  debug_mode             (bool,  false): Enable verbose logs
══════════════════════════════════════════════════════════════════
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

        self.declare_parameter('distance_threshold',     0.80)
        self.declare_parameter('angle_range_low_deg',    22.5)
        self.declare_parameter('angle_range_high_deg',   30.0)
        self.declare_parameter('velocity_threshold',     1.0)
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

        qos_lidar = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._pub_front = self.create_publisher(
            Bool, '/neuracar/lidar/obstacle_alert', 10)
        self._pub_rear  = self.create_publisher(
            Bool, '/neuracar/lidar/obstacle_alert_rear', 10)

        self.create_subscription(
            LaserScan, '/scan', self._lidar_callback, qos_lidar)
        self.create_subscription(
            TwistStamped, '/neuracar/velocity', self._velocity_callback, 10)

        self.get_logger().info(
            f'Obstacle detector — frente={math.degrees(self._front):.0f}°  '
            f'trasero={math.degrees(self._rear):.0f}°  '
            f'umbral={self._dist_thr}m')

    def _velocity_callback(self, msg: TwistStamped):
        # Keep sign — needed to know direction of travel
        self._velocity    = msg.twist.linear.x
        self._angle_range = (
            self._ang_high if abs(self._velocity) > self._vel_thr
            else self._ang_low)

    def _check_window(self, ranges, msg, center_offset_rad):
        angles   = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        rel      = (angles - center_offset_rad + np.pi) % (2 * np.pi) - np.pi
        half_rad = np.radians(self._angle_range)
        window   = np.abs(rel) <= half_rad
        valid    = (np.isfinite(ranges) &
                    (ranges > msg.range_min) &
                    (ranges < msg.range_max))
        in_window = window & valid
        close     = ranges < self._dist_thr
        detected  = bool(np.any(in_window & close))
        min_dist  = (float(np.min(ranges[in_window]))
                     if np.any(in_window) else float('inf'))
        return detected, min_dist

    def _lidar_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)

        if self._debug:
            valid = (ranges > msg.range_min) & np.isfinite(ranges)
            if np.any(valid):
                idx = np.argmin(np.where(valid, ranges, np.inf))
                ang = math.degrees(msg.angle_min + idx * msg.angle_increment)
                self.get_logger().info(
                    f'Nearest: {ranges[idx]:.2f}m @ {ang:.1f}°  '
                    f'vel={self._velocity:+.2f}m/s')

        front_detected, front_dist = self._check_window(ranges, msg, self._front)
        rear_detected,  rear_dist  = self._check_window(ranges, msg, self._rear)

        front_msg = Bool(); front_msg.data = front_detected
        rear_msg  = Bool(); rear_msg.data  = rear_detected
        self._pub_front.publish(front_msg)
        self._pub_rear.publish(rear_msg)

        if front_detected:
            self.get_logger().warn(
                f'FRONT obstacle at {front_dist:.2f}m '
                f'(vel={self._velocity:+.2f}m/s, cone=±{self._angle_range:.0f}°)',
                throttle_duration_sec=0.5)
        if rear_detected:
            self.get_logger().warn(
                f'REAR obstacle at {rear_dist:.2f}m '
                f'(vel={self._velocity:+.2f}m/s)',
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