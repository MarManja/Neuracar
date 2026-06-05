#!/usr/bin/env python3
"""
remote_control_ps_node.py — Neuracar v2.0
Cambio v2.0: publica en cmd_velocity [m/s] y cmd_steering [-1,1]
en lugar de user_command directo. El PID mantiene velocidad constante.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Float32


class PlayStationRemoteControlNode(Node):
    def __init__(self):
        super().__init__('remote_control_ps_node')

        self.declare_parameter('max_speed', 0.5)   # m/s
        self._max_speed = self.get_parameter('max_speed').value

        self._pub_vel = self.create_publisher(Float32, '/neuracar/cmd_velocity', 10)
        self._pub_str = self.create_publisher(Float32, '/neuracar/cmd_steering', 10)

        self.create_subscription(Joy, '/joy', self._joy_cb, 10)

        self._obstacle      = False
        self._obstacle_rear = False
        self.create_subscription(Bool, '/neuracar/lidar/obstacle_alert',
                                 self._alert_cb, 10)
        self.create_subscription(Bool, '/neuracar/lidar/obstacle_alert_rear',
                                 self._alert_rear_cb, 10)

        self._deadman  = False
        self._speed_ms = 0.0
        self._steering = 0.0

        self.create_timer(0.02, self._publish)
        self.get_logger().info(
            f'PS control v2.0 — max_speed={self._max_speed} m/s | PID activo')

    def _alert_cb(self, msg: Bool):
        self._obstacle = msg.data
        if msg.data:
            self.get_logger().warn('Obstáculo al FRENTE', throttle_duration_sec=1.0)

    def _alert_rear_cb(self, msg: Bool):
        self._obstacle_rear = msg.data
        if msg.data:
            self.get_logger().warn('Obstáculo TRASERO', throttle_duration_sec=1.0)

    def _joy_cb(self, msg: Joy):
        # PS mapping:
        #   axes[0] = left stick horizontal
        #   axes[2] = L2, axes[5] = R2
        #   axes[4] = right stick vertical
        #   buttons[4] = L1 (deadman)

        self._deadman = (msg.buttons[4] == 1)
        self._steering = msg.axes[0]

        forward_ms = reverse_ms = 0.0

        if msg.axes[5] < 0 or msg.buttons[7] == 1:
            forward_ms = ((1.0 - msg.axes[5]) / 2.0) * self._max_speed
        if msg.axes[2] < 0 or msg.buttons[6] == 1:
            reverse_ms = ((1.0 - msg.axes[2]) / 2.0) * self._max_speed

        if msg.axes[4] < 0:   reverse_ms = self._max_speed
        elif msg.axes[4] > 0: forward_ms = self._max_speed

        self._speed_ms = forward_ms - reverse_ms

        if abs(self._steering)  < 0.05: self._steering  = 0.0
        if abs(self._speed_ms)  < 0.02: self._speed_ms  = 0.0

    def _publish(self):
        if not self._deadman:
            v = Float32(); v.data = 0.0; self._pub_vel.publish(v)
            s = Float32(); s.data = 0.0; self._pub_str.publish(s)
            return

        speed = self._speed_ms
        if self._obstacle      and speed > 0.0: speed = 0.0
        if self._obstacle_rear and speed < 0.0: speed = 0.0

        v = Float32(); v.data = float(speed);        self._pub_vel.publish(v)
        s = Float32(); s.data = float(self._steering); self._pub_str.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node = PlayStationRemoteControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()