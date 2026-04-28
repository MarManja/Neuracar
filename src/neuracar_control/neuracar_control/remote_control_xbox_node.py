#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Vector3Stamped


class XboxRemoteControlNode(Node):
    def __init__(self):
        super().__init__('remote_control_xbox_node')

        self.publisher = self.create_publisher(
            Vector3Stamped,
            '/neuracar/user_command',
            10
        )

        self.subscription = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

        self.deadman_pressed = False
        self.throttle = 0.0
        self.steering = 0.0

        self.timer = self.create_timer(0.02, self.publish_command)

        self.get_logger().info('Xbox remote control node started')

    def joy_callback(self, msg: Joy):
        # Xbox typical mapping:
        # axes[0] = left stick horizontal
        # axes[2] = LT
        # axes[5] = RT
        # buttons[4] = LB deadman
        #
        # RT and LT usually go from 1.0 released to -1.0 pressed

        self.deadman_pressed = msg.buttons[4] == 1

        steering_axis = msg.axes[0]
        rt = msg.axes[5]
        lt = msg.axes[2]

        forward = (1.0 - rt) / 2.0
        reverse = (1.0 - lt) / 2.0

        self.steering = steering_axis
        self.throttle = forward - reverse

        if abs(self.steering) < 0.05:
            self.steering = 0.0

        if abs(self.throttle) < 0.05:
            self.throttle = 0.0

    def publish_command(self):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'xbox_controller'

        if self.deadman_pressed:
            msg.vector.x = float(self.throttle)
            msg.vector.y = float(self.steering)
            msg.vector.z = 1.0
        else:
            msg.vector.x = 0.0
            msg.vector.y = 0.0
            msg.vector.z = 0.0

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = XboxRemoteControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()