#!/usr/bin/env python3

import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped


class KeyboardControlNode(Node):
    def __init__(self):
        super().__init__('keyboard_control_node')

        self.publisher = self.create_publisher(
            Vector3Stamped,
            '/neuracar/user_command',
            10
        )

        self.throttle = 0.0
        self.steering = 0.0

        self.throttle_step = 0.05
        self.steering_step = 0.10

        self.max_throttle = 0.30
        self.max_steering = 1.00

        self.get_logger().info('Keyboard control started')
        self.get_logger().info('W/S: throttle | A/D: steering | SPACE: stop | Q: quit')

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return key

    def publish_command(self):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'keyboard'

        msg.vector.x = float(self.throttle)
        msg.vector.y = float(self.steering)
        msg.vector.z = 1.0

        self.publisher.publish(msg)

    def run(self):
        while rclpy.ok():
            key = self.get_key()

            if key == 'w':
                self.throttle += self.throttle_step

            elif key == 's':
                self.throttle -= self.throttle_step

            elif key == 'a':
                self.steering += self.steering_step

            elif key == 'd':
                self.steering -= self.steering_step

            elif key == ' ':
                self.throttle = 0.0
                self.steering = 0.0

            elif key == 'q':
                self.throttle = 0.0
                self.steering = 0.0
                self.publish_command()
                break

            self.throttle = max(-self.max_throttle, min(self.max_throttle, self.throttle))
            self.steering = max(-self.max_steering, min(self.max_steering, self.steering))

            self.publish_command()

            self.get_logger().info(
                f'throttle={self.throttle:.2f}, steering={self.steering:.2f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControlNode()

    try:
        node.run()
    except KeyboardInterrupt:
        node.throttle = 0.0
        node.steering = 0.0
        node.publish_command()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()