#!/usr/bin/env python3

import serial
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped


class VehicleInterfaceNode(Node):
    def __init__(self):
        super().__init__('vehicle_interface_node')

        self.declare_parameter('port', '/dev/ttyUSB1')
        self.declare_parameter('baudrate', 115200)

        self.port = self.get_parameter('port').value
        self.baudrate = self.get_parameter('baudrate').value

        self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)

        self.subscription = self.create_subscription(
            Vector3Stamped,
            '/neuracar/user_command',
            self.command_callback,
            10
        )

        self.get_logger().info(f'Vehicle interface connected to {self.port}')

    def command_callback(self, msg: Vector3Stamped):
        throttle = max(-1.0, min(1.0, msg.vector.x))
        steering = max(-1.0, min(1.0, msg.vector.y))

        command = f'T:{throttle:.3f},S:{steering:.3f}\n'
        self.serial.write(command.encode('utf-8'))


def main(args=None):
    rclpy.init(args=args)
    node = VehicleInterfaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()