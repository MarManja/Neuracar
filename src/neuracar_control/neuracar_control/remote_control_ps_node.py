#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Vector3Stamped


class PlayStationRemoteControlNode(Node):
    def __init__(self):
        super().__init__('remote_control_ps_node')

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

        self.get_logger().info('PlayStation remote control node started')

    def joy_callback(self, msg: Joy):
        # Mapeo actualizado:
        # axes[0] = joystick izquierdo horizontal (dirección "X")
        # axes[1] = joystick izquierdo vertical (dirección "Y")
        # axes[2] = L2 (reversa)
        # axes[5] = R2 (aceleración)
        # axes[4] = joystick derecho horizontal (derecha/izquierda)

        self.deadman_pressed = msg.buttons[4] == 1  # Deadman con L1

        # Dirección con el joystick izquierdo (X-axis)
        self.steering = msg.axes[0]  # Dirección con el joystick izquierdo (horizontal)

        # Inicializamos forward y reverse en 0
        forward = 0.0
        reverse = 0.0

        # Si L2 está presionado, la acción es reversa (verificando tanto axes como buttons)
        if msg.axes[2] < 0 or msg.buttons[6] == 1:  # L2 presionado
            reverse = (1.0 - msg.axes[2]) / 2.0  # L2 (reversa)

        # Si R2 está presionado, la acción es aceleración (verificando tanto axes como buttons)
        if msg.axes[5] < 0 or msg.buttons[7] == 1:  # R2 presionado
            forward = (1.0 - msg.axes[5]) / 2.0  # R2 (aceleración)

        # Si el joystick derecho se mueve a la izquierda o derecha, también debe controlar
        if msg.axes[4] < 0:  # Joystick derecho a la izquierda (retroceder)
            reverse = 1.0  # Retrocede
        elif msg.axes[4] > 0:  # Joystick derecho a la derecha (avanzar)
            forward = 1.0  # Acelera

        self.throttle = forward - reverse  # Aceleración

        # Deadzone para suavizar la respuesta:
        if abs(self.steering) < 0.05:
            self.steering = 0.0

        if abs(self.throttle) < 0.05:
            self.throttle = 0.0

    def publish_command(self):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'playstation_controller'

        if self.deadman_pressed:
            msg.vector.x = float(self.throttle)  # Aceleración
            msg.vector.y = float(self.steering)  # Dirección
            msg.vector.z = 1.0  # Valor fijo para Z (se puede ajustar si lo necesitas)
        else:
            msg.vector.x = 0.0
            msg.vector.y = 0.0
            msg.vector.z = 0.0

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PlayStationRemoteControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()