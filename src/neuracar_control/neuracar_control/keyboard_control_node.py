import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Bool


class KeyboardControlNode(Node):
    def __init__(self):
        super().__init__('keyboard_control_node')

        self.publisher = self.create_publisher(
            Vector3Stamped, '/neuracar/user_command', 10)

        # Parada por obstáculo LiDAR
        self._obstacle = False
        self._obstacle_rear = False
        self.create_subscription(
            Bool, '/neuracar/lidar/obstacle_alert', self._alert_cb, 10)
        self.create_subscription(
            Bool, '/neuracar/lidar/obstacle_alert_rear', self._alert_rear_cb, 10)

        self.throttle = 0.0
        self.steering = 0.0
        self.throttle_step = 0.05
        self.steering_step = 0.10
        self.max_throttle = 0.30
        self.max_steering = 1.00

        self.get_logger().info('Keyboard control started')
        self.get_logger().info('W/S: throttle | A/D: steering | SPACE: stop | Q: quit')

    def _alert_cb(self, msg: Bool):
        self._obstacle = msg.data
        if msg.data:
            self.get_logger().warn(
                'Obstáculo al FRENTE — avance bloqueado',
                throttle_duration_sec=1.0)

    def _alert_rear_cb(self, msg: Bool):
        self._obstacle_rear = msg.data
        if msg.data:
            self.get_logger().warn(
                'Obstáculo TRASERO — reversa bloqueada',
                throttle_duration_sec=1.0)

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

        throttle = self.throttle
        # Bloquear avance si hay obstáculo al frente
        if self._obstacle and throttle > 0.0:
            throttle = 0.0
        # Bloquear reversa si hay obstáculo atrás
        if self._obstacle_rear and throttle < 0.0:
            throttle = 0.0

        msg.vector.x = float(throttle)
        msg.vector.y = float(self.steering)
        msg.vector.z = 1.0
        self.publisher.publish(msg)

    def run(self):
        while rclpy.ok():
            # Procesar callbacks pendientes (obstacle alert, etc.)
            # antes de leer la siguiente tecla
            rclpy.spin_once(self, timeout_sec=0)

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
                f'throttle={self.throttle:.2f}  steering={self.steering:.2f}'
                + ('  [FRENTE]' if self._obstacle else '')
                + ('  [TRASERO]' if self._obstacle_rear else ''))


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