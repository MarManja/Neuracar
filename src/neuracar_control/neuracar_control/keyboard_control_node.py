"""
keyboard_control_node.py — NeuraCar
══════════════════════════════════════════════════════════════════
Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026

Keyboard teleop node. Publishes cmd_velocity [m/s] and
cmd_steering [-1,1] for velocity_pid_node to convert to throttle.

W/S → increase/decrease speed (steps of speed_step m/s)
      W goes positive (forward), S goes negative (reverse)
A/D → steer left / right
SPACE → stop (velocity and steering to 0)
Q   → stop and quit

Obstacle logic:
  Front obstacle → blocks forward motion  (speed > 0 → 0)
                   allows reverse         (speed < 0 → unchanged)
  Rear obstacle  → blocks reverse motion  (speed < 0 → 0)
                   allows forward         (speed > 0 → unchanged)

Subscriptions:
  /neuracar/lidar/obstacle_alert      std_msgs/Bool
  /neuracar/lidar/obstacle_alert_rear std_msgs/Bool

Publications:
  /neuracar/cmd_velocity  std_msgs/Float32  [m/s]
  /neuracar/cmd_steering  std_msgs/Float32  [-1, 1]

Parameters:
  max_speed     (float, 0.5):  Maximum speed [m/s]
  speed_step    (float, 0.05): Speed increment per W/S keypress [m/s]
  steering_step (float, 0.10): Steering increment per A/D keypress
  max_steering  (float, 1.0):  Steering limit [-1, 1]
══════════════════════════════════════════════════════════════════
"""
import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


class KeyboardControlNode(Node):

    def __init__(self):
        super().__init__('keyboard_control_node')

        self.declare_parameter('max_speed',     0.5)
        self.declare_parameter('speed_step',    0.05)
        self.declare_parameter('steering_step', 0.10)
        self.declare_parameter('max_steering',  1.0)

        self._max_speed     = self.get_parameter('max_speed').value
        self._speed_step    = self.get_parameter('speed_step').value
        self._steering_step = self.get_parameter('steering_step').value
        self._max_steering  = self.get_parameter('max_steering').value

        self._speed_ms      = 0.0
        self._steering      = 0.0
        self._obstacle      = False
        self._obstacle_rear = False

        self.create_subscription(
            Bool, '/neuracar/lidar/obstacle_alert',
            self._alert_cb, 10)
        self.create_subscription(
            Bool, '/neuracar/lidar/obstacle_alert_rear',
            self._alert_rear_cb, 10)

        self._pub_vel = self.create_publisher(Float32, '/neuracar/cmd_velocity', 10)
        self._pub_str = self.create_publisher(Float32, '/neuracar/cmd_steering', 10)

        self.get_logger().info(
            f'Keyboard control — max_speed={self._max_speed} m/s  '
            f'step={self._speed_step} m/s')
        self.get_logger().info('W/S: speed | A/D: steer | SPACE: stop | Q: quit')

    def _alert_cb(self, msg: Bool):
        self._obstacle = msg.data
        if msg.data:
            self.get_logger().warn(
                'Obstáculo FRENTE — avance bloqueado, reversa permitida',
                throttle_duration_sec=1.0)

    def _alert_rear_cb(self, msg: Bool):
        self._obstacle_rear = msg.data
        if msg.data:
            self.get_logger().warn(
                'Obstáculo TRASERO — reversa bloqueada, avance permitido',
                throttle_duration_sec=1.0)

    def _get_key(self) -> str:
        """Non-blocking key read with 0.1 s timeout.
        Allows ROS callbacks to be processed between keypresses."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            rr, _, _ = select.select([sys.stdin], [], [], 0.1)
            key = sys.stdin.read(1) if rr else ''
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def _publish(self):
        speed = self._speed_ms

        if self._obstacle and speed > 0.0:
            speed = 0.0

        if self._obstacle_rear and speed < 0.0:
            speed = 0.0

        v = Float32(); v.data = float(speed)
        s = Float32(); s.data = float(self._steering)
        self._pub_vel.publish(v)
        self._pub_str.publish(s)

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            key = self._get_key()

            if key == 'w':
                self._speed_ms += self._speed_step
            elif key == 's':
                self._speed_ms -= self._speed_step
            elif key == 'a':
                self._steering += self._steering_step
            elif key == 'd':
                self._steering -= self._steering_step
            elif key == ' ':
                self._speed_ms = 0.0
                self._steering = 0.0
            elif key == 'q':
                self._speed_ms = 0.0
                self._steering = 0.0
                self._publish()
                break

            self._speed_ms = max(
                -self._max_speed, min(self._max_speed, self._speed_ms))
            self._steering = max(
                -self._max_steering, min(self._max_steering, self._steering))

            self._publish()

            status = ''
            if self._obstacle:      status += '  [FRENTE BLOQUEADO]'
            if self._obstacle_rear: status += '  [TRASERO BLOQUEADO]'
            self.get_logger().info(
                f'vel={self._speed_ms:+.2f}m/s  steer={self._steering:+.2f}{status}')


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControlNode()
    try:
        node.run()
    except KeyboardInterrupt:
        node._speed_ms = 0.0
        node._steering = 0.0
        node._publish()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()