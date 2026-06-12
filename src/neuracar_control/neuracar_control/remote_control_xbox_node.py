"""
remote_control_xbox_node.py — NeuraCar
══════════════════════════════════════════════════════════════════
Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026

Xbox controller teleop node. Publishes cmd_velocity [m/s] and
cmd_steering [-1,1] for velocity_pid_node to convert to throttle.

RT  → forward  [0, max_speed] m/s
LT  → reverse  [0, -max_speed] m/s
LB  → deadman switch (must be held to move)
Left stick → steering

Obstacle logic:
  Front obstacle → blocks forward motion  (speed > 0 → 0)
                   allows reverse         (speed < 0 → unchanged)
  Rear obstacle  → blocks reverse motion  (speed < 0 → 0)
                   allows forward         (speed > 0 → unchanged)

Subscriptions:
  /joy                               sensor_msgs/Joy
  /neuracar/lidar/obstacle_alert      std_msgs/Bool
  /neuracar/lidar/obstacle_alert_rear std_msgs/Bool

Publications:
  /neuracar/cmd_velocity  std_msgs/Float32  [m/s]
  /neuracar/cmd_steering  std_msgs/Float32  [-1, 1]

Parameters:
  max_speed (float, 0.5): Maximum speed [m/s]
══════════════════════════════════════════════════════════════════
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Float32


class XboxRemoteControlNode(Node):
    def __init__(self):
        super().__init__('remote_control_xbox_node')

        self.declare_parameter('max_speed', 0.5)
        self._max_speed = self.get_parameter('max_speed').value

        self._pub_vel = self.create_publisher(Float32, '/neuracar/cmd_velocity', 10)
        self._pub_str = self.create_publisher(Float32, '/neuracar/cmd_steering', 10)

        self.create_subscription(Joy, '/joy', self._joy_cb, 10)
        self.create_subscription(Bool, '/neuracar/lidar/obstacle_alert',
                                 self._alert_cb, 10)
        self.create_subscription(Bool, '/neuracar/lidar/obstacle_alert_rear',
                                 self._alert_rear_cb, 10)

        self._obstacle      = False
        self._obstacle_rear = False
        self._deadman       = False
        self._speed_ms      = 0.0
        self._steering      = 0.0

        self.create_timer(0.02, self._publish)   # 50 Hz
        self.get_logger().info(
            f'Xbox control — max_speed={self._max_speed} m/s')

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

    def _joy_cb(self, msg: Joy):
        self._deadman = (msg.buttons[4] == 1)

        rt = msg.axes[5]
        lt = msg.axes[2]

        forward_ms = ((1.0 - rt) / 2.0) * self._max_speed
        reverse_ms = ((1.0 - lt) / 2.0) * self._max_speed
        self._speed_ms = forward_ms - reverse_ms

        self._steering = -msg.axes[0]  # invert: joy left → car left

        if abs(self._steering) < 0.05: self._steering = 0.0
        if abs(self._speed_ms) < 0.02: self._speed_ms = 0.0

    def _publish(self):
        if not self._deadman:
            v = Float32(); v.data = 0.0; self._pub_vel.publish(v)
            s = Float32(); s.data = 0.0; self._pub_str.publish(s)
            return

        speed = self._speed_ms

        if self._obstacle and speed > 0.0:
            speed = 0.0

        if self._obstacle_rear and speed < 0.0:
            speed = 0.0

        v = Float32(); v.data = float(speed)
        s = Float32(); s.data = float(self._steering)
        self._pub_vel.publish(v)
        self._pub_str.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node = XboxRemoteControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()