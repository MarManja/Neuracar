#!/usr/bin/env python3
"""
remote_control_xbox_node.py — Neuracar v2.0
Cambio v2.0: el joystick ya NO publica user_command directamente.
Publica en cmd_velocity [m/s] y cmd_steering [-1,1] para que el
velocity_pid_node mantenga velocidad constante con batería descargada.

El eje RT controla velocidad en m/s (mapeado a [0, max_speed]).
El eje LT controla reversa en m/s (mapeado a [0, -max_speed]).
max_speed es parámetro configurable (default 0.5 m/s).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Float32   # ← CAMBIO v2.0


class XboxRemoteControlNode(Node):
    def __init__(self):
        super().__init__('remote_control_xbox_node')

        # ── Parámetros ──────────────────────────────────────────────
        self.declare_parameter('max_speed', 0.5)   # m/s máximo que pide el joystick

        self._max_speed = self.get_parameter('max_speed').value

        # ── Publishers — ← CAMBIO v2.0 ──────────────────────────────
        # Ya NO publica en /neuracar/user_command.
        # El velocity_pid_node toma cmd_velocity [m/s] y calcula el throttle.
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
        self._speed_ms = 0.0   # m/s deseados
        self._steering = 0.0   # [-1,1]

        self.create_timer(0.02, self._publish)   # 50 Hz
        self.get_logger().info(
            f'Xbox control v2.0 — max_speed={self._max_speed} m/s | PID activo')

    def _alert_cb(self, msg: Bool):
        self._obstacle = msg.data
        if msg.data:
            self.get_logger().warn('Obstáculo al FRENTE',
                                   throttle_duration_sec=1.0)

    def _alert_rear_cb(self, msg: Bool):
        self._obstacle_rear = msg.data
        if msg.data:
            self.get_logger().warn('Obstáculo TRASERO',
                                   throttle_duration_sec=1.0)

    def _joy_cb(self, msg: Joy):
        # Xbox mapping:
        #   axes[0] = left stick horizontal (left=+1, right=-1)
        #   axes[2] = LT (1.0=suelto, -1.0=presionado)
        #   axes[5] = RT (1.0=suelto, -1.0=presionado)
        #   buttons[4] = LB (deadman)

        self._deadman = (msg.buttons[4] == 1)

        steering_raw = msg.axes[0]
        rt = msg.axes[5]
        lt = msg.axes[2]

        # RT → velocidad adelante [0, max_speed] m/s
        forward_ms = ((1.0 - rt) / 2.0) * self._max_speed
        # LT → velocidad atrás [0, max_speed] m/s
        reverse_ms = ((1.0 - lt) / 2.0) * self._max_speed

        self._speed_ms = forward_ms - reverse_ms   # m/s, positivo=adelante

        # Steering: invertido (joy left=+1 → carro izquierda, firmware +1=derecha)
        self._steering = -steering_raw

        # Deadbands
        if abs(self._steering)  < 0.05: self._steering  = 0.0
        if abs(self._speed_ms)  < 0.02: self._speed_ms  = 0.0

    def _publish(self):
        if not self._deadman:
            # Sin deadman → parar
            v = Float32(); v.data = 0.0; self._pub_vel.publish(v)
            s = Float32(); s.data = 0.0; self._pub_str.publish(s)
            return

        speed = self._speed_ms
        # Bloqueo de obstáculos
        if self._obstacle      and speed > 0.0: speed = 0.0
        if self._obstacle_rear and speed < 0.0: speed = 0.0

        v = Float32(); v.data = float(speed);        self._pub_vel.publish(v)
        s = Float32(); s.data = float(self._steering); self._pub_str.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node = XboxRemoteControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()