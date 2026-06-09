#!/usr/bin/env python3
"""
keyboard_control_node.py — Neuracar v2.0
=========================================
Controla el vehículo desde el teclado publicando velocidad en m/s y
steering normalizado, igual que los nodos xbox y ps.

Arquitectura (consistente con xbox/ps v2.0):
  Este nodo  →  /neuracar/cmd_velocity  [Float32, m/s]
  Este nodo  →  /neuracar/cmd_steering  [Float32, -1..1]
               ↓
  velocity_pid_node  →  /neuracar/user_command  →  esp32_actuadores

El velocity_pid_node se encarga de convertir la velocidad deseada en
throttle real, compensando caída de batería y no-linealidades del ESC.

Suscripciones:
  /neuracar/lidar/obstacle_alert       (Bool) — bloquea avance
  /neuracar/lidar/obstacle_alert_rear  (Bool) — bloquea reversa

Teclas:
  W / S   — aumentar / reducir velocidad (pasos de speed_step m/s)
  A / D   — girar izquierda / derecha
  SPACE   — parar (velocidad y steering a 0)
  Q       — parar y salir
"""

import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


class KeyboardControlNode(Node):

    def __init__(self):
        super().__init__('keyboard_control_node')

        # ── Parámetros ──────────────────────────────────────────────────
        self.declare_parameter('max_speed',    0.5)   # m/s máximo pedido
        self.declare_parameter('speed_step',   0.05)  # m/s por pulsación W/S
        self.declare_parameter('steering_step', 0.10) # incremento por pulsación A/D
        self.declare_parameter('max_steering',  1.0)  # límite steering [-1, 1]

        self._max_speed     = self.get_parameter('max_speed').value
        self._speed_step    = self.get_parameter('speed_step').value
        self._steering_step = self.get_parameter('steering_step').value
        self._max_steering  = self.get_parameter('max_steering').value

        # ── Estado interno ──────────────────────────────────────────────
        self._speed_ms  = 0.0   # velocidad deseada [m/s]
        self._steering  = 0.0   # steering deseado [-1, 1]

        # ── Obstacle alerts ─────────────────────────────────────────────
        self._obstacle      = False   # obstáculo al frente
        self._obstacle_rear = False   # obstáculo atrás
        self.create_subscription(
            Bool, '/neuracar/lidar/obstacle_alert',
            self._alert_cb, 10)
        self.create_subscription(
            Bool, '/neuracar/lidar/obstacle_alert_rear',
            self._alert_rear_cb, 10)

        # ── Publishers — mismos tópicos que xbox/ps v2.0 ────────────────
        # El velocity_pid_node escucha aquí y genera el throttle real.
        self._pub_vel = self.create_publisher(Float32, '/neuracar/cmd_velocity', 10)
        self._pub_str = self.create_publisher(Float32, '/neuracar/cmd_steering', 10)

        self.get_logger().info('Keyboard control v2.0 — publicando cmd_velocity / cmd_steering')
        self.get_logger().info(
            f'max_speed={self._max_speed} m/s  |  speed_step={self._speed_step} m/s')
        self.get_logger().info('W/S: velocidad | A/D: steering | SPACE: stop | Q: salir')

    # ── Callbacks de obstáculos ─────────────────────────────────────────
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

    # ── Lectura de teclado (bloqueante, raw mode) ───────────────────────
    def _get_key(self) -> str:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    # ── Publicar velocidad y steering actuales ──────────────────────────
    def _publish(self):
        speed = self._speed_ms

        # Bloqueo por obstáculo: anula velocidad en la dirección bloqueada
        if self._obstacle      and speed > 0.0:
            speed = 0.0
        if self._obstacle_rear and speed < 0.0:
            speed = 0.0

        vel_msg = Float32()
        vel_msg.data = float(speed)
        self._pub_vel.publish(vel_msg)

        str_msg = Float32()
        str_msg.data = float(self._steering)
        self._pub_str.publish(str_msg)

    # ── Loop principal ──────────────────────────────────────────────────
    def run(self):
        while rclpy.ok():
            # Procesar callbacks (obstacle alerts) antes de leer tecla
            rclpy.spin_once(self, timeout_sec=0)

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

            # Clamp a límites
            self._speed_ms = max(
                -self._max_speed, min(self._max_speed, self._speed_ms))
            self._steering = max(
                -self._max_steering, min(self._max_steering, self._steering))

            self._publish()

            self.get_logger().info(
                f'vel={self._speed_ms:+.2f} m/s  steer={self._steering:+.2f}'
                + ('  [FRENTE BLOQUEADO]' if self._obstacle      else '')
                + ('  [TRASERO BLOQUEADO]' if self._obstacle_rear else ''))


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControlNode()
    try:
        node.run()
    except KeyboardInterrupt:
        # Enviar parada segura antes de salir
        node._speed_ms = 0.0
        node._steering = 0.0
        node._publish()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()