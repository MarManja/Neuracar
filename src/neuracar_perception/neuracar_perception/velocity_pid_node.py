#!/usr/bin/env python3
"""
velocity_pid_node.py — Neuracar v1.0
======================================
Nodo PID de velocidad transparente entre los controladores y el ESP32-A.

PROBLEMA QUE RESUELVE:
  La batería NiMH (motor/servo) se descarga durante la prueba.
  A igual throttle normalizado, el motor entrega menos potencia → el carro
  va más lento. Stanley/Pure Pursuit asumen velocidad constante en su modelo
  cinemático → error de seguimiento acumulado sin causa aparente.

SOLUCIÓN:
  Los controladores publican velocidad DESEADA en m/s (no throttle).
  Este nodo lee la velocidad REAL del encoder y calcula el throttle necesario
  para mantener la velocidad deseada, compensando la batería automáticamente.

FLUJO:
  Joystick/Stanley/PurePursuit
          ↓  /neuracar/cmd_velocity  (Float32, m/s)
    velocity_pid_node
          ↓  /neuracar/user_command  (Vector3Stamped, throttle∈[-1,1])
          ↑  /neuracar/wheel_speed   (Float32, m/s) — feedback encoder

  El steering pasa TRANSPARENTE: /neuracar/cmd_steering → user_command.y
  No hay PID en steering — el servo es posición, no velocidad.

TÓPICOS:
  Suscribe:
    /neuracar/cmd_velocity   (std_msgs/Float32)           — velocidad deseada [m/s]
    /neuracar/cmd_steering   (std_msgs/Float32)           — steering norm [-1,1]
    /neuracar/wheel_speed    (std_msgs/Float32)           — velocidad real [m/s]

  Publica:
    /neuracar/user_command   (geometry_msgs/Vector3Stamped) — al ESP32-A

PARÁMETROS:
  kp          (float) — ganancia proporcional         [default: 0.8]
  ki          (float) — ganancia integral              [default: 0.3]
  kd          (float) — ganancia derivativa            [default: 0.05]
  max_throttle(float) — límite superior throttle       [default: 0.85]
  min_throttle(float) — mínimo físico motor (deadband) [default: 0.0]
  max_integral(float) — anti-windup integral           [default: 0.4]
  v_deadband  (float) — velocidad mínima para PID [m/s][default: 0.02]
  freq_hz     (float) — frecuencia del loop PID        [default: 50.0]

CALIBRACIÓN INICIAL:
  Con batería cargada, kp=0.8 ki=0.3 kd=0.05 son un buen punto de partida.
  Si el carro oscila → bajar kp o subir kd.
  Si llega lento a la velocidad → subir ki.
  Si hay overshoot → bajar ki o subir kd.
"""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32


class VelocityPIDNode(Node):

    def __init__(self):
        super().__init__('velocity_pid_node')

        # ── Parámetros PID ──────────────────────────────────────────
        self.declare_parameter('kp',           0.8)
        self.declare_parameter('ki',           0.3)
        self.declare_parameter('kd',           0.05)
        self.declare_parameter('max_throttle', 0.85)   # nunca llegar a 1.0
        self.declare_parameter('min_throttle', 0.0)    # deadband: 0 = motor apagado
        self.declare_parameter('max_integral', 0.4)    # anti-windup
        self.declare_parameter('v_deadband',   0.02)   # m/s — debajo de esto → throttle=0
        self.declare_parameter('freq_hz',      50.0)

        self._kp          = self.get_parameter('kp').value
        self._ki          = self.get_parameter('ki').value
        self._kd          = self.get_parameter('kd').value
        self._max_thr     = self.get_parameter('max_throttle').value
        self._min_thr     = self.get_parameter('min_throttle').value
        self._max_int     = self.get_parameter('max_integral').value
        self._v_deadband  = self.get_parameter('v_deadband').value
        self._freq        = self.get_parameter('freq_hz').value

        # ── Estado PID ──────────────────────────────────────────────
        self._lock        = threading.Lock()
        self._v_setpoint  = 0.0    # m/s deseada
        self._v_measured  = 0.0    # m/s real del encoder
        self._steering    = 0.0    # [-1,1] pasa transparente

        self._integral    = 0.0
        self._prev_error  = 0.0
        self._prev_time   = time.monotonic()

        self._throttle_out = 0.0   # último throttle calculado

        # ── QoS RELIABLE para compatibilidad con todos los nodos ─────
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ── Subscribers ──────────────────────────────────────────────
        self.create_subscription(
            Float32, '/neuracar/cmd_velocity', self._vel_sp_cb, qos)
        self.create_subscription(
            Float32, '/neuracar/cmd_steering', self._steering_cb, qos)
        self.create_subscription(
            Float32, '/neuracar/wheel_speed',  self._feedback_cb, 10)

        # ── Publisher ────────────────────────────────────────────────
        self.pub_cmd = self.create_publisher(
            Vector3Stamped, '/neuracar/user_command', 10)

        # ── Loop PID en timer ROS2 ───────────────────────────────────
        self.create_timer(1.0 / self._freq, self._pid_loop)

        self.get_logger().info('Velocity PID node v1.0 listo.')
        self.get_logger().info(
            f'  kp={self._kp}  ki={self._ki}  kd={self._kd}')
        self.get_logger().info(
            f'  max_throttle={self._max_thr}  freq={self._freq}Hz')

    # ── Callbacks — solo guardan estado ─────────────────────────────

    def _vel_sp_cb(self, msg: Float32):
        """Velocidad deseada en m/s. Positivo=adelante, negativo=atrás."""
        with self._lock:
            self._v_setpoint = float(msg.data)
            # Reset integral cuando el setpoint cambia de signo o a cero
            if abs(self._v_setpoint) < self._v_deadband:
                self._integral   = 0.0
                self._prev_error = 0.0

    def _steering_cb(self, msg: Float32):
        """Steering normalizado [-1,1]. Pasa directo sin PID."""
        with self._lock:
            self._steering = float(msg.data)

    def _feedback_cb(self, msg: Float32):
        """Velocidad real del encoder [m/s]."""
        with self._lock:
            self._v_measured = float(msg.data)

    # ── Loop PID ────────────────────────────────────────────────────

    def _pid_loop(self):
        now = time.monotonic()

        with self._lock:
            v_sp      = self._v_setpoint
            v_meas    = self._v_measured
            steering  = self._steering

        dt = now - self._prev_time
        self._prev_time = now

        if dt <= 0.0 or dt > 0.5:
            # dt inválido (pausa/reanudación) — saltar este ciclo
            return

        # ── Caso: setpoint cero o deadband → motor apagado ───────────
        if abs(v_sp) < self._v_deadband:
            self._integral    = 0.0
            self._prev_error  = 0.0
            self._throttle_out = 0.0
            self._publish(0.0, steering)
            return

        # ── PID ──────────────────────────────────────────────────────
        error = v_sp - v_meas

        # Término proporcional
        p = self._kp * error

        # Término integral con anti-windup
        self._integral += error * dt
        self._integral  = max(-self._max_int,
                               min(self._max_int, self._integral))
        i = self._ki * self._integral

        # Término derivativo (sobre el error, no sobre la medida)
        d = self._kd * (error - self._prev_error) / dt
        self._prev_error = error

        # Throttle calculado
        raw_throttle = p + i + d

        # ── Manejo del signo (adelante/atrás) ────────────────────────
        # Si el setpoint es negativo, el throttle también lo es
        if v_sp < 0:
            throttle = max(-self._max_thr, min(-self._min_thr, raw_throttle))
        else:
            throttle = max(self._min_thr, min(self._max_thr, raw_throttle))

        self._throttle_out = throttle

        self.get_logger().debug(
            f'sp={v_sp:.3f} meas={v_meas:.3f} err={error:+.3f} '
            f'p={p:+.3f} i={i:+.3f} d={d:+.3f} → thr={throttle:.3f}',
            throttle_duration_sec=0.5)

        self._publish(throttle, steering)

    # ── Publish ─────────────────────────────────────────────────────

    def _publish(self, throttle: float, steering: float):
        msg = Vector3Stamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.vector.x        = float(throttle)
        msg.vector.y        = float(steering)
        self.pub_cmd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityPIDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Parada segura
        stop = Vector3Stamped()
        stop.header.stamp = node.get_clock().now().to_msg()
        stop.vector.x = 0.0
        stop.vector.y = 0.0
        node.pub_cmd.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()