#!/usr/bin/env python3
"""
velocity_pid_node.py — Neuracar v1.1
======================================
Fixes v1.1 vs v1.0:
  1. Derivativo sobre la MEDIDA (no sobre el error)
     → elimina spikes cuando el setpoint cambia bruscamente
  2. Filtro low-pass en la medida del encoder
     → el encoder al aire tiene ruido de cuantización que amplifica el derivativo
  3. Throttle feedforward
     → en lugar de partir de 0, el PID parte de una estimación inicial
     → reduce el tiempo de convergencia y la oscilación inicial
  4. Rate limit en el throttle de salida
     → limita cuánto puede cambiar el throttle por ciclo → suaviza brincos
  5. Gains por defecto más conservadores para prueba sin carga

PARÁMETROS:
  kp            (float) — proporcional              [default: 0.4]
  ki            (float) — integral                  [default: 0.1]
  kd            (float) — derivativo                [default: 0.05]
  kff           (float) — feedforward (m/s → thr)   [default: 0.8]
  alpha         (float) — low-pass medida [0-1]     [default: 0.3]
  max_throttle  (float) — límite superior            [default: 0.85]
  min_throttle  (float) — deadband motor             [default: 0.0]
  max_integral  (float) — anti-windup                [default: 0.3]
  max_rate      (float) — máx cambio throttle/s      [default: 2.0]
  v_deadband    (float) — vel mínima para PID [m/s]  [default: 0.02]
  freq_hz       (float) — frecuencia loop            [default: 50.0]

CALIBRACIÓN (carro al aire primero):
  Paso 1: kff solo (kp=0, ki=0, kd=0) — ajusta kff hasta que el motor
          gire aproximadamente a la velocidad correcta sin PID.
          kff = throttle_necesario / velocidad_deseada
          Ej: si 0.3 m/s necesita throttle ≈ 0.24 → kff = 0.24/0.3 = 0.8
  Paso 2: agrega kp=0.2 — observa la convergencia
  Paso 3: agrega ki=0.05 — elimina error residual
  Paso 4: kd si oscila
  Paso 5: repite con batería a mitad de carga — ki compensará la diferencia
"""

import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32

try:
    import serial
except ImportError:
    pass


class VelocityPIDNode(Node):

    def __init__(self):
        super().__init__('velocity_pid_node')

        # ── Parámetros ──────────────────────────────────────────────
        self.declare_parameter('kp',           0.4)
        self.declare_parameter('ki',           0.1)
        self.declare_parameter('kd',           0.05)
        self.declare_parameter('kff',          0.8)   # feedforward m/s→throttle
        self.declare_parameter('alpha',        0.3)   # low-pass: 0=sin filtro, 1=máx suavizado
        self.declare_parameter('max_throttle', 0.85)
        self.declare_parameter('min_throttle', 0.0)
        self.declare_parameter('max_integral', 0.3)
        self.declare_parameter('max_rate',     2.0)   # máx throttle/s de cambio
        self.declare_parameter('v_deadband',   0.02)
        self.declare_parameter('freq_hz',      50.0)

        self._kp       = self.get_parameter('kp').value
        self._ki       = self.get_parameter('ki').value
        self._kd       = self.get_parameter('kd').value
        self._kff      = self.get_parameter('kff').value
        self._alpha    = self.get_parameter('alpha').value
        self._max_thr  = self.get_parameter('max_throttle').value
        self._min_thr  = self.get_parameter('min_throttle').value
        self._max_int  = self.get_parameter('max_integral').value
        self._max_rate = self.get_parameter('max_rate').value
        self._vdb      = self.get_parameter('v_deadband').value
        self._freq     = self.get_parameter('freq_hz').value

        # ── Estado PID ──────────────────────────────────────────────
        self._lock          = threading.Lock()
        self._v_setpoint    = 0.0
        self._v_measured    = 0.0    # raw del encoder
        self._v_filtered    = 0.0    # después del low-pass
        self._v_filtered_prev = 0.0  # para el derivativo sobre la medida
        self._steering      = 0.0

        self._integral      = 0.0
        self._throttle_prev = 0.0   # para rate limiting
        self._prev_time     = time.monotonic()

        # ── QoS ─────────────────────────────────────────────────────
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ── Subscribers ──────────────────────────────────────────────
        self.create_subscription(
            Float32, '/neuracar/cmd_velocity',  self._sp_cb,  qos)
        self.create_subscription(
            Float32, '/neuracar/cmd_steering',  self._str_cb, qos)
        self.create_subscription(
            Float32, '/neuracar/wheel_speed',   self._fb_cb,  10)

        # ── Publisher ────────────────────────────────────────────────
        self.pub_cmd = self.create_publisher(
            Vector3Stamped, '/neuracar/user_command', 10)

        # ── Timer PID ────────────────────────────────────────────────
        self.create_timer(1.0 / self._freq, self._pid_loop)

        self.get_logger().info('Velocity PID node v1.1 listo.')
        self.get_logger().info(
            f'  kp={self._kp}  ki={self._ki}  kd={self._kd}  kff={self._kff}')
        self.get_logger().info(
            f'  alpha={self._alpha}  max_rate={self._max_rate}/s  freq={self._freq}Hz')

    # ── Callbacks ────────────────────────────────────────────────────

    def _sp_cb(self, msg: Float32):
        with self._lock:
            new_sp = float(msg.data)
            # Reset integral si el setpoint cambia de signo o va a cero
            if (abs(new_sp) < self._vdb or
                    (new_sp * self._v_setpoint < 0)):
                self._integral = 0.0
            self._v_setpoint = new_sp

    def _str_cb(self, msg: Float32):
        with self._lock:
            self._steering = float(msg.data)

    def _fb_cb(self, msg: Float32):
        """Low-pass sobre la medida del encoder para suavizar ruido."""
        with self._lock:
            raw = float(msg.data)
            # α bajo = más suavizado, α alto = más reactivo
            # v_f[n] = α·raw + (1-α)·v_f[n-1]
            self._v_filtered = (self._alpha * raw +
                                (1.0 - self._alpha) * self._v_filtered)
            self._v_measured = raw

    # ── Loop PID ─────────────────────────────────────────────────────

    def _pid_loop(self):
        now = time.monotonic()
        dt  = now - self._prev_time
        self._prev_time = now

        if dt <= 0.0 or dt > 0.5:
            return

        with self._lock:
            v_sp      = self._v_setpoint
            v_filt    = self._v_filtered
            v_filt_p  = self._v_filtered_prev
            steering  = self._steering

        # ── Setpoint en deadband → parar limpio ──────────────────────
        if abs(v_sp) < self._vdb:
            with self._lock:
                self._integral        = 0.0
                self._v_filtered_prev = v_filt
                self._throttle_prev   = 0.0
            self._publish(0.0, steering)
            return

        # ── Cálculo PID ───────────────────────────────────────────────
        error = v_sp - v_filt

        # P — proporcional al error
        p = self._kp * error

        # I — integral con anti-windup
        self._integral += error * dt
        self._integral  = max(-self._max_int,
                               min(self._max_int, self._integral))
        i = self._ki * self._integral

        # D — derivativo sobre la MEDIDA filtrada (no sobre el error)
        # d/dt(v_medida) ≈ (v_filt[n] - v_filt[n-1]) / dt
        # Negativo porque si la medida sube, queremos reducir el throttle
        d = -self._kd * (v_filt - v_filt_p) / dt

        # FF — feedforward: estimación inicial basada en el setpoint
        # Parte de un throttle razonable en lugar de 0
        ff = self._kff * abs(v_sp)

        # Throttle raw
        raw = ff + p + i + d

        # ── Saturación con signo correcto ─────────────────────────────
        if v_sp >= 0:
            throttle_sat = max(self._min_thr, min(self._max_thr, raw))
        else:
            throttle_sat = max(-self._max_thr, min(-self._min_thr, -raw))

        # ── Rate limiting — suaviza cambios bruscos ───────────────────
        max_delta    = self._max_rate * dt
        throttle_out = self._throttle_prev + max(
            -max_delta, min(max_delta,
                            throttle_sat - self._throttle_prev))

        # Actualizar estado
        with self._lock:
            self._v_filtered_prev = v_filt
            self._throttle_prev   = throttle_out

        self.get_logger().debug(
            f'sp={v_sp:.3f} filt={v_filt:.3f} err={error:+.3f} '
            f'ff={ff:.3f} p={p:+.3f} i={i:+.3f} d={d:+.3f} '
            f'→ thr={throttle_out:.3f}',
            throttle_duration_sec=0.2)

        self._publish(throttle_out, steering)

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
        stop = Vector3Stamped()
        stop.header.stamp = node.get_clock().now().to_msg()
        node.pub_cmd.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()