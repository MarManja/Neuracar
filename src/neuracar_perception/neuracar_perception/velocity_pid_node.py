#!/usr/bin/env python3
"""
velocity_pid_node_v1_6_steering_aware.py — Neuracar
====================================================
Versión con feedforward dependiente del steering.

Por qué:
  El carro no requiere el mismo throttle en recta que en curva con steering alto.
  Esta versión mezcla dos LUTs:
    - LUT_STRAIGHT: calibrada en línea recta.
    - LUT_CURVE:    calibrada en curva con steering≈1.0.

También corrige la lógica de zona inestable: para setpoints dentro del salto del ESC,
manda directamente el throttle post-salto en lugar de interpolar en una zona no usable.

Entradas:
  /neuracar/cmd_velocity   std_msgs/Float32   [m/s]
  /neuracar/cmd_steering   std_msgs/Float32   [-1, 1]
  /neuracar/wheel_speed    std_msgs/Float32   [m/s]

Salida:
  /neuracar/user_command   geometry_msgs/Vector3Stamped
    vector.x = throttle [-1, 1]
    vector.y = steering [-1, 1]
"""

import time
import threading
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32

LUT = List[Tuple[float, float]]  # (velocidad_m_s, throttle)


# ── LUT calibrada en línea recta (2026-06-07) ────────────────────────
# Fuente: lut_recta_20260607_180527.csv  steering=0.0  NiMH 7S 8.4V
# 18 puntos medidos, zona densa 0.58-0.70 para detectar salto preciso.
#
# Salto detectado: throttle 0.62→0.63 = 0.437→0.734 m/s
# Zona 0.63-0.70: promediada (velocidades similares ~0.73-0.81 m/s)
# Puntos 0.80 y 0.85 promediados (medición ruidosa, invertidos entre sí)
# Puntos 0.95 y 1.00 descartados (encoder saturó — excedió rango)
#
# ZONA NO USABLE: 0.00 - 0.46 m/s (throttle < 0.58 no arranca)
# ZONA INESTABLE: 0.46 - 0.76 m/s (salto abrupto ESC entre 0.62→0.63)
_LUT_STRAIGHT: LUT = [
    (0.000, 0.000),   # motor parado
    (0.464, 0.610),   # mínimo que arranca en recta (pre-salto)
    # ── SALTO: throttle 0.62→0.63 = 0.437→0.734 m/s ──
    (0.756, 0.661),   # post-salto — promedio zona 0.63-0.70
    (1.036, 0.750),
    (1.360, 0.825),   # promedio mediciones 0.80 y 0.85
    (1.577, 0.900),   # máximo medido válido
]

# ── LUT calibrada en curva / steering alto ───────────────────────────
# Basada en tus mediciones con steering≈1.0.
# Se volvió monotónica para poder interpolar sin saltos raros:
#   0.22@0.58 y 0.22@0.60  -> se conserva 0.22@0.60 por confiabilidad
#   0.59@0.65 y 0.58@0.70  -> se usa 0.59@0.65 como primer post-salto
#   1.29@0.80 y 1.28@0.85  -> se promedia a 1.285@0.825
_LUT_CURVE: LUT = [
    (0.00, 0.000),
    (0.18, 0.550),
    (0.22, 0.600),
    (0.59, 0.650),
    (0.95, 0.750),
    (1.285, 0.825),
    (1.61, 0.900),
    (1.74, 0.950),
    (2.01, 1.000),
]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def interpolate_lut(v_abs: float, lut: LUT) -> float:
    """Interpola throttle para una velocidad positiva usando una LUT monotónica."""
    if v_abs <= lut[0][0]:
        return lut[0][1]
    if v_abs >= lut[-1][0]:
        return lut[-1][1]

    for i in range(len(lut) - 1):
        v0, t0 = lut[i]
        v1, t1 = lut[i + 1]
        if v0 <= v_abs <= v1:
            if abs(v1 - v0) < 1e-9:
                return max(t0, t1)
            frac = (v_abs - v0) / (v1 - v0)
            return t0 + frac * (t1 - t0)

    return lut[-1][1]


def feedforward_from_lut(v_target: float, lut: LUT,
                         stable_min_v: float,
                         stable_min_throttle: float,
                         v_deadband: float) -> float:
    """
    Devuelve throttle inicial.

    Si el setpoint cae dentro de la zona inestable del ESC, se manda directo
    el throttle post-salto estable. Esto evita quedarse en el salto 0.60→0.65.
    """
    if abs(v_target) < v_deadband:
        return 0.0

    sign = 1.0 if v_target >= 0 else -1.0
    v_abs = abs(v_target)

    if v_abs < stable_min_v:
        return sign * stable_min_throttle

    return sign * interpolate_lut(v_abs, lut)


# ── Gain Scheduling ─────────────────────────────────────────────────
# Fuente: pid_tuner_recta_20260607 — barrido secuencial kp→ki→kd
# Resultado: kp=0.02, ki=0.25, kd=0.02, max_int=0.20
# Velocidades usables en recta: 0.756 - 1.577 m/s
_GAIN_SCHEDULE = [
    # v_m/s   kp     ki     kd     max_int
    (0.756,  0.02,  0.25,  0.02,   0.20),  # post-salto mínimo
    (1.036,  0.02,  0.25,  0.02,   0.20),  # calibrado recta 2026-06-07
    (1.360,  0.02,  0.25,  0.02,   0.20),  # calibrado recta 2026-06-07
    (1.577,  0.02,  0.20,  0.02,   0.15),  # máximo medido
]


def gain_schedule(v_setpoint: float) -> tuple:
    v_abs = abs(v_setpoint)

    if v_abs <= _GAIN_SCHEDULE[0][0]:
        return _GAIN_SCHEDULE[0][1:]
    if v_abs >= _GAIN_SCHEDULE[-1][0]:
        return _GAIN_SCHEDULE[-1][1:]

    for i in range(len(_GAIN_SCHEDULE) - 1):
        v0, kp0, ki0, kd0, mi0 = _GAIN_SCHEDULE[i]
        v1, kp1, ki1, kd1, mi1 = _GAIN_SCHEDULE[i + 1]
        if v0 <= v_abs <= v1:
            frac = (v_abs - v0) / (v1 - v0)
            return (
                kp0 + frac * (kp1 - kp0),
                ki0 + frac * (ki1 - ki0),
                kd0 + frac * (kd1 - kd0),
                mi0 + frac * (mi1 - mi0),
            )

    return _GAIN_SCHEDULE[-1][1:]


class VelocityPIDNode(Node):

    def __init__(self):
        super().__init__('velocity_pid_node')

        self.declare_parameter('gain_scheduling', True)
        self.declare_parameter('kp',           0.08)
        self.declare_parameter('ki',           0.30)
        self.declare_parameter('kd',           0.01)
        self.declare_parameter('max_integral', 0.40)
        self.declare_parameter('alpha',        0.3)
        self.declare_parameter('max_throttle', 1.0)
        self.declare_parameter('max_rate',     2.0)
        self.declare_parameter('v_deadband',   0.05)
        self.declare_parameter('freq_hz',      50.0)

        # Parámetros nuevos para mezcla de LUTs
        self.declare_parameter('steer_lut_start', 0.25)
        self.declare_parameter('steer_lut_full',  0.90)
        self.declare_parameter('straight_stable_min_v',        0.756)  # calibrado 2026-06-07
        self.declare_parameter('straight_stable_min_throttle', 0.661)  # calibrado 2026-06-07
        self.declare_parameter('curve_stable_min_v',           0.590)  # calibrado pista5
        self.declare_parameter('curve_stable_min_throttle',    0.650)  # calibrado pista5

        self._use_gs = bool(self.get_parameter('gain_scheduling').value)
        self._kp_fixed = float(self.get_parameter('kp').value)
        self._ki_fixed = float(self.get_parameter('ki').value)
        self._kd_fixed = float(self.get_parameter('kd').value)
        self._max_int_fixed = float(self.get_parameter('max_integral').value)
        self._alpha = float(self.get_parameter('alpha').value)
        self._max_thr = float(self.get_parameter('max_throttle').value)
        self._max_rate = float(self.get_parameter('max_rate').value)
        self._vdb = float(self.get_parameter('v_deadband').value)
        self._freq = float(self.get_parameter('freq_hz').value)
        self._steer_lut_start = float(self.get_parameter('steer_lut_start').value)
        self._steer_lut_full = float(self.get_parameter('steer_lut_full').value)
        self._straight_min_v = float(self.get_parameter('straight_stable_min_v').value)
        self._straight_min_thr = float(self.get_parameter('straight_stable_min_throttle').value)
        self._curve_min_v = float(self.get_parameter('curve_stable_min_v').value)
        self._curve_min_thr = float(self.get_parameter('curve_stable_min_throttle').value)

        self._lock = threading.Lock()
        self._v_setpoint = 0.0
        self._v_filtered = 0.0
        self._v_filtered_prev = 0.0
        self._steering = 0.0
        self._integral = 0.0
        self._throttle_prev = 0.0
        self._prev_time = time.monotonic()

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Float32, '/neuracar/cmd_velocity', self._sp_cb, qos)
        self.create_subscription(Float32, '/neuracar/cmd_steering', self._str_cb, qos)
        self.create_subscription(Float32, '/neuracar/wheel_speed', self._fb_cb, 10)

        self.pub_cmd = self.create_publisher(Vector3Stamped, '/neuracar/user_command', 10)
        self.create_timer(1.0 / self._freq, self._pid_loop)

        self.get_logger().info('Velocity PID node v1.7 — LUT recta calibrada 2026-06-07')
        self.get_logger().info(
            f'  gain_scheduling={self._use_gs} max_throttle={self._max_thr} max_rate={self._max_rate}')
        self.get_logger().info(
            f'  LUT blend: start={self._steer_lut_start:.2f} full={self._steer_lut_full:.2f}')
        self.get_logger().info(
            f'  straight jump: v<{self._straight_min_v:.2f} -> thr={self._straight_min_thr:.3f}')
        self.get_logger().info(
            f'  curve jump:    v<{self._curve_min_v:.2f} -> thr={self._curve_min_thr:.3f}')

    def _sp_cb(self, msg: Float32):
        with self._lock:
            new_sp = float(msg.data)
            if abs(new_sp) < self._vdb or (new_sp * self._v_setpoint < 0):
                self._integral = 0.0
            self._v_setpoint = new_sp

    def _str_cb(self, msg: Float32):
        with self._lock:
            self._steering = clamp(float(msg.data), -1.0, 1.0)

    def _fb_cb(self, msg: Float32):
        with self._lock:
            raw = float(msg.data)
            self._v_filtered = self._alpha * raw + (1.0 - self._alpha) * self._v_filtered

    def _ff_steering_aware(self, v_sp: float, steering: float) -> float:
        ff_straight = feedforward_from_lut(
            v_sp, _LUT_STRAIGHT,
            stable_min_v=self._straight_min_v,
            stable_min_throttle=self._straight_min_thr,
            v_deadband=self._vdb,
        )
        ff_curve = feedforward_from_lut(
            v_sp, _LUT_CURVE,
            stable_min_v=self._curve_min_v,
            stable_min_throttle=self._curve_min_thr,
            v_deadband=self._vdb,
        )

        abs_s = abs(steering)
        denom = max(1e-6, self._steer_lut_full - self._steer_lut_start)
        w_curve = clamp((abs_s - self._steer_lut_start) / denom, 0.0, 1.0)
        return (1.0 - w_curve) * ff_straight + w_curve * ff_curve

    def _pid_loop(self):
        now = time.monotonic()
        dt = now - self._prev_time
        self._prev_time = now

        if dt <= 0.0 or dt > 0.5:
            return

        with self._lock:
            v_sp = self._v_setpoint
            v_filt = self._v_filtered
            v_filt_p = self._v_filtered_prev
            steering = self._steering

        if abs(v_sp) < self._vdb:
            with self._lock:
                self._integral = 0.0
                self._v_filtered_prev = v_filt
                self._throttle_prev = 0.0
            self._publish(0.0, steering)
            return

        ff = self._ff_steering_aware(v_sp, steering)

        if self._use_gs:
            kp, ki, kd, max_int = gain_schedule(v_sp)
        else:
            kp, ki, kd, max_int = (self._kp_fixed, self._ki_fixed,
                                    self._kd_fixed, self._max_int_fixed)

        error = v_sp - v_filt
        p = kp * error

        self._integral += error * dt
        self._integral = clamp(self._integral, -max_int, max_int)
        i = ki * self._integral

        # Derivativo sobre medida para evitar spike al cambiar setpoint.
        d = -kd * (v_filt - v_filt_p) / dt

        raw = ff + p + i + d

        if v_sp >= 0:
            throttle_sat = clamp(raw, 0.0, self._max_thr)
        else:
            throttle_sat = clamp(raw, -self._max_thr, 0.0)

        # Kickstart: si estaba casi parado, permite saltar directo al FF.
        ff_abs = abs(ff)
        if ff_abs > 0 and abs(self._throttle_prev) < ff_abs * 0.5:
            throttle_out = throttle_sat
        else:
            max_delta = self._max_rate * dt
            throttle_out = self._throttle_prev + clamp(throttle_sat - self._throttle_prev,
                                                       -max_delta, max_delta)

        with self._lock:
            self._v_filtered_prev = v_filt
            self._throttle_prev = throttle_out

        self.get_logger().debug(
            f'sp={v_sp:.2f} filt={v_filt:.2f} str={steering:+.2f} '
            f'ff={ff:.3f} err={error:+.3f} p={p:+.3f} i={i:+.3f} d={d:+.3f} '
            f'-> thr={throttle_out:.3f}',
            throttle_duration_sec=0.2)

        self._publish(throttle_out, steering)

    def _publish(self, throttle: float, steering: float):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.vector.x = float(throttle)
        msg.vector.y = float(steering)
        self.pub_cmd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityPIDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop = Vector3Stamped()
            node.pub_cmd.publish(stop)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()