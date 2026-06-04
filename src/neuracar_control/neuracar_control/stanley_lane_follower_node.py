#!/usr/bin/env python3
"""
=======================================================================
 Stanley Lane Follower — Neuracar
 Proyecto: Neuracar / Smart Mobility
-----------------------------------------------------------------------
 Controlador Stanley para seguimiento de carril por visión.
 Integra parada de emergencia por obstáculo LiDAR.

 Suscribe:
   /neuracar/lane_error           (geometry_msgs/Vector3Stamped)
   /neuracar/velocity             (geometry_msgs/TwistStamped)
   /neuracar/lidar/obstacle_alert (std_msgs/Bool)

 Publica:
   /neuracar/user_command  (geometry_msgs/Vector3Stamped)
     vector.x = throttle  [-1, 1]
     vector.y = steering  [-1, 1]

 Parámetros ROS2:
   k             (float) — ganancia CTE            [default: 0.4]
   k_yaw         (float) — ganancia heading         [default: 0.2]
   k_soft        (float) — softening constant       [default: 0.3]
   speed         (float) — velocidad crucero m/s    [default: 0.25]
   speed_curve   (float) — velocidad en curva m/s   [default: 0.15]
   max_steer     (float) — límite steering rad      [default: 0.5]
   max_lost      (int)   — frames sin línea → STOP  [default: 60]
   steer_curve_th(float) — umbral curva abs(steer)  [default: 0.15]
=======================================================================
"""

import collections
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Vector3Stamped
from std_msgs.msg import Bool


class StanleyLaneFollower(Node):

    def __init__(self):
        super().__init__('stanley_lane_follower')

        # ── Parámetros ─────────────────────────────────────────────
        self.declare_parameter('k',               0.4)
        self.declare_parameter('k_yaw',           0.2)
        self.declare_parameter('k_soft',          0.3)
        self.declare_parameter('speed',           0.50)  # throttle crucero
        self.declare_parameter('speed_curve',     0.45)  # throttle en curva
        self.declare_parameter('min_throttle',    0.45)  # mínimo físico del motor
        self.declare_parameter('max_steer',       0.5)   # ángulo máx. servo en rad → normaliza a [-1,1] para ESP32
        self.declare_parameter('max_lost',        60)
        self.declare_parameter('steer_curve_th',  0.15)

        self._k             = self.get_parameter('k').value
        self._k_yaw         = self.get_parameter('k_yaw').value
        self._k_soft        = self.get_parameter('k_soft').value
        self._speed         = self.get_parameter('speed').value
        self._speed_curve   = self.get_parameter('speed_curve').value
        self._min_throttle  = self.get_parameter('min_throttle').value
        self._max_steer     = self.get_parameter('max_steer').value
        self._max_lost      = self.get_parameter('max_lost').value
        self._curve_th      = self.get_parameter('steer_curve_th').value

        # ── Estado ─────────────────────────────────────────────────
        self._error       = 0.0
        self._confidence  = 0.0
        self._v           = 0.0
        self._lost_count  = 0
        self._last_steer  = 0.0
        self._obstacle    = False
        self._obs_stops   = 0
        self._error_hist  = collections.deque(maxlen=5)

        # ── Publisher ──────────────────────────────────────────────
        self._pub = self.create_publisher(
            Vector3Stamped, '/neuracar/user_command', 10)

        # ── Subscribers ────────────────────────────────────────────
        self.create_subscription(
            Vector3Stamped, '/neuracar/lane_error',
            self._lane_cb, 10)
        self.create_subscription(
            TwistStamped, '/neuracar/velocity',
            self._vel_cb, 10)
        self.create_subscription(
            Bool, '/neuracar/lidar/obstacle_alert',
            self._obs_cb, 10)

        # ── Timer 20 Hz ────────────────────────────────────────────
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info('=' * 52)
        self.get_logger().info(' STANLEY LANE FOLLOWER — Neuracar')
        self.get_logger().info('=' * 52)
        self.get_logger().info(
            f'  k={self._k}  k_yaw={self._k_yaw}  k_soft={self._k_soft}')
        self.get_logger().info(
            f'  throttle crucero={self._speed} | curva={self._speed_curve} | mínimo={self._min_throttle}')
        self.get_logger().info(
            f'  max_steer={self._max_steer} rad (ángulo físico máximo del servo)  '
            f'max_lost={self._max_lost}')

    # ── Callbacks ──────────────────────────────────────────────────
    def _lane_cb(self, msg: Vector3Stamped):
        self._error      = msg.vector.x
        self._confidence = msg.vector.y
        self._error_hist.append(self._error)

    def _vel_cb(self, msg: TwistStamped):
        self._v = msg.twist.linear.x

    def _obs_cb(self, msg: Bool):
        was = self._obstacle
        self._obstacle = msg.data
        if self._obstacle and not was:
            self._obs_stops += 1
            self.get_logger().warn(
                f'¡Obstáculo! Parada #{self._obs_stops}')
        elif not self._obstacle and was:
            self.get_logger().info('Obstáculo despejado — reanudando')

    # ── Estimación de heading de línea ─────────────────────────────
    def _psi_lane(self) -> float:
        if len(self._error_hist) < 2:
            return 0.0
        return ((self._error_hist[-1] - self._error_hist[0])
                / max(len(self._error_hist) - 1, 1))

    # ── Loop de control ────────────────────────────────────────────
    def _control_loop(self):
        # Parada de emergencia por obstáculo
        if self._obstacle:
            self._publish(0.0, 0.0)
            return

        # Pérdida de línea
        if self._confidence < 0.1:
            self._lost_count += 1
            if self._lost_count > self._max_lost:
                self.get_logger().warn(
                    'Línea perdida mucho tiempo — STOP',
                    throttle_duration_sec=1.0)
                self._publish(0.0, 0.0)
                return
            reduced = self._last_steer * 0.5
            self.get_logger().warn(
                f'Baja confianza ({self._lost_count}/{self._max_lost}) '
                f'steer={math.degrees(reduced):+.1f}°',
                throttle_duration_sec=0.5)
            self._publish(self._speed_curve, reduced)
            return

        self._lost_count = 0

        # ── Stanley ───────────────────────────────────────────────
        v_eff     = abs(self._v) + self._k_soft
        cte_term  = math.atan2(self._k * self._error, v_eff)
        yaw_term  = self._k_yaw * self._psi_lane()
        steer_rad = self._normalize(cte_term + yaw_term)

        # Convertir rad → normalizado [-1,1] para ESP32-A
        # Protocolo: C,throttle,steering  donde steering∈[-1,1]
        #   +1.0 → SERVO_RIGHT (1799 µs) = DERECHA
        #   -1.0 → SERVO_LEFT  (1299 µs) = IZQUIERDA
        steering_norm = steer_rad / self._max_steer
        steering_norm = max(-1.0, min(1.0, steering_norm))

        # Nunca por debajo del mínimo físico del motor
        # Umbral en espacio normalizado (0.3 ≈ 30% del servo)
        raw_speed = self._speed_curve if abs(steering_norm) > self._curve_th else self._speed
        speed = max(raw_speed, self._min_throttle)
        self._last_steer = steering_norm

        self.get_logger().info(
            f'e={self._error:+.3f} conf={self._confidence:.2f} | '
            f'steer={steer_rad:+.3f}rad norm={steering_norm:+.3f} | '
            f'throttle={speed:.2f}',
            throttle_duration_sec=0.2)

        self._publish(speed, steering)

    @staticmethod
    def _normalize(a: float) -> float:
        while a > math.pi:  a -= 2 * math.pi
        while a <= -math.pi: a += 2 * math.pi
        return a

    def _publish(self, speed: float, steering: float):
        msg = Vector3Stamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.vector.x        = float(speed)
        msg.vector.y        = float(steering)
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StanleyLaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()
        print('Stanley Lane Follower detenido.')


if __name__ == '__main__':
    main()