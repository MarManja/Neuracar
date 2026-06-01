#!/usr/bin/env python3
"""
=======================================================================
 Stanley Lane Follower — Neuracar
 Proyecto: Neuracar / Smart Mobility
-----------------------------------------------------------------------
 Controlador Stanley adaptado para seguimiento de carril por visión.
 En lugar de un CSV de waypoints, el error lateral viene del
 Lane Detector (lane_detector_neuracar.py).

 Arquitectura:
   lane_detector_neuracar  →  /neuracar/lane_error
   odometry_node           →  /neuracar/velocity
   THIS NODE               →  /neuracar/user_command

 Ley de control (Stanley simplificado para carril):
   δ = arctan( k · e / (v + k_s) ) + k_yaw · ψ_lane

   donde:
     e       = cross-track error (de lane_detector, normalizado [-1,1])
     v       = velocidad lineal del vehículo (m/s)
     ψ_lane  = estimado del ángulo de la línea (derivada del error)
     k       = ganancia Stanley para CTE
     k_yaw   = ganancia para compensación de heading
     k_s     = softening constant (anti-división-por-cero)

 Suscribe:
   /neuracar/lane_error  (geometry_msgs/Vector3Stamped)  — del LaneDetector
   /neuracar/velocity    (geometry_msgs/TwistStamped)    — del OdometryNode

 Publica:
   /neuracar/user_command  (geometry_msgs/Vector3Stamped)
       vector.x = throttle [m/s]
       vector.y = steering [rad]

 Parámetros ROS2:
   k             (float) — ganancia Stanley CTE        [default: 0.4]
   k_yaw         (float) — ganancia corrección heading [default: 0.2]
   k_soft        (float) — softening constant          [default: 0.3]
   speed         (float) — velocidad crucero m/s       [default: 0.25]
   speed_curve   (float) — velocidad en curva m/s      [default: 0.15]
   max_steer     (float) — límite steering rad         [default: 0.5]
   max_lost      (int)   — frames sin línea → STOP     [default: 60]
   steer_curve_th(float) — umbral abs(steer) curva     [default: 0.15]

 Flujo de fallo:
   - Si confianza < 0.1: mantiene último steering reducido al 50%
   - Si lost_count > max_lost: STOP total
=======================================================================
"""

import math
import collections

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Vector3Stamped


class StanleyLaneFollower(Node):

    def __init__(self):
        super().__init__('stanley_lane_follower')

        # ── Parámetros ─────────────────────────────────────────────
        self.declare_parameter('k',              0.4)
        self.declare_parameter('k_yaw',          0.2)
        self.declare_parameter('k_soft',         0.3)
        self.declare_parameter('speed',          0.25)
        self.declare_parameter('speed_curve',    0.15)
        self.declare_parameter('max_steer',      0.5)
        self.declare_parameter('max_lost',       60)
        self.declare_parameter('steer_curve_th', 0.15)

        self._k             = self.get_parameter('k').value
        self._k_yaw         = self.get_parameter('k_yaw').value
        self._k_soft        = self.get_parameter('k_soft').value
        self._speed         = self.get_parameter('speed').value
        self._speed_curve   = self.get_parameter('speed_curve').value
        self._max_steer     = self.get_parameter('max_steer').value
        self._max_lost      = self.get_parameter('max_lost').value
        self._steer_curve_th= self.get_parameter('steer_curve_th').value

        # ── Estado ─────────────────────────────────────────────────
        self._error       = 0.0   # CTE normalizado actual
        self._confidence  = 0.0   # confianza de detección [0,1]
        self._v           = 0.0   # velocidad del vehículo
        self._lost_count  = 0
        self._last_steer  = 0.0

        # Buffer para estimar ψ_lane (derivada del error)
        self._error_history = collections.deque(maxlen=5)

        # ── Publishers ─────────────────────────────────────────────
        self._pub = self.create_publisher(
            Vector3Stamped, '/neuracar/user_command', 10)

        # ── Subscribers ────────────────────────────────────────────
        self.create_subscription(
            Vector3Stamped, '/neuracar/lane_error',
            self._lane_cb, 10)
        self.create_subscription(
            TwistStamped, '/neuracar/velocity',
            self._vel_cb, 10)

        # ── Timer de control 20 Hz ─────────────────────────────────
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info('=== Stanley Lane Follower iniciado ===')
        self.get_logger().info(
            f'  k={self._k}  k_yaw={self._k_yaw}  k_soft={self._k_soft}')
        self.get_logger().info(
            f'  speed={self._speed} m/s  |  curve={self._speed_curve} m/s')
        self.get_logger().info(
            f'  max_steer={math.degrees(self._max_steer):.1f}°  |  max_lost={self._max_lost}')

    # ── Callbacks ──────────────────────────────────────────────────
    def _lane_cb(self, msg: Vector3Stamped):
        self._error      = msg.vector.x   # CTE normalizado
        self._confidence = msg.vector.y   # [0, 1]
        self._error_history.append(self._error)

    def _vel_cb(self, msg: TwistStamped):
        self._v = msg.twist.linear.x

    # ── Estimación de heading de la línea ──────────────────────────
    def _estimate_psi_lane(self) -> float:
        """
        Estima la derivada del CTE como proxy del ángulo de la línea.
        Usando los últimos N errores (a 20 Hz → ~0.25 s de ventana).
        """
        if len(self._error_history) < 2:
            return 0.0
        # Derivada simple: diferencia entre el último y el primero
        delta = self._error_history[-1] - self._error_history[0]
        # Normalizar por el número de pasos
        return delta / max(len(self._error_history) - 1, 1)

    # ── Loop de control principal ──────────────────────────────────
    def _control_loop(self):
        # ── Manejo de pérdida de línea ────────────────────────────
        if self._confidence < 0.1:
            self._lost_count += 1

            if self._lost_count > self._max_lost:
                # STOP total
                self.get_logger().warn(
                    'Línea perdida por mucho tiempo. STOP.',
                    throttle_duration_sec=1.0)
                self._publish(0.0, 0.0)
                return
            else:
                # Continúa con último steering reducido
                reduced_steer = self._last_steer * 0.5
                self.get_logger().warn(
                    f'Baja confianza ({self._lost_count}/{self._max_lost}) '
                    f'steer_reducido={math.degrees(reduced_steer):+.1f}°',
                    throttle_duration_sec=0.5)
                self._publish(self._speed_curve, reduced_steer)
                return
        else:
            self._lost_count = 0

        # ── Stanley: CTE term ─────────────────────────────────────
        v_eff    = abs(self._v) + self._k_soft
        cte_term = math.atan2(self._k * self._error, v_eff)

        # ── Stanley: heading term ─────────────────────────────────
        psi_lane  = self._estimate_psi_lane()
        yaw_term  = self._k_yaw * psi_lane

        # ── Ángulo total ──────────────────────────────────────────
        steering = cte_term + yaw_term
        steering = self._normalize(steering)
        steering = max(-self._max_steer, min(steering, self._max_steer))

        # ── Velocidad adaptativa ──────────────────────────────────
        speed = self._speed_curve if abs(steering) > self._steer_curve_th else self._speed

        self._last_steer = steering

        self.get_logger().info(
            f'e={self._error:+.3f} conf={self._confidence:.2f} | '
            f'cte={math.degrees(cte_term):+.1f}° yaw={math.degrees(yaw_term):+.1f}° | '
            f'steer={math.degrees(steering):+.1f}° v={speed:.2f}',
            throttle_duration_sec=0.2)

        self._publish(speed, steering)

    # ── Helpers ───────────────────────────────────────────────────
    @staticmethod
    def _normalize(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle <= -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _publish(self, speed: float, steering: float):
        msg = Vector3Stamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.vector.x = float(speed)
        msg.vector.y = float(steering)
        self._pub.publish(msg)


# ────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = StanleyLaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # STOP al salir
        stop = Vector3Stamped()
        stop.header.stamp = node.get_clock().now().to_msg()
        stop.vector.x = 0.0
        stop.vector.y = 0.0
        node._pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()
        print('Stanley Lane Follower detenido.')


if __name__ == '__main__':
    main()