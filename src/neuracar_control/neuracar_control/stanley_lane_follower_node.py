#!/usr/bin/env python3
"""
stanley_lane_follower_node.py — Neuracar v2.0
Cambio v2.0: publica en cmd_velocity [m/s] y cmd_steering [-1,1]
en lugar de user_command directo. El PID mantiene velocidad constante.
Lógica Stanley de seguimiento de carril idéntica a v1.0.
"""

import collections
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool, Float32   # ← CAMBIO v2.0


class StanleyLaneFollower(Node):

    def __init__(self):
        super().__init__('stanley_lane_follower')

        self.declare_parameter('k',              0.4)
        self.declare_parameter('k_yaw',          0.2)
        self.declare_parameter('k_soft',         0.3)
        self.declare_parameter('speed',          0.3)    # m/s crucero ← CAMBIO v2.0
        self.declare_parameter('speed_curve',    0.2)    # m/s en curva ← CAMBIO v2.0
        self.declare_parameter('min_speed',      0.0)    # m/s mínimo
        self.declare_parameter('max_steer',      0.5)
        self.declare_parameter('max_lost',       60)
        self.declare_parameter('steer_curve_th', 0.15)

        self._k           = self.get_parameter('k').value
        self._k_yaw       = self.get_parameter('k_yaw').value
        self._k_soft      = self.get_parameter('k_soft').value
        self._speed       = self.get_parameter('speed').value
        self._speed_curve = self.get_parameter('speed_curve').value
        self._min_speed   = self.get_parameter('min_speed').value
        self._max_steer   = self.get_parameter('max_steer').value
        self._max_lost    = self.get_parameter('max_lost').value
        self._curve_th    = self.get_parameter('steer_curve_th').value

        self._error      = 0.0
        self._confidence = 0.0
        self._v          = 0.0
        self._lost_count = 0
        self._last_steer = 0.0
        self._obstacle   = False
        self._obs_stops  = 0
        self._error_hist = collections.deque(maxlen=5)

        # ── Publishers — ← CAMBIO v2.0 ──────────────────────────────
        self._pub_vel = self.create_publisher(Float32, '/neuracar/cmd_velocity', 10)
        self._pub_str = self.create_publisher(Float32, '/neuracar/cmd_steering',  10)

        from geometry_msgs.msg import Vector3Stamped
        self.create_subscription(Vector3Stamped, '/neuracar/lane_error',
                                 self._lane_cb, 10)
        self.create_subscription(TwistStamped, '/neuracar/velocity',
                                 self._vel_cb, 10)
        self.create_subscription(Bool, '/neuracar/lidar/obstacle_alert',
                                 self._obs_cb, 10)

        self.create_timer(0.05, self._control_loop)  # 20 Hz

        self.get_logger().info('=' * 52)
        self.get_logger().info(' STANLEY LANE FOLLOWER v2.0 — con PID velocidad')
        self.get_logger().info('=' * 52)
        self.get_logger().info(
            f'  k={self._k} k_yaw={self._k_yaw} k_soft={self._k_soft}')
        self.get_logger().info(
            f'  speed={self._speed}m/s curve={self._speed_curve}m/s')
        self.get_logger().info('  → PID compensa batería NiMH automáticamente')

    def _lane_cb(self, msg):
        self._error = msg.vector.x
        self._confidence = msg.vector.y
        self._error_hist.append(self._error)

    def _vel_cb(self, msg: TwistStamped):
        self._v = msg.twist.linear.x

    def _obs_cb(self, msg: Bool):
        was = self._obstacle; self._obstacle = msg.data
        if self._obstacle and not was:
            self._obs_stops += 1
            self.get_logger().warn(f'¡Obstáculo! Parada #{self._obs_stops}')
        elif not self._obstacle and was:
            self.get_logger().info('Despejado — reanudando')

    def _psi_lane(self):
        if len(self._error_hist) < 2: return 0.0
        return ((self._error_hist[-1] - self._error_hist[0])
                / max(len(self._error_hist)-1, 1))

    def _control_loop(self):
        if self._obstacle:
            self._publish(0.0, 0.0); return

        if self._confidence < 0.1:
            self._lost_count += 1
            if self._lost_count > self._max_lost:
                self.get_logger().warn('Línea perdida — STOP',
                                       throttle_duration_sec=1.0)
                self._publish(0.0, 0.0); return
            reduced = self._last_steer * 0.5
            self._publish(self._speed_curve, reduced); return

        self._lost_count = 0

        v_eff     = abs(self._v) + self._k_soft
        cte_term  = math.atan2(self._k * self._error, v_eff)
        yaw_term  = self._k_yaw * self._psi_lane()
        steer_rad = self._normalize(cte_term + yaw_term)

        steering_norm = max(-1.0, min(1.0, steer_rad / self._max_steer))

        # Velocidad en m/s — el PID la mantiene con batería descargada
        speed_ms = self._speed_curve if abs(steering_norm) > self._curve_th else self._speed
        speed_ms = max(speed_ms, self._min_speed)
        self._last_steer = steering_norm

        self.get_logger().info(
            f'e={self._error:+.3f} conf={self._confidence:.2f} | '
            f'steer={steer_rad:+.3f}rad({steering_norm:+.3f}) | '
            f'v_sp={speed_ms:.3f}m/s',
            throttle_duration_sec=0.2)

        self._publish(speed_ms, steering_norm)

    @staticmethod
    def _normalize(a):
        while a >  math.pi: a -= 2*math.pi
        while a <= -math.pi: a += 2*math.pi
        return a

    def _publish(self, speed_ms: float, steering: float):
        # ← CAMBIO v2.0
        try:
            v = Float32(); v.data = float(speed_ms); self._pub_vel.publish(v)
            s = Float32(); s.data = float(steering);  self._pub_str.publish(s)
        except Exception:
            pass


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