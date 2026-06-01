#!/usr/bin/env python3
"""
=======================================================================
 Stanley Controller — Neuracar  (trayectoria CSV)
 Proyecto: Neuracar / Smart Mobility
-----------------------------------------------------------------------
 Implementa el controlador Stanley para seguimiento de trayectoria.
 Combina error de heading + error lateral (cross-track error) para
 calcular el ángulo de dirección.

 Ley de control Stanley:
   δ(t) = ψ_e(t) + arctan( k · e(t) / (v(t) + k_s) )

   donde:
     ψ_e = error de heading entre el vehículo y el segmento de trayectoria
     e   = cross-track error (distancia lateral al waypoint más cercano)
     v   = velocidad lineal del vehículo
     k   = ganancia Stanley (tunable)
     k_s = softening constant (evita división por cero a v≈0)

 Suscribe:
   /neuracar/odometry  (nav_msgs/Odometry)   — pose (x, y, θ)
   /neuracar/velocity  (geometry_msgs/TwistStamped) — velocidad v

 Publica:
   /neuracar/user_command  (geometry_msgs/Vector3Stamped)
       twist.linear.x   = throttle [m/s]  (velocidad objetivo)
       twist.angular.z  = steering [rad]  (ángulo de dirección)

 Parámetros ROS2:
   csv_path      (str)   — ruta al archivo CSV de waypoints (obligatorio)
   k             (float) — ganancia Stanley          [default: 0.5]
   k_soft        (float) — softening constant        [default: 0.5]
   speed         (float) — velocidad de crucero m/s  [default: 0.3]
   speed_curve   (float) — velocidad en curva m/s    [default: 0.2]
   max_steer     (float) — límite de steering rad    [default: 0.5]
   lookahead_idx (int)   — waypoints hacia adelante  [default: 3]
   goal_radius   (m)     — radio para detectar meta  [default: 0.3]
   loop          (bool)  — repetir trayectoria       [default: True]

 Uso:
   ros2 run neuracar stanley_controller \
     --ros-args -p csv_path:=/home/user/waypoints.csv -p k:=0.5
=======================================================================
"""

import csv
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry


# ────────────────────────────────────────────────────────────────────
def load_csv(path: str):
    """Carga waypoints desde CSV. Columnas: x, y, theta."""
    points = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append((float(row['x']), float(row['y']), float(row['theta'])))
    return points


def normalize_angle(angle: float) -> float:
    """Normaliza ángulo a (-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def dist2d(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def yaw_from_quaternion(qx, qy, qz, qw) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
    return math.atan2(siny, cosy)


# ────────────────────────────────────────────────────────────────────
class StanleyController(Node):

    def __init__(self):
        super().__init__('stanley_controller')

        # ── Parámetros ─────────────────────────────────────────────
        self.declare_parameter('csv_path',      '')
        self.declare_parameter('k',             0.5)
        self.declare_parameter('k_soft',        0.5)
        self.declare_parameter('speed',         0.3)
        self.declare_parameter('speed_curve',   0.2)
        self.declare_parameter('max_steer',     0.5)
        self.declare_parameter('lookahead_idx', 3)
        self.declare_parameter('goal_radius',   0.3)
        self.declare_parameter('loop',          True)

        csv_path = self.get_parameter('csv_path').value
        self._k           = self.get_parameter('k').value
        self._k_soft      = self.get_parameter('k_soft').value
        self._speed       = self.get_parameter('speed').value
        self._speed_curve = self.get_parameter('speed_curve').value
        self._max_steer   = self.get_parameter('max_steer').value
        self._look        = self.get_parameter('lookahead_idx').value
        self._goal_r      = self.get_parameter('goal_radius').value
        self._loop        = self.get_parameter('loop').value

        # ── Cargar trayectoria ─────────────────────────────────────
        if not csv_path:
            self.get_logger().fatal('Parámetro csv_path vacío. Terminando.')
            raise RuntimeError('csv_path requerido')

        self._waypoints = load_csv(csv_path)
        self._n = len(self._waypoints)
        if self._n < 2:
            raise RuntimeError('El CSV necesita al menos 2 waypoints')

        self.get_logger().info(
            f'Trayectoria cargada: {self._n} waypoints desde {csv_path}')

        # ── Estado ─────────────────────────────────────────────────
        self._x     = 0.0
        self._y     = 0.0
        self._yaw   = 0.0
        self._v     = 0.0
        self._idx   = 0        # índice del waypoint objetivo actual
        self._done  = False

        # ── Publisher ──────────────────────────────────────────────
        self._pub = self.create_publisher(
            Vector3Stamped, '/neuracar/user_command', 10)

        # ── Subscribers ────────────────────────────────────────────
        self.create_subscription(
            Odometry, '/neuracar/odometry', self._odom_cb, 10)
        self.create_subscription(
            TwistStamped, '/neuracar/velocity', self._vel_cb, 10)

        # ── Timer de control (20 Hz) ───────────────────────────────
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info('=== Stanley Controller iniciado ===')
        self.get_logger().info(f'  k={self._k}  k_soft={self._k_soft}')
        self.get_logger().info(f'  speed={self._speed} m/s  | curve={self._speed_curve} m/s')
        self.get_logger().info(f'  max_steer={math.degrees(self._max_steer):.1f}°')
        self.get_logger().info(f'  loop={self._loop}')

    # ── Callbacks ──────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def _vel_cb(self, msg: TwistStamped):
        self._v = msg.twist.linear.x

    # ── Lógica principal ───────────────────────────────────────────
    def _nearest_waypoint(self) -> int:
        """Retorna el índice del waypoint más cercano al frente del vehículo."""
        min_d = float('inf')
        best  = self._idx
        # Busca en ventana de ±50 waypoints para eficiencia
        search_range = range(
            max(0, self._idx - 5),
            min(self._n, self._idx + 50)
        )
        for i in search_range:
            wx, wy, _ = self._waypoints[i]
            d = dist2d((self._x, self._y), (wx, wy))
            if d < min_d:
                min_d = d
                best  = i
        return best

    def _control_loop(self):
        if self._done:
            self._publish(0.0, 0.0)
            return

        # ── Encontrar waypoint objetivo (con lookahead) ───────────
        nearest = self._nearest_waypoint()
        target_idx = min(nearest + self._look, self._n - 1)
        tx, ty, t_theta = self._waypoints[target_idx]

        # ── Verificar si llegó al final de la trayectoria ─────────
        end_x, end_y, _ = self._waypoints[-1]
        if dist2d((self._x, self._y), (end_x, end_y)) < self._goal_r:
            if self._loop:
                self._idx = 0
                self.get_logger().info('Vuelta completada — reiniciando trayectoria')
            else:
                self._done = True
                self.get_logger().info('¡Meta alcanzada! Deteniendo.')
                self._publish(0.0, 0.0)
                return

        self._idx = nearest

        # ── Error de heading (ψ_e) ────────────────────────────────
        # Heading del segmento de trayectoria
        if target_idx < self._n - 1:
            nx, ny, _ = self._waypoints[target_idx + 1]
            path_heading = math.atan2(ny - ty, nx - tx)
        else:
            path_heading = t_theta

        psi_e = normalize_angle(path_heading - self._yaw)

        # ── Cross-track error (e) ─────────────────────────────────
        # Signo: positivo = vehículo a la derecha del camino
        dx = tx - self._x
        dy = ty - self._y
        # Proyección lateral (perpendicular al heading del vehículo)
        e = -math.sin(self._yaw) * dx + math.cos(self._yaw) * dy

        # ── Ley de Stanley ────────────────────────────────────────
        v_eff = abs(self._v) + self._k_soft
        cte_term = math.atan2(self._k * e, v_eff)
        steering = normalize_angle(psi_e + cte_term)
        steering = max(-self._max_steer, min(steering, self._max_steer))

        # ── Velocidad adaptativa ──────────────────────────────────
        # Reduce velocidad en curvas pronunciadas
        speed = self._speed_curve if abs(steering) > 0.15 else self._speed

        self.get_logger().info(
            f'wp={target_idx}/{self._n} | '
            f'pos=({self._x:.2f},{self._y:.2f}) | '
            f'ψ_e={math.degrees(psi_e):+.1f}° | '
            f'e={e:+.3f}m | '
            f'steer={math.degrees(steering):+.1f}° | '
            f'v={speed:.2f}m/s',
            throttle_duration_sec=0.3
        )

        self._publish(speed, steering)

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
    try:
        node = StanleyController()
    except RuntimeError as e:
        print(f'[ERROR] {e}')
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Publicar STOP antes de salir
        stop = Vector3Stamped()
        stop.header.stamp = node.get_clock().now().to_msg()
        stop.vector.x = 0.0
        stop.vector.y = 0.0
        node._pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()
        print('Stanley Controller detenido.')


if __name__ == '__main__':
    main()