#!/usr/bin/env python3
"""
Odometry Node — Neuracar  v1.0
===================================
Fusiona encoder + IMU para producir pose y velocidad del vehículo.
Equivalente al nodo de odometría del QCar pero usando los tópicos Neuracar.

Suscribe:
  /neuracar/encoder  (sensor_msgs/JointState)   50 Hz
      effort[0]   = velocidad lineal  [m/s]       ← fuente principal de v
      velocity[0] = vel. angular rueda [rad/s]
  /neuracar/imu     (sensor_msgs/Imu)            50 Hz
      orientation         = quaternión (fusión BNO055 NDOF)  ← fuente de θ
      angular_velocity.z  = yaw rate  [rad/s]

Publica:
  /neuracar/odometry  (nav_msgs/Odometry)        50 Hz
      pose.pose.position.x/y    [m]    dead reckoning
      pose.pose.orientation     quat   directo del IMU
      twist.twist.linear.x      [m/s]  del encoder
      twist.twist.angular.z     [rad/s] del IMU

  /neuracar/velocity  (geometry_msgs/TwistStamped)  50 Hz
      twist.linear.x    [m/s]
      twist.angular.z   [rad/s]
      (resumen compacto para el Stanley Controller)

Broadcast TF:
  odom → base_link  (transform completo a 50 Hz)

Frames:
  odom       = frame de referencia global (inicia en (0,0) al arrancar)
  base_link  = centro del vehículo, nivel del suelo
  imu_link   = frame del BNO055 (TF estático en perception.launch.py)
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState

from tf2_ros import TransformBroadcaster


class OdometryNode(Node):

    def __init__(self):
        super().__init__('odometry_node')

        # ── Estado de dead reckoning ────────────────────────────────
        self._x   = 0.0   # m
        self._y   = 0.0   # m
        self._yaw = 0.0   # rad — extraído del IMU
        self._v   = 0.0   # m/s — del encoder
        self._w   = 0.0   # rad/s — del IMU gyro z

        # Quaternión del IMU (se actualiza en _imu_callback)
        self._q_w = 1.0
        self._q_x = 0.0
        self._q_y = 0.0
        self._q_z = 0.0

        self._last_time = None   # para calcular dt

        # ── TF broadcaster ─────────────────────────────────────────
        self._tf_br = TransformBroadcaster(self)

        # ── Publishers ─────────────────────────────────────────────
        self._pub_odom = self.create_publisher(
            Odometry, '/neuracar/odometry', 10)
        self._pub_vel  = self.create_publisher(
            TwistStamped, '/neuracar/velocity', 10)

        # ── Subscribers ────────────────────────────────────────────
        # El encoder lleva la cadencia principal: cada mensaje encoder
        # dispara la actualización de odometría.
        self.create_subscription(
            JointState, '/neuracar/encoder', self._enc_callback, 10)
        # El IMU se guarda en cache; se usa en cada ciclo de encoder.
        self.create_subscription(
            Imu, '/neuracar/imu', self._imu_callback, 10)

        self.get_logger().info('Odometry node iniciado.')

    # ──────────────────────────────────────────────────────────────────
    #  IMU callback — solo actualiza el cache de orientación y vel. angular
    # ──────────────────────────────────────────────────────────────────
    def _imu_callback(self, msg: Imu):
        # Guardar cuaternión completo para la pose en Odometry
        self._q_w = msg.orientation.w
        self._q_x = msg.orientation.x
        self._q_y = msg.orientation.y
        self._q_z = msg.orientation.z

        # Extraer yaw para dead reckoning
        # Conversión cuaternión ZYX → yaw
        siny = 2.0 * (self._q_w * self._q_z + self._q_x * self._q_y)
        cosy = 1.0 - 2.0 * (self._q_y**2 + self._q_z**2)
        self._yaw = math.atan2(siny, cosy)

        # Yaw rate del giroscopio (alrededor de Z, positivo = giro CCW)
        self._w = msg.angular_velocity.z

    # ──────────────────────────────────────────────────────────────────
    #  Encoder callback — dispara la actualización de odometría
    # ──────────────────────────────────────────────────────────────────
    def _enc_callback(self, msg: JointState):
        now = self.get_clock().now()

        # Velocidad lineal del vehículo [m/s] — convenio Neuracar
        # Negativo = marcha atrás (throttle < 0)
        if len(msg.effort) > 0:
            self._v = msg.effort[0]

        # ── Dead reckoning ─────────────────────────────────────────
        if self._last_time is not None:
            dt = (now - self._last_time).nanoseconds * 1e-9

            # Modelo unicycle: integrar posición con heading del IMU
            # x y θ son en el frame 'odom'
            self._x += self._v * math.cos(self._yaw) * dt
            self._y += self._v * math.sin(self._yaw) * dt
            # θ lo da el IMU directamente, no necesitamos integrarlo

        self._last_time = now

        # ── Publicar ───────────────────────────────────────────────
        stamp = now.to_msg()
        self._publish_odometry(stamp)
        self._publish_velocity(stamp)
        self._broadcast_tf(stamp)

    # ──────────────────────────────────────────────────────────────────
    #  Publish: nav_msgs/Odometry
    # ──────────────────────────────────────────────────────────────────
    def _publish_odometry(self, stamp):
        msg                    = Odometry()
        msg.header.stamp       = stamp
        msg.header.frame_id    = 'odom'
        msg.child_frame_id     = 'base_link'

        # Pose — posición por dead reckoning, orientación del IMU
        msg.pose.pose.position.x    = self._x
        msg.pose.pose.position.y    = self._y
        msg.pose.pose.position.z    = 0.0
        msg.pose.pose.orientation.w = self._q_w
        msg.pose.pose.orientation.x = self._q_x
        msg.pose.pose.orientation.y = self._q_y
        msg.pose.pose.orientation.z = self._q_z

        # Twist — en el frame del vehículo (base_link)
        msg.twist.twist.linear.x  = self._v   # m/s hacia adelante
        msg.twist.twist.angular.z = self._w   # rad/s yaw rate

        # Covarianzas (diagonal 6×6 = [xx,xy,xz,xroll,xpitch,xyaw,...])
        # Posición: crece con el tiempo (dead reckoning deriva)
        # Orientación: del IMU, precisión ±1° → 3e-4 rad²
        pose_cov = [
            0.01,  0.0,     0.0,     0.0,     0.0,     0.0,
            0.0,     0.01,  0.0,     0.0,     0.0,     0.0,
            0.0,     0.0,     1e-6,  0.0,     0.0,     0.0,
            0.0,     0.0,     0.0,     1e-6,  0.0,     0.0,
            0.0,     0.0,     0.0,     0.0,     1e-6,  0.0,
            0.0,     0.0,     0.0,     0.0,     0.0,     3e-4,
        ]
        twist_cov = [
            1e-3,  0.0,     0.0,     0.0,     0.0,     0.0,
            0.0,     1e-6,  0.0,     0.0,     0.0,     0.0,
            0.0,     0.0,     1e-6,  0.0,     0.0,     0.0,
            0.0,     0.0,     0.0,     1e-6,  0.0,     0.0,
            0.0,     0.0,     0.0,     0.0,     1e-6,  0.0,
            0.0,     0.0,     0.0,     0.0,     0.0,     1e-4,
        ]
        msg.pose.covariance  = pose_cov
        msg.twist.covariance = twist_cov

        self._pub_odom.publish(msg)

    # ──────────────────────────────────────────────────────────────────
    #  Publish: geometry_msgs/TwistStamped
    #  Versión compacta para el Stanley Controller
    # ──────────────────────────────────────────────────────────────────
    def _publish_velocity(self, stamp):
        msg                     = TwistStamped()
        msg.header.stamp        = stamp
        msg.header.frame_id     = 'base_link'
        msg.twist.linear.x      = self._v   # m/s
        msg.twist.angular.z     = self._w   # rad/s
        self._pub_vel.publish(msg)

    # ──────────────────────────────────────────────────────────────────
    #  Broadcast TF: odom → base_link
    # ──────────────────────────────────────────────────────────────────
    def _broadcast_tf(self, stamp):
        t                          = TransformStamped()
        t.header.stamp             = stamp
        t.header.frame_id          = 'odom'
        t.child_frame_id           = 'base_link'
        t.transform.translation.x  = self._x
        t.transform.translation.y  = self._y
        t.transform.translation.z  = 0.0
        t.transform.rotation.w     = self._q_w
        t.transform.rotation.x     = self._q_x
        t.transform.rotation.y     = self._q_y
        t.transform.rotation.z     = self._q_z
        self._tf_br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()