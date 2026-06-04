#!/usr/bin/env python3
"""
odometry_node.py — Neuracar v3.0
===================================
Fusiona encoder + IMU para producir pose y velocidad del vehículo.

Cambios v3.0 vs v2.0:
  - Calibración de yaw en la primera lectura del IMU (yaw_offset).
    El BNO055 en modo NDOF reporta heading absoluto de brújula —
    si el carro cambia de orientación entre la grabación de la
    trayectoria y la ejecución del controlador, el heading difiere
    y el Stanley ve un psi_e enorme desde el primer frame.
    Con yaw_offset el heading siempre arranca en 0° relativo,
    igual que hacía pose_ekf_qcar_2.py en el QCar.

  - El quaternión publicado también se rota por yaw_offset para que
    el TF odom→base_link sea consistente con el yaw calibrado.

  - Parámetro reset_yaw_on_start (bool, default True): si False,
    desactiva la calibración y usa el heading absoluto del IMU
    (útil si se integra con robot_localization en el futuro).

Suscribe:
  /neuracar/wheel_speed  (std_msgs/Float32)       50 Hz
      data = velocidad lineal vehículo [m/s]      ← fuente principal de v
  /neuracar/imu          (sensor_msgs/Imu)        50 Hz
      orientation       = quaternión BNO055 NDOF  ← fuente de θ
      angular_velocity.z = yaw rate [rad/s]
  /neuracar/motor_rpm    (std_msgs/Float32)        50 Hz  (solo diagnóstico)

Publica:
  /neuracar/odometry   (nav_msgs/Odometry)         50 Hz
      pose.pose.position.x/y  [m]    dead reckoning integrado
      pose.pose.orientation   quat   directo del IMU (no integrado)
      twist.twist.linear.x    [m/s]  del encoder
      twist.twist.angular.z   [rad/s] del IMU giroscopio z

  /neuracar/velocity   (geometry_msgs/TwistStamped) 50 Hz
      twist.linear.x   [m/s]   velocidad lineal
      twist.angular.z  [rad/s] yaw rate

Broadcast TF:
  odom → base_link  a 50 Hz

Frames:
  odom      = frame global, origen en (0,0) al arrancar
  base_link = centro del vehículo a nivel del suelo
  imu_link  = frame BNO055 (TF estático en launch)
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster


class OdometryNode(Node):

    def __init__(self):
        super().__init__('odometry_node')

        # ── Parámetro de calibración ────────────────────────────────
        self.declare_parameter('reset_yaw_on_start', True)
        self._reset_yaw = self.get_parameter('reset_yaw_on_start').value

        # ── Estado dead reckoning ────────────────────────────────────
        self._x   = 0.0   # m — posición global X
        self._y   = 0.0   # m — posición global Y
        self._yaw = 0.0   # rad — yaw calibrado (relativo al arranque)
        self._v   = 0.0   # m/s — del encoder via /neuracar/wheel_speed
        self._w   = 0.0   # rad/s — del IMU angular_velocity.z

        # Calibración de yaw — igual que pose_ekf_qcar_2.py
        # Primera lectura del IMU → se guarda como offset
        # A partir de ahí: yaw_calibrado = yaw_imu - yaw_offset
        # Efecto: el heading siempre arranca en 0° sin importar la
        # orientación absoluta del BNO055 al encender.
        self._yaw_offset    = 0.0
        self._yaw_calibrated = False

        # Quaternión calibrado — se reconstruye en _imu_cb tras aplicar offset
        self._q_w = 1.0
        self._q_x = 0.0
        self._q_y = 0.0
        self._q_z = 0.0

        self._last_time = None

        # ── TF broadcaster ───────────────────────────────────────────
        self._tf_br = TransformBroadcaster(self)

        # ── Publishers ───────────────────────────────────────────────
        self.pub_odom = self.create_publisher(
            Odometry,      '/neuracar/odometry', 10)
        self.pub_vel  = self.create_publisher(
            TwistStamped,  '/neuracar/velocity',  10)

        # ── Subscribers ──────────────────────────────────────────────

        # Velocidad lineal — topic dedicado, semántica correcta
        self.create_subscription(
            Float32, '/neuracar/wheel_speed', self._wheel_speed_cb, 10)

        # IMU — cache de orientación y yaw rate
        self.create_subscription(
            Imu, '/neuracar/imu', self._imu_cb, 10)

        # RPM motor — solo diagnóstico, no afecta odometría
        self.create_subscription(
            Float32, '/neuracar/motor_rpm', self._rpm_cb, 10)

        self.get_logger().info('Odometry node v3.0 iniciado.')
        self.get_logger().info(
            f'  reset_yaw_on_start = {self._reset_yaw}  '
            f'(calibra heading en primera lectura IMU)')
        self.get_logger().info(
            '  Esperando primera lectura IMU para calibrar yaw...')

    # ── Subscriber: velocidad lineal del vehículo ────────────────────

    def _wheel_speed_cb(self, msg: Float32):
        """
        Recibe velocidad lineal del vehículo [m/s] desde el sensores_node.
        Positivo = adelante, negativo = atrás.
        Dispara la actualización de odometría (misma cadencia que encoder).
        """
        self._v = float(msg.data)

        now = self.get_clock().now()

        # Dead reckoning — modelo unicycle
        # x = x + v·cos(θ)·dt
        # y = y + v·sin(θ)·dt
        # θ viene directo del IMU (absoluto, no integrado) → no acumula drift
        if self._last_time is not None:
            dt = (now - self._last_time).nanoseconds * 1e-9
            if 0.0 < dt < 0.5:   # ignorar saltos grandes (pausa/reanudación)
                self._x += self._v * math.cos(self._yaw) * dt
                self._y += self._v * math.sin(self._yaw) * dt

        self._last_time = now

        stamp = now.to_msg()
        self._publish_odometry(stamp)
        self._publish_velocity(stamp)
        self._broadcast_tf(stamp)

    # ── Subscriber: IMU ─────────────────────────────────────────────

    def _imu_cb(self, msg: Imu):
        """
        Guarda orientación calibrada y yaw rate.

        Calibración de yaw (reset_yaw_on_start=True):
          1. Primera lectura: yaw_offset = yaw_imu_raw
          2. Lecturas siguientes: yaw = normalize(yaw_imu_raw - yaw_offset)
          3. El quaternión publicado se reconstruye desde el yaw calibrado
             manteniendo roll y pitch del IMU sin modificar.

        Por qué es necesario:
          El BNO055 en modo NDOF fusiona giroscopio + acelerómetro +
          magnetómetro → produce heading absoluto de brújula.
          Si el carro se mueve o el IMU recalibra entre sesiones,
          el heading cambia aunque el carro esté en el mismo sitio.
          Sin calibración, el Stanley ve psi_e enorme al arrancar.
        """
        # Extraer yaw raw del quaternión del IMU
        qw = msg.orientation.w
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z

        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy**2 + qz**2)
        yaw_raw = math.atan2(siny, cosy)

        # ── Calibración en la primera lectura ─────────────────────
        if self._reset_yaw and not self._yaw_calibrated:
            self._yaw_offset     = yaw_raw
            self._yaw_calibrated = True
            self.get_logger().info(
                f'Yaw calibrado. Offset = {math.degrees(yaw_raw):.2f}° '
                f'(heading absoluto del BNO055 al arrancar)'
            )

        # ── Aplicar offset ────────────────────────────────────────
        if self._reset_yaw:
            yaw_cal = yaw_raw - self._yaw_offset
            # Normalizar a (−π, π]
            while yaw_cal >  math.pi: yaw_cal -= 2.0 * math.pi
            while yaw_cal <= -math.pi: yaw_cal += 2.0 * math.pi
        else:
            yaw_cal = yaw_raw   # heading absoluto, sin calibrar

        self._yaw = yaw_cal

        # ── Reconstruir quaternión desde yaw calibrado ────────────
        # Mantiene roll y pitch del IMU, reemplaza solo el componente yaw.
        # Extrae roll y pitch del quaternión original:
        roll  = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = math.asin(max(-1.0, min(1.0, 2*(qw*qy - qz*qx))))

        cy = math.cos(yaw_cal * 0.5)
        sy = math.sin(yaw_cal * 0.5)
        cp = math.cos(pitch   * 0.5)
        sp = math.sin(pitch   * 0.5)
        cr = math.cos(roll    * 0.5)
        sr = math.sin(roll    * 0.5)

        self._q_w = cr * cp * cy + sr * sp * sy
        self._q_x = sr * cp * cy - cr * sp * sy
        self._q_y = cr * sp * cy + sr * cp * sy
        self._q_z = cr * cp * sy - sr * sp * cy

        # Yaw rate — positivo = CCW (ROS2), del giroscopio BNO055
        self._w = msg.angular_velocity.z

    # ── Subscriber: RPM motor (solo diagnóstico) ─────────────────────

    def _rpm_cb(self, msg: Float32):
        # No afecta la odometría — disponible para debug si se necesita
        pass

    # ── Publish: nav_msgs/Odometry ───────────────────────────────────

    def _publish_odometry(self, stamp):
        """
        Covarianzas de pose (6×6 aplanada, diagonal):
          x, y       : 0.01 m²  — incertidumbre posición crece con distancia
          z          : 1e-6     — vehículo plano, z conocido
          roll, pitch: 1e-6     — superficie plana asumida
          yaw        : 3e-4 rad² — del IMU BNO055 (±1° RMS)

        Covarianzas de twist (6×6 aplanada, diagonal):
          linear.x   : 1e-3 (m/s)²  — encoder AS5047P, ruido bajo
          linear.y   : 1e-6         — no hay movimiento lateral medido
          linear.z   : 1e-6         — no hay movimiento vertical
          angular.x/y: 1e-6         — no relevantes para vehículo plano
          angular.z  : 1e-4 (rad/s)² — giroscopio BNO055
        """
        msg                 = Odometry()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id  = 'base_link'

        # Pose — posición por dead reckoning, orientación absoluta del IMU
        msg.pose.pose.position.x    = self._x
        msg.pose.pose.position.y    = self._y
        msg.pose.pose.position.z    = 0.0
        msg.pose.pose.orientation.w = self._q_w
        msg.pose.pose.orientation.x = self._q_x
        msg.pose.pose.orientation.y = self._q_y
        msg.pose.pose.orientation.z = self._q_z

        # Twist en el frame del vehículo (base_link)
        msg.twist.twist.linear.x  = self._v   # m/s adelante
        msg.twist.twist.angular.z = self._w   # rad/s yaw rate

        # Covarianza pose 6×6 diagonal
        # [x, y, z, roll, pitch, yaw]
        pc = 0.0
        msg.pose.covariance = [
            0.01, pc,   pc,   pc,   pc,   pc,
            pc,   0.01, pc,   pc,   pc,   pc,
            pc,   pc,   1e-6, pc,   pc,   pc,
            pc,   pc,   pc,   1e-6, pc,   pc,
            pc,   pc,   pc,   pc,   1e-6, pc,
            pc,   pc,   pc,   pc,   pc,   3e-4,
        ]

        # Covarianza twist 6×6 diagonal
        # [vx, vy, vz, wx, wy, wz]
        msg.twist.covariance = [
            1e-3, pc,   pc,   pc,   pc,   pc,
            pc,   1e-6, pc,   pc,   pc,   pc,
            pc,   pc,   1e-6, pc,   pc,   pc,
            pc,   pc,   pc,   1e-6, pc,   pc,
            pc,   pc,   pc,   pc,   1e-6, pc,
            pc,   pc,   pc,   pc,   pc,   1e-4,
        ]

        self.pub_odom.publish(msg)

    # ── Publish: geometry_msgs/TwistStamped ─────────────────────────

    def _publish_velocity(self, stamp):
        """
        Resumen compacto de velocidad para el Stanley Controller.
        Solo linear.x [m/s] y angular.z [rad/s] — lo que el controlador necesita.
        """
        msg                   = TwistStamped()
        msg.header.stamp      = stamp
        msg.header.frame_id   = 'base_link'
        msg.twist.linear.x    = self._v   # m/s
        msg.twist.angular.z   = self._w   # rad/s
        self.pub_vel.publish(msg)

    # ── Broadcast TF: odom → base_link ──────────────────────────────

    def _broadcast_tf(self, stamp):
        t                         = TransformStamped()
        t.header.stamp            = stamp
        t.header.frame_id         = 'odom'
        t.child_frame_id          = 'base_link'
        t.transform.translation.x = self._x
        t.transform.translation.y = self._y
        t.transform.translation.z = 0.0
        t.transform.rotation.w    = self._q_w
        t.transform.rotation.x    = self._q_x
        t.transform.rotation.y    = self._q_y
        t.transform.rotation.z    = self._q_z
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