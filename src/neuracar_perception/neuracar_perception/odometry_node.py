"""
odometry_node.py — NeuraCar
══════════════════════════════════════════════════════════════════
Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026

Fuses AS5047P encoder velocity and BNO055 IMU heading to estimate
vehicle pose via unicycle dead-reckoning at 50 Hz.

  x(t+dt) = x(t) + v·cos(θ)·dt
  y(t+dt) = y(t) + v·sin(θ)·dt

The BNO055 operates in NDOF fusion mode and reports absolute compass
heading (clockwise-positive). The node applies θ_ROS = -θ_compass
to comply with REP 103. On startup, the first IMU reading is saved
as a yaw offset so all subsequent headings are relative to the
vehicle's initial orientation, consistent with the QCar reference
platform behavior.

Subscriptions:
  /neuracar/wheel_speed  std_msgs/Float32        50 Hz  encoder velocity [m/s]
  /neuracar/imu          sensor_msgs/Imu         50 Hz  BNO055 NDOF orientation
  /neuracar/motor_rpm    std_msgs/Float32         50 Hz  diagnostic only

Publications:
  /neuracar/odometry     nav_msgs/Odometry        50 Hz
                         pose: dead-reckoning (x, y) + IMU orientation
                         twist: encoder velocity + IMU yaw rate
  /neuracar/velocity     geometry_msgs/TwistStamped  50 Hz
                         linear.x  = wheel speed [m/s]
                         angular.z = IMU yaw rate [rad/s]

TF broadcast:
  odom → base_link  at 50 Hz

Parameters:
  reset_yaw_on_start (bool, true): Calibrate yaw offset on first IMU
                                   reading. Set false to use absolute
                                   IMU heading (for robot_localization).
══════════════════════════════════════════════════════════════════
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

        self.declare_parameter('reset_yaw_on_start', True)
        self._reset_yaw = self.get_parameter('reset_yaw_on_start').value

        self._x   = 0.0   
        self._y   = 0.0   
        self._yaw = 0.0   
        self._v   = 0.0   
        self._w   = 0.0   

        self._yaw_offset    = 0.0
        self._yaw_calibrated = False

        self._q_w = 1.0
        self._q_x = 0.0
        self._q_y = 0.0
        self._q_z = 0.0

        self._last_time = None

        self._tf_br = TransformBroadcaster(self)

        self.pub_odom = self.create_publisher(
            Odometry,      '/neuracar/odometry', 10)
        self.pub_vel  = self.create_publisher(
            TwistStamped,  '/neuracar/velocity',  10)

        self.create_subscription(
            Float32, '/neuracar/wheel_speed', self._wheel_speed_cb, 10)

        self.create_subscription(
            Imu, '/neuracar/imu', self._imu_cb, 10)

        self.create_subscription(
            Float32, '/neuracar/motor_rpm', self._rpm_cb, 10)

        self.get_logger().info('Odometry node v3.0 iniciado.')
        self.get_logger().info(
            f'  reset_yaw_on_start = {self._reset_yaw}  '
            f'(calibra heading en primera lectura IMU)')
        self.get_logger().info(
            '  Esperando primera lectura IMU para calibrar yaw...')

    def _wheel_speed_cb(self, msg: Float32):
        self._v = float(msg.data)

        now = self.get_clock().now()

        if self._last_time is not None:
            dt = (now - self._last_time).nanoseconds * 1e-9
            if 0.0 < dt < 0.5:  
                self._x += self._v * math.cos(self._yaw) * dt
                self._y += self._v * math.sin(self._yaw) * dt

        self._last_time = now

        stamp = now.to_msg()
        self._publish_odometry(stamp)
        self._publish_velocity(stamp)
        self._broadcast_tf(stamp)

    def _imu_cb(self, msg: Imu):
       
        qw = msg.orientation.w
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z

        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy**2 + qz**2)
        yaw_raw = math.atan2(siny, cosy)

        if self._reset_yaw and not self._yaw_calibrated:
            self._yaw_offset     = yaw_raw
            self._yaw_calibrated = True
            self.get_logger().info(
                f'Yaw calibrado. Offset = {math.degrees(yaw_raw):.2f}° '
                f'(heading absoluto del BNO055 al arrancar)'
            )

        if self._reset_yaw:
            yaw_cal = yaw_raw - self._yaw_offset
            while yaw_cal >  math.pi: yaw_cal -= 2.0 * math.pi
            while yaw_cal <= -math.pi: yaw_cal += 2.0 * math.pi
        else:
            yaw_cal = yaw_raw 

        self._yaw = yaw_cal

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

        self._w = msg.angular_velocity.z


    def _rpm_cb(self, msg: Float32):
        pass


    def _publish_odometry(self, stamp):
        msg                 = Odometry()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id  = 'base_link'

        msg.pose.pose.position.x    = self._x
        msg.pose.pose.position.y    = self._y
        msg.pose.pose.position.z    = 0.0
        msg.pose.pose.orientation.w = self._q_w
        msg.pose.pose.orientation.x = self._q_x
        msg.pose.pose.orientation.y = self._q_y
        msg.pose.pose.orientation.z = self._q_z

        msg.twist.twist.linear.x  = self._v  
        msg.twist.twist.angular.z = self._w   

        pc = 0.0
        msg.pose.covariance = [
            0.01, pc,   pc,   pc,   pc,   pc,
            pc,   0.01, pc,   pc,   pc,   pc,
            pc,   pc,   1e-6, pc,   pc,   pc,
            pc,   pc,   pc,   1e-6, pc,   pc,
            pc,   pc,   pc,   pc,   1e-6, pc,
            pc,   pc,   pc,   pc,   pc,   3e-4,
        ]

        msg.twist.covariance = [
            1e-3, pc,   pc,   pc,   pc,   pc,
            pc,   1e-6, pc,   pc,   pc,   pc,
            pc,   pc,   1e-6, pc,   pc,   pc,
            pc,   pc,   pc,   1e-6, pc,   pc,
            pc,   pc,   pc,   pc,   1e-6, pc,
            pc,   pc,   pc,   pc,   pc,   1e-4,
        ]

        self.pub_odom.publish(msg)

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