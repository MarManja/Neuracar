#!/usr/bin/env python3
"""
esp32_sensores_node.py — NeuraCar
══════════════════════════════════════════════════════════════════
Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026

Serial bridge for the sensing ESP32 (port /dev/esp32s, 921600 baud).
Parses incoming CSV frames from the ESP32 firmware and publishes
structured ROS 2 messages.

Serial frame types received:
  E,{angleDeg},{motorRPM},{vLinear}                    @ 50 Hz  encoder
  I,{yaw},{roll},{pitch},{ax},{ay},{az},{gx},{gy},{gz}  @ 50 Hz  IMU
  STA,{code}                                           async    status events
    STA codes: ENC_SYNC, IMU_OK, ERR_IMU,
               WARN_OLED_NOT_FOUND, READY_SENSORES,
               SHUTDOWN_ACK

Publications:
  /neuracar/wheel_speed   std_msgs/Float32        50 Hz
                          linear velocity [m/s], positive=forward
  /neuracar/motor_rpm     std_msgs/Float32        50 Hz
                          motor shaft RPM (before gear reduction)
  /neuracar/imu           sensor_msgs/Imu         50 Hz
                          BNO055 NDOF; heading compass convention;
                          acceleration [m/s²]; angular velocity [rad/s]
  /neuracar/status        std_msgs/String         async
                          raw STA frame string
  /neuracar/system_status std_msgs/String         async
                          parsed status code

Parameters:
  port         (str,   /dev/esp32s): Serial port — must be USB Type-C
                                     port on Jetson carrier board
  baudrate     (int,   921600):      Serial baud rate
  wheel_radius (float, 0.033):       Wheel radius [m]
  gear_ratio   (float, 9.2459):      Motor-to-wheel gear ratio
══════════════════════════════════════════════════════════════════
"""

import math
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32, String

try:
    import serial
except ImportError:
    raise SystemExit('pip install pyserial --break-system-packages')

class SensoresBridgeNode(Node):

    def __init__(self):
        super().__init__('esp32_sensores_node')

        self.declare_parameter('port',         '/dev/esp32s')
        self.declare_parameter('baudrate',     921600)
        self.declare_parameter('wheel_radius', 0.033)
        self.declare_parameter('gear_ratio',   9.2459)

        port        = self.get_parameter('port').value
        baud        = self.get_parameter('baudrate').value
        self._R     = self.get_parameter('wheel_radius').value
        self._ratio = self.get_parameter('gear_ratio').value

        try:
            self._ser = serial.Serial(port, baud, timeout=0.1)
            self.get_logger().info(f'ESP32-S connected: {port} @ {baud}')
        except serial.SerialException as exc:
            self.get_logger().fatal(f'Cannot open {port}: {exc}')
            raise

        self.pub_enc = self.create_publisher(
            JointState, '/neuracar/encoder', 10)

        self.pub_wheel_speed = self.create_publisher(
            Float32, '/neuracar/wheel_speed', 10)

        self.pub_motor_rpm = self.create_publisher(
            Float32, '/neuracar/motor_rpm', 10)

        self.pub_imu = self.create_publisher(
            Imu, '/neuracar/imu', 10)

        self.pub_status = self.create_publisher(
            String, '/neuracar/status', 10)
        self.pub_sys = self.create_publisher(
            String, '/neuracar/system_status', 10)

        self._alive  = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self.get_logger().info(
            f'Sensores bridge ready — '
            f'wheel_radius={self._R} m  gear_ratio={self._ratio}')
        self._pub_sys('SENSORES_READY')

    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _pub_sys(self, s: str):
        m = String(); m.data = s
        self.pub_sys.publish(m)

    def _read_loop(self):
        while self._alive:
            try:
                raw = self._ser.readline()
                if not raw:
                    continue
                line = raw.decode('utf-8', errors='ignore').strip()
                if line:
                    self._parse(line)
            except serial.SerialException as exc:
                if self._alive:
                    self.get_logger().error(f'Serial error: {exc}')
                break
            except Exception as exc:
                self.get_logger().warn(f'Parse error: {exc}')

    def _parse(self, line: str):
        parts = line.split(',')
        try:
            if parts[0] == 'E' and len(parts) == 4:
                self._enc(float(parts[1]), float(parts[2]), float(parts[3]))
            elif parts[0] == 'I' and len(parts) == 10:
                self._imu(*[float(p) for p in parts[1:]])
            elif parts[0] == 'STA':
                self._sta(','.join(parts[1:]))
        except (ValueError, IndexError) as exc:
            self.get_logger().warn(f'Malformed frame "{line}": {exc}')

    def _enc(self, angle_deg: float, motor_rpm: float, v_linear: float):

        stamp = self._stamp()

        motor_rad_s = motor_rpm * (2.0 * math.pi / 60.0)
        wheel_rad_s = motor_rad_s / self._ratio
        angle_rad   = math.radians(angle_deg)

        enc_msg              = JointState()
        enc_msg.header.stamp = stamp
        enc_msg.name         = ['drive_motor']
        enc_msg.position     = [angle_rad]
        enc_msg.velocity     = [wheel_rad_s]
        enc_msg.effort       = []
        self.pub_enc.publish(enc_msg)

        ws_msg      = Float32()
        ws_msg.data = float(v_linear)
        self.pub_wheel_speed.publish(ws_msg)

        rpm_msg      = Float32()
        rpm_msg.data = float(motor_rpm)
        self.pub_motor_rpm.publish(rpm_msg)


    def _imu(self, heading, roll, pitch, ax, ay, az, gx, gy, gz):
        msg                 = Imu()
        msg.header.stamp    = self._stamp()
        msg.header.frame_id = 'imu_link'

        # Euler (ZYX) → Quaternion
        yaw = -math.radians(heading)   
        r   =  math.radians(roll)
        p   =  math.radians(pitch)

        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(p   * 0.5), math.sin(p   * 0.5)
        cr, sr = math.cos(r   * 0.5), math.sin(r   * 0.5)

        msg.orientation.w = cr * cp * cy + sr * sp * sy
        msg.orientation.x = sr * cp * cy - cr * sp * sy
        msg.orientation.y = cr * sp * cy + sr * cp * sy
        msg.orientation.z = cr * cp * sy - sr * sp * cy

        msg.angular_velocity.x    = gx
        msg.angular_velocity.y    = gy
        msg.angular_velocity.z    = gz
        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az

        def diag(v):
            return [v, 0., 0.,  0., v, 0.,  0., 0., v]

        msg.orientation_covariance         = diag(3e-4)
        msg.angular_velocity_covariance    = diag(1e-4)
        msg.linear_acceleration_covariance = diag(2e-3)

        self.pub_imu.publish(msg)

    def _sta(self, status: str):
        m = String(); m.data = status
        self.pub_status.publish(m)

        if 'ERR' in status:
            self.get_logger().error(f'[ESP32-S] {status}')
        elif 'WARN' in status:
            self.get_logger().warn(f'[ESP32-S] {status}')
        else:
            self.get_logger().info(f'[ESP32-S] {status}')


    def destroy_node(self):
        self._alive = False
        try:
            self._ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SensoresBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()