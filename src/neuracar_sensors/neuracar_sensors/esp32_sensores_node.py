#!/usr/bin/env python3
"""
esp32_sensores_bridge.py — Neuracar v1.0
==========================================
Nodo exclusivo para ESP32-S (sensores).
Lee encoder, IMU y batería. Sin actuadores.

Publica:
  /neuracar/imu            (sensor_msgs/Imu)
  /neuracar/encoder        (sensor_msgs/JointState)
  /neuracar/battery        (sensor_msgs/BatteryState)
  /neuracar/status         (std_msgs/String)
  /neuracar/system_status  (std_msgs/String)
"""

import math
import subprocess
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Imu, JointState
from std_msgs.msg import String

try:
    import serial
except ImportError:
    raise SystemExit('pip install pyserial --break-system-packages')


class SensoresBridgeNode(Node):

    def __init__(self):
        super().__init__('esp32_sensores_bridge')

        self.declare_parameter('port',          '/dev/esp32s')
        self.declare_parameter('baudrate',      921600)
        self.declare_parameter('wheel_radius',  0.033)
        self.declare_parameter('gear_ratio',    9.2459)
        self.declare_parameter('auto_shutdown', True)

        port  = self.get_parameter('port').value
        baud  = self.get_parameter('baudrate').value
        self._R      = self.get_parameter('wheel_radius').value
        self._ratio  = self.get_parameter('gear_ratio').value
        self._auto_sd= self.get_parameter('auto_shutdown').value

        self._bat_state      = 'NORMAL'
        self._shutdown_sent  = False
        self._shutdown_timer = None

        try:
            self._ser = serial.Serial(port, baud, timeout=0.1)
            self.get_logger().info(f'ESP32-S conectada: {port} @ {baud}')
        except serial.SerialException as exc:
            self.get_logger().fatal(f'No se pudo abrir {port}: {exc}')
            raise

        self.pub_imu    = self.create_publisher(Imu,          '/neuracar/imu',           10)
        self.pub_enc    = self.create_publisher(JointState,   '/neuracar/encoder',        10)
        self.pub_bat    = self.create_publisher(BatteryState, '/neuracar/battery',        10)
        self.pub_status = self.create_publisher(String,       '/neuracar/status',         10)
        self.pub_sys    = self.create_publisher(String,       '/neuracar/system_status',  10)

        self._alive  = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self.get_logger().info('Sensores bridge listo.')
        self._pub_sys('SENSORES_READY')

    # ── Helpers ─────────────────────────────────────────────────────

    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _pub_sys(self, s: str):
        m = String(); m.data = s; self.pub_sys.publish(m)

    def _send(self, data: bytes):
        try:
            self._ser.write(data)
        except serial.SerialException as exc:
            self.get_logger().error(f'Error escritura serial: {exc}')

    # ── Lector serial ───────────────────────────────────────────────

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

    # ── Parser ──────────────────────────────────────────────────────

    def _parse(self, line: str):
        parts = line.split(',')
        try:
            if parts[0] == 'E' and len(parts) == 4:
                self._enc(float(parts[1]), float(parts[2]), float(parts[3]))
            elif parts[0] == 'I' and len(parts) == 10:
                self._imu(*[float(p) for p in parts[1:]])
            elif parts[0] == 'B' and len(parts) == 7:
                self._bat(float(parts[1]), float(parts[2]), float(parts[3]),
                          float(parts[4]) / 100.0, int(parts[5]), int(parts[6]))
            elif parts[0] == 'STA':
                self._sta(','.join(parts[1:]))
        except (ValueError, IndexError) as exc:
            self.get_logger().warn(f'Malformado "{line}": {exc}')

    # ── Encoder ─────────────────────────────────────────────────────

    def _enc(self, angle_deg: float, motor_rpm: float, v_linear: float):
        msg              = JointState()
        msg.header.stamp = self._stamp()
        msg.name         = ['drive_motor']
        wheel_rad_s      = (motor_rpm / self._ratio) * (2.0 * math.pi / 60.0)
        msg.position     = [math.radians(angle_deg)]
        msg.velocity     = [wheel_rad_s]
        msg.effort       = [v_linear]
        self.pub_enc.publish(msg)

    # ── IMU ─────────────────────────────────────────────────────────

    def _imu(self, heading, roll, pitch, ax, ay, az, gx, gy, gz):
        msg                 = Imu()
        msg.header.stamp    = self._stamp()
        msg.header.frame_id = 'imu_link'

        yaw = -math.radians(heading)   # CW → CCW (ROS)
        r   =  math.radians(roll)
        p   =  math.radians(pitch)

        cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
        cp, sp = math.cos(p  *0.5), math.sin(p  *0.5)
        cr, sr = math.cos(r  *0.5), math.sin(r  *0.5)

        msg.orientation.w = cr*cp*cy + sr*sp*sy
        msg.orientation.x = sr*cp*cy - cr*sp*sy
        msg.orientation.y = cr*sp*cy + sr*cp*sy
        msg.orientation.z = cr*cp*sy - sr*sp*cy

        msg.angular_velocity.x    = gx
        msg.angular_velocity.y    = gy
        msg.angular_velocity.z    = gz
        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az

        c3 = [3e-4, 0.0, 0.0,  0.0, 3e-4, 0.0,  0.0, 0.0, 3e-4]
        c1 = [1e-4, 0.0, 0.0,  0.0, 1e-4, 0.0,  0.0, 0.0, 1e-4]
        c2 = [2e-3, 0.0, 0.0,  0.0, 2e-3, 0.0,  0.0, 0.0, 2e-3]
        msg.orientation_covariance         = c3
        msg.angular_velocity_covariance    = c1
        msg.linear_acceleration_covariance = c2

        self.pub_imu.publish(msg)

    # ── Batería ─────────────────────────────────────────────────────

    def _bat(self, voltage, current, power, soc, warn, crit):
        msg                         = BatteryState()
        msg.header.stamp            = self._stamp()
        msg.voltage                 = voltage
        msg.current                 = current
        msg.design_capacity         = 3.3
        msg.percentage              = float(max(0.0, min(1.0, soc)))
        msg.power_supply_status     = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
        msg.present                 = True
        msg.power_supply_health = (
            BatteryState.POWER_SUPPLY_HEALTH_DEAD
            if crit else BatteryState.POWER_SUPPLY_HEALTH_GOOD)
        self.pub_bat.publish(msg)

        if warn and self._bat_state == 'NORMAL':
            self._bat_state = 'LOW'
            self._pub_sys('LOW_BAT')
            self.get_logger().warn(f'Batería baja: {voltage:.2f}V  SoC={soc*100:.0f}%')

        if crit and self._bat_state not in ('CRITICAL', 'SHUTDOWN'):
            self._bat_state = 'CRITICAL'
            self._pub_sys('CRITICAL_BAT')
            self.get_logger().error(f'Batería crítica: {voltage:.2f}V')
            self._initiate_shutdown()

    # ── STA ─────────────────────────────────────────────────────────

    def _sta(self, status: str):
        m = String(); m.data = status
        self.pub_status.publish(m)
        if 'ERR' in status:
            self.get_logger().error(f'[ESP32-S] {status}')
        elif 'WARN' in status or 'ESTOP' in status:
            self.get_logger().warn(f'[ESP32-S] {status}')
        else:
            self.get_logger().info(f'[ESP32-S] {status}')

        if status == 'CRITICAL_BAT' and self._bat_state not in ('CRITICAL','SHUTDOWN'):
            self._bat_state = 'CRITICAL'
            self._initiate_shutdown()

    # ── Shutdown ─────────────────────────────────────────────────────

    def _initiate_shutdown(self):
        if self._shutdown_sent:
            return
        self._shutdown_sent = True
        self._send(b'SHUTDOWN\n')
        self.get_logger().warn('Apagado del SO en 5s...')
        if self._auto_sd:
            self._shutdown_timer = threading.Timer(5.0, self._do_shutdown)
            self._shutdown_timer.daemon = True
            self._shutdown_timer.start()

    def _do_shutdown(self):
        self._bat_state = 'SHUTDOWN'
        self._pub_sys('SHUTTING_DOWN')
        try:
            subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=True)
        except subprocess.CalledProcessError as exc:
            self.get_logger().error(f'Shutdown fallido: {exc}')

    # ── Cleanup ──────────────────────────────────────────────────────

    def destroy_node(self):
        self._alive = False
        if self._shutdown_timer and self._shutdown_timer.is_alive():
            self._shutdown_timer.cancel()
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