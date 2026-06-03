#!/usr/bin/env python3
"""
esp32_sensores_node.py — Neuracar v2.0
========================================
Nodo exclusivo para ESP32-S (sensores).
Lee encoder, IMU y batería. Sin actuadores.

Publica:
  /neuracar/encoder        (sensor_msgs/JointState)    50 Hz
      name[0]       = 'drive_motor'
      position[0]   = ángulo acumulado motor   [rad]   sin wrap
      velocity[0]   = velocidad angular rueda  [rad/s]

  /neuracar/wheel_speed    (std_msgs/Float32)           50 Hz
      data = velocidad lineal vehículo  [m/s]
      positivo = adelante, negativo = atrás

  /neuracar/motor_rpm      (std_msgs/Float32)           50 Hz
      data = RPM eje motor (antes de reducción)

  /neuracar/imu            (sensor_msgs/Imu)            50 Hz
  /neuracar/battery        (sensor_msgs/BatteryState)   10 Hz
  /neuracar/status         (std_msgs/String)
  /neuracar/system_status  (std_msgs/String)
"""

import math
import subprocess
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Imu, JointState
from std_msgs.msg import Float32, String

try:
    import serial
except ImportError:
    raise SystemExit('pip install pyserial --break-system-packages')


class SensoresBridgeNode(Node):

    def __init__(self):
        super().__init__('esp32_sensores_node')

        # ── Parámetros ──────────────────────────────────────────────
        self.declare_parameter('port',          '/dev/esp32s')
        self.declare_parameter('baudrate',      921600)
        self.declare_parameter('wheel_radius',  0.033)   # m
        self.declare_parameter('gear_ratio',    9.2459)  # motor/rueda
        self.declare_parameter('auto_shutdown', True)

        port          = self.get_parameter('port').value
        baud          = self.get_parameter('baudrate').value
        self._R       = self.get_parameter('wheel_radius').value
        self._ratio   = self.get_parameter('gear_ratio').value
        self._auto_sd = self.get_parameter('auto_shutdown').value

        # ── Estado batería ───────────────────────────────────────────
        self._bat_state      = 'NORMAL'
        self._shutdown_sent  = False
        self._shutdown_timer = None

        # ── Serial ──────────────────────────────────────────────────
        try:
            self._ser = serial.Serial(port, baud, timeout=0.1)
            self.get_logger().info(f'ESP32-S conectada: {port} @ {baud}')
        except serial.SerialException as exc:
            self.get_logger().fatal(f'No se pudo abrir {port}: {exc}')
            raise

        # ── Publishers ──────────────────────────────────────────────

        # Encoder — cinemática pura del eje
        # position [rad] acumulado sin wrap, velocity [rad/s] de la rueda
        # effort vacío — no se usa, semánticamente incorrecto para velocidad
        self.pub_enc = self.create_publisher(
            JointState, '/neuracar/encoder', 10)

        # Velocidad lineal del vehículo [m/s]
        # Topic dedicado con semántica clara — lo consume odometry_node
        self.pub_wheel_speed = self.create_publisher(
            Float32, '/neuracar/wheel_speed', 10)

        # RPM del eje motor (antes de la reducción)
        # Útil para diagnóstico y verificar gear_ratio
        self.pub_motor_rpm = self.create_publisher(
            Float32, '/neuracar/motor_rpm', 10)

        # IMU — orientación, velocidad angular, aceleración lineal
        self.pub_imu = self.create_publisher(
            Imu, '/neuracar/imu', 10)

        # Batería
        self.pub_bat = self.create_publisher(
            BatteryState, '/neuracar/battery', 10)

        # Eventos firmware y estado del sistema
        self.pub_status = self.create_publisher(
            String, '/neuracar/status', 10)
        self.pub_sys = self.create_publisher(
            String, '/neuracar/system_status', 10)

        # ── Hilo lector serial ───────────────────────────────────────
        self._alive  = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self.get_logger().info('Sensores bridge v2.0 listo.')
        self.get_logger().info(
            f'  wheel_radius={self._R} m  gear_ratio={self._ratio}')
        self._pub_sys('SENSORES_READY')

    # ── Helpers ─────────────────────────────────────────────────────

    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _pub_sys(self, s: str):
        m = String(); m.data = s
        self.pub_sys.publish(m)

    def _send(self, data: bytes):
        try:
            self._ser.write(data)
        except serial.SerialException as exc:
            self.get_logger().error(f'Error escritura serial: {exc}')

    # ── Lector serial ────────────────────────────────────────────────

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

    # ── Parser ───────────────────────────────────────────────────────

    def _parse(self, line: str):
        parts = line.split(',')
        try:
            if parts[0] == 'E' and len(parts) == 4:
                self._enc(float(parts[1]), float(parts[2]), float(parts[3]))
            elif parts[0] == 'I' and len(parts) == 10:
                self._imu(*[float(p) for p in parts[1:]])
            elif parts[0] == 'B' and len(parts) == 7:
                self._bat(
                    float(parts[1]),          # voltage  [V]
                    float(parts[2]),          # current  [A]
                    float(parts[3]),          # power    [W]
                    float(parts[4]) / 100.0,  # soc 0-100 → 0-1
                    int(parts[5]),            # warn flag
                    int(parts[6]),            # crit flag
                )
            elif parts[0] == 'STA':
                self._sta(','.join(parts[1:]))
        except (ValueError, IndexError) as exc:
            self.get_logger().warn(f'Malformado "{line}": {exc}')

    # ── Encoder ─────────────────────────────────────────────────────

    def _enc(self, angle_deg: float, motor_rpm: float, v_linear: float):
        """
        Del firmware ESP32-S:
          angle_deg : ángulo acumulado del motor [grados], sin wrap, continuo
          motor_rpm : RPM del eje motor (antes de la reducción)
          v_linear  : velocidad lineal del vehículo [m/s], ya calculada en firmware
                      = (omega_motor / gear_ratio) × wheel_radius

        Conversiones aquí:
          motor_rad_s  = motor_rpm × 2π/60         [rad/s] eje motor
          wheel_rad_s  = motor_rad_s / gear_ratio   [rad/s] eje rueda
          angle_rad    = angle_deg × π/180           [rad]  acumulado motor
        """
        stamp = self._stamp()

        # ── Cinemática ──────────────────────────────────────────────
        motor_rad_s = motor_rpm * (2.0 * math.pi / 60.0)  # rad/s motor
        wheel_rad_s = motor_rad_s / self._ratio            # rad/s rueda
        angle_rad   = math.radians(angle_deg)              # rad acumulado

        # ── JointState — semántica ROS2 correcta ────────────────────
        #   position [rad] : ángulo acumulado del eje motor
        #   velocity [rad/s]: velocidad angular de la RUEDA (post-reducción)
        #   effort   []    : vacío — no hay torque medido
        enc_msg          = JointState()
        enc_msg.header.stamp = stamp
        enc_msg.name     = ['drive_motor']
        enc_msg.position = [angle_rad]
        enc_msg.velocity = [wheel_rad_s]
        enc_msg.effort   = []   # vacío — sin torque medido
        self.pub_enc.publish(enc_msg)

        # ── Velocidad lineal — topic dedicado ────────────────────────
        #   Positivo = adelante, negativo = atrás
        #   Fuente: calculado en firmware con gear_ratio y wheel_radius
        ws_msg      = Float32()
        ws_msg.data = float(v_linear)   # m/s
        self.pub_wheel_speed.publish(ws_msg)

        # ── RPM motor — para diagnóstico ────────────────────────────
        rpm_msg      = Float32()
        rpm_msg.data = float(motor_rpm)
        self.pub_motor_rpm.publish(rpm_msg)

    # ── IMU ─────────────────────────────────────────────────────────

    def _imu(self, heading, roll, pitch, ax, ay, az, gx, gy, gz):
        """
        BNO055 en modo NDOF (fusión interna):
          heading : 0-360° en sentido HORARIO (convención brújula)
          roll    : grados
          pitch   : grados
          ax,ay,az: aceleración lineal SIN gravedad [m/s²] (VECTOR_LINEARACCEL)
          gx,gy,gz: velocidad angular [rad/s] (VECTOR_GYROSCOPE)

        Conversión heading → yaw ROS2:
          ROS2 usa CCW positivo (regla mano derecha)
          BNO055 usa CW positivo (convención brújula)
          → yaw_ros = -heading_rad

        Covarianzas (diagonal 3×3 aplanada a 9 elementos):
          orientation_covariance    : (π/180)² ≈ 3e-4 rad²
            Fuente: BNO055 datasheet BST-BNO055-DS000 §3.6, precisión ±1° RMS
          angular_velocity_covariance: 1e-4 (rad/s)²
            Fuente: ruido giroscopio ~0.01°/s/√Hz, valor conservador en chassis
          linear_acceleration_covariance: 2e-3 (m/s²)²
            Fuente: ruido acelerómetro ~0.3mg, amplificado por vibración del motor
        """
        msg                 = Imu()
        msg.header.stamp    = self._stamp()
        msg.header.frame_id = 'imu_link'

        # Euler → Quaternion (ZYX: yaw, pitch, roll)
        yaw = -math.radians(heading)   # CW brújula → CCW ROS2
        r   =  math.radians(roll)
        p   =  math.radians(pitch)

        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(p   * 0.5), math.sin(p   * 0.5)
        cr, sr = math.cos(r   * 0.5), math.sin(r   * 0.5)

        msg.orientation.w = cr * cp * cy + sr * sp * sy
        msg.orientation.x = sr * cp * cy - cr * sp * sy
        msg.orientation.y = cr * sp * cy + sr * cp * sy
        msg.orientation.z = cr * cp * sy - sr * sp * cy

        # Velocidad angular — directamente del BNO055 [rad/s]
        msg.angular_velocity.x = gx
        msg.angular_velocity.y = gy
        msg.angular_velocity.z = gz

        # Aceleración lineal sin gravedad — directamente del BNO055 [m/s²]
        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az

        # Covarianzas diagonales 3×3 (requeridas por EKF/robot_localization)
        # Formato: [xx, xy, xz, yx, yy, yz, zx, zy, zz] (fila por fila)
        diag = lambda v: [v, 0.0, 0.0,  0.0, v, 0.0,  0.0, 0.0, v]
        msg.orientation_covariance          = diag(3e-4)  # rad²
        msg.angular_velocity_covariance     = diag(1e-4)  # (rad/s)²
        msg.linear_acceleration_covariance  = diag(2e-3)  # (m/s²)²

        self.pub_imu.publish(msg)

    # ── Batería ─────────────────────────────────────────────────────

    def _bat(self, voltage: float, current: float, power: float,
             soc: float, warn: int, crit: int):
        """
        INA226 mide en el lado INPUT del regulador (voltaje RAW LiPo 3S).
        soc : State of Charge calculado en firmware con Coulomb counter [0.0-1.0]
        warn: flag software — voltaje < 11.10V (3.70V/celda)
        crit: flag software — voltaje < 10.50V (3.50V/celda)
        """
        msg                         = BatteryState()
        msg.header.stamp            = self._stamp()
        msg.voltage                 = voltage    # V
        msg.current                 = current    # A (positivo = descarga)
        msg.design_capacity         = 3.3        # Ah (LiPo 3S 3300mAh)
        msg.percentage              = float(max(0.0, min(1.0, soc)))
        msg.power_supply_status     = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
        msg.present                 = True
        msg.power_supply_health     = (
            BatteryState.POWER_SUPPLY_HEALTH_DEAD
            if crit else BatteryState.POWER_SUPPLY_HEALTH_GOOD)
        self.pub_bat.publish(msg)

        # Máquina de estados batería
        if warn and self._bat_state == 'NORMAL':
            self._bat_state = 'LOW'
            self._pub_sys('LOW_BAT')
            self.get_logger().warn(
                f'Batería baja: {voltage:.2f}V  SoC={soc*100:.0f}%')

        if crit and self._bat_state not in ('CRITICAL', 'SHUTDOWN'):
            self._bat_state = 'CRITICAL'
            self._pub_sys('CRITICAL_BAT')
            self.get_logger().error(f'Batería crítica: {voltage:.2f}V')
            self._initiate_shutdown()

    # ── STA — eventos del firmware ───────────────────────────────────

    def _sta(self, status: str):
        m = String(); m.data = status
        self.pub_status.publish(m)

        if 'ERR' in status:
            self.get_logger().error(f'[ESP32-S] {status}')
        elif 'WARN' in status or 'ESTOP' in status:
            self.get_logger().warn(f'[ESP32-S] {status}')
        else:
            self.get_logger().info(f'[ESP32-S] {status}')

        if status == 'CRITICAL_BAT' and self._bat_state not in ('CRITICAL', 'SHUTDOWN'):
            self._bat_state = 'CRITICAL'
            self._initiate_shutdown()

    # ── Apagado seguro ───────────────────────────────────────────────

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