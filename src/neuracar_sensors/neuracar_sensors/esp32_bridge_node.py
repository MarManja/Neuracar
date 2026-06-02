#!/usr/bin/env python3
"""
esp32_bridge_dual.py — Neuracar  v2.0
=======================================
Bridge para arquitectura dual ESP32:
  • ESP32-S (Sensores): encoder + IMU + batería + OLED
      puerto: /dev/esp32s   (udev symlink)
      publica:  E, I, B, STA
  • ESP32-A (Actuadores): ESC + servo
      puerto: /dev/esp32a   (udev symlink)
      recibe:  C, SHUTDOWN, CLRESTOP
      publica: ACT (eco del comando), STA

Suscribe:
  /neuracar/user_command   (geometry_msgs/Vector3Stamped)
    vector.x = throttle ∈ [-1, 1]
    vector.y = steering  ∈ [-1, 1]

Publica:
  /neuracar/imu            (sensor_msgs/Imu)
  /neuracar/encoder        (sensor_msgs/JointState)
  /neuracar/battery        (sensor_msgs/BatteryState)
  /neuracar/status         (std_msgs/String)   — eventos firmware
  /neuracar/system_status  (std_msgs/String)   — estado global
  /neuracar/actuator_echo  (std_msgs/String)   — eco ESP32-A

Puertos udev — agrega a /etc/udev/rules.d/99-neuracar.rules:
  # ESP32-S (Sensores) — CP2102
  SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60",
    ATTRS{serial}=="SENSOR_SERIAL_NUMBER", SYMLINK+="esp32s"

  # ESP32-A (Actuadores) — CP2102
  SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60",
    ATTRS{serial}=="ACTUADOR_SERIAL_NUMBER", SYMLINK+="esp32a"

  Obtener serial: udevadm info -a -n /dev/ttyUSB0 | grep serial
"""

import math
import subprocess
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import BatteryState, Imu, JointState
from std_msgs.msg import String

try:
    import serial
except ImportError:
    raise SystemExit('pyserial no instalado.\n'
                     'pip install pyserial --break-system-packages')

# ─── Constantes batería ───────────────────────────────────────────
BAT_LOW_V      = 11.10
BAT_CRITICAL_V = 10.50
SHUTDOWN_GRACE_S = 5.0


class ESP32DualBridgeNode(Node):

    def __init__(self):
        super().__init__('esp32_dual_bridge')

        # ── Parámetros ──────────────────────────────────────────────
        self.declare_parameter('port_sensores',  '/dev/esp32s')
        self.declare_parameter('port_actuadores','/dev/esp32a')
        self.declare_parameter('baudrate',        921600)
        self.declare_parameter('wheel_radius',    0.033)
        self.declare_parameter('gear_ratio',      9.2459)
        self.declare_parameter('watchdog_s',      0.5)
        self.declare_parameter('auto_shutdown',   True)

        port_s  = self.get_parameter('port_sensores').value
        port_a  = self.get_parameter('port_actuadores').value
        baud    = self.get_parameter('baudrate').value
        self._R      = self.get_parameter('wheel_radius').value
        self._ratio  = self.get_parameter('gear_ratio').value
        self._wd_s   = self.get_parameter('watchdog_s').value
        self._auto_sd= self.get_parameter('auto_shutdown').value

        # ── Estado ──────────────────────────────────────────────────
        self._bat_state     = 'NORMAL'
        self._shutdown_sent = False
        self._shutdown_timer= None

        # ── Conexión serial — Sensores ───────────────────────────────
        try:
            self._ser_s = serial.Serial(port_s, baud, timeout=0.1)
            self.get_logger().info(f'ESP32-S (sensores): {port_s} @ {baud}')
        except serial.SerialException as exc:
            self.get_logger().fatal(f'No se pudo abrir ESP32-S {port_s}: {exc}')
            raise

        # ── Conexión serial — Actuadores ─────────────────────────────
        try:
            self._ser_a = serial.Serial(port_a, baud, timeout=0.1)
            self.get_logger().info(f'ESP32-A (actuadores): {port_a} @ {baud}')
        except serial.SerialException as exc:
            self.get_logger().fatal(f'No se pudo abrir ESP32-A {port_a}: {exc}')
            raise

        # ── Publishers ──────────────────────────────────────────────
        self.pub_imu    = self.create_publisher(Imu,          '/neuracar/imu',           10)
        self.pub_enc    = self.create_publisher(JointState,   '/neuracar/encoder',        10)
        self.pub_bat    = self.create_publisher(BatteryState, '/neuracar/battery',        10)
        self.pub_status = self.create_publisher(String,       '/neuracar/status',         10)
        self.pub_sys    = self.create_publisher(String,       '/neuracar/system_status',  10)
        self.pub_act    = self.create_publisher(String,       '/neuracar/actuator_echo',  10)

        # ── Subscriber ──────────────────────────────────────────────
        self.sub_cmd = self.create_subscription(
            Vector3Stamped, '/neuracar/user_command', self._cmd_callback, 10)

        # ── Watchdog ────────────────────────────────────────────────
        self._last_cmd_time = self.get_clock().now()
        self.create_timer(0.1, self._watchdog_cb)

        # ── Hilos lectores seriales ──────────────────────────────────
        self._alive = True

        self._reader_s = threading.Thread(
            target=self._serial_reader,
            args=(self._ser_s, 'S'),
            daemon=True)
        self._reader_s.start()

        self._reader_a = threading.Thread(
            target=self._serial_reader,
            args=(self._ser_a, 'A'),
            daemon=True)
        self._reader_a.start()

        self.get_logger().info('ESP32 Dual Bridge v2.0 listo.')
        self._publish_sys('READY')

    # ── Helpers ─────────────────────────────────────────────────────

    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _send_a(self, data: bytes):
        """Enviar comando al ESP32-A (actuadores)."""
        try:
            self._ser_a.write(data)
        except serial.SerialException as exc:
            self.get_logger().error(f'Error escritura ESP32-A: {exc}')

    def _send_s(self, data: bytes):
        """Enviar al ESP32-S (solo SHUTDOWN)."""
        try:
            self._ser_s.write(data)
        except serial.SerialException as exc:
            self.get_logger().error(f'Error escritura ESP32-S: {exc}')

    def _publish_sys(self, status: str):
        msg = String(); msg.data = status
        self.pub_sys.publish(msg)

    # ── Subscriber callback ─────────────────────────────────────────

    def _cmd_callback(self, msg: Vector3Stamped):
        self._last_cmd_time = self.get_clock().now()
        if self._bat_state in ('CRITICAL', 'SHUTDOWN'):
            self._send_a(b'C,0.000,0.000\n')
            return
        t = max(-1.0, min(1.0, msg.vector.x))
        s = max(-1.0, min(1.0, msg.vector.y))
        self._send_a(f'C,{t:.3f},{s:.3f}\n'.encode())

    # ── Watchdog ────────────────────────────────────────────────────

    def _watchdog_cb(self):
        elapsed = (self.get_clock().now() - self._last_cmd_time).nanoseconds * 1e-9
        if elapsed > self._wd_s:
            self._send_a(b'C,0.000,0.000\n')

    # ── Lector serial genérico ──────────────────────────────────────

    def _serial_reader(self, ser: serial.Serial, tag: str):
        """
        tag='S' → mensajes de sensores (E, I, B, STA)
        tag='A' → mensajes de actuadores (ACT, STA)
        """
        while self._alive:
            try:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode('utf-8', errors='ignore').strip()
                if line:
                    self._parse_line(line, tag)
            except serial.SerialException as exc:
                if self._alive:
                    self.get_logger().error(f'Error lectura ESP32-{tag}: {exc}')
                break
            except Exception as exc:
                self.get_logger().warn(f'Parse error ESP32-{tag}: {exc}  raw={raw!r}')

    # ── Parser ──────────────────────────────────────────────────────

    def _parse_line(self, line: str, tag: str):
        parts    = line.split(',')
        msg_type = parts[0]
        try:
            # ── Mensajes del ESP32-S ─────────────────────────────────
            if tag == 'S':
                if msg_type == 'E' and len(parts) == 4:
                    self._handle_encoder(
                        float(parts[1]), float(parts[2]), float(parts[3]))

                elif msg_type == 'I' and len(parts) == 10:
                    self._handle_imu(*[float(p) for p in parts[1:]])

                elif msg_type == 'B' and len(parts) == 7:
                    self._handle_battery(
                        float(parts[1]),          # V
                        float(parts[2]),          # A
                        float(parts[3]),          # W
                        float(parts[4]) / 100.0,  # SoC 0-100 → 0-1
                        int(parts[5]),            # warn
                        int(parts[6]),            # crit
                    )

                elif msg_type == 'STA':
                    self._handle_status(','.join(parts[1:]), 'S')

            # ── Mensajes del ESP32-A ─────────────────────────────────
            elif tag == 'A':
                if msg_type == 'ACT' and len(parts) == 3:
                    # Eco del comando activo — publicar para diagnóstico
                    m = String()
                    m.data = f'ACT,{parts[1]},{parts[2]}'
                    self.pub_act.publish(m)

                elif msg_type == 'STA':
                    self._handle_status(','.join(parts[1:]), 'A')

        except (ValueError, IndexError) as exc:
            self.get_logger().warn(
                f'Mensaje malformado ESP32-{tag} "{line}": {exc}')

    # ── Handler: Encoder ────────────────────────────────────────────

    def _handle_encoder(self, angle_deg: float, motor_rpm: float,
                         v_linear: float):
        msg              = JointState()
        msg.header.stamp = self._stamp()
        msg.name         = ['drive_motor']

        wheel_rpm   = motor_rpm / self._ratio
        wheel_rad_s = wheel_rpm * (2.0 * math.pi / 60.0)

        msg.position = [math.radians(angle_deg)]
        msg.velocity = [wheel_rad_s]
        msg.effort   = [v_linear]   # m/s — calculado en firmware

        self.pub_enc.publish(msg)

    # ── Handler: IMU ────────────────────────────────────────────────

    def _handle_imu(self, heading, roll, pitch, ax, ay, az, gx, gy, gz):
        """
        BNO055: X=derecha, Y=adelante, Z=arriba
        heading: 0-360° CW (brújula) → negamos para ROS2 CCW
        Validar: girar izquierda → orientation.z positivo
        """
        msg                 = Imu()
        msg.header.stamp    = self._stamp()
        msg.header.frame_id = 'imu_link'

        yaw = -math.radians(heading)   # CW→CCW
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

    # ── Handler: Batería ────────────────────────────────────────────

    def _handle_battery(self, voltage, current, power, soc, warn, crit):
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
            self._publish_sys('LOW_BAT')
            self.get_logger().warn(f'Batería baja: {voltage:.2f}V SoC={soc*100:.0f}%')

        if crit and self._bat_state != 'SHUTDOWN':
            self._bat_state = 'CRITICAL'
            self._publish_sys('CRITICAL_BAT')
            self.get_logger().error(f'Batería crítica: {voltage:.2f}V — apagando')
            self._initiate_shutdown()

    # ── Handler: STA ────────────────────────────────────────────────

    def _handle_status(self, status: str, tag: str):
        m = String(); m.data = f'ESP32{tag},{status}'
        self.pub_status.publish(m)

        label = f'[ESP32-{tag}]'
        if 'ERR' in status:
            self.get_logger().error(f'{label} {status}')
        elif 'ESTOP' in status or 'WARN' in status:
            self.get_logger().warn(f'{label} {status}')
        else:
            self.get_logger().info(f'{label} {status}')

        if status == 'CRITICAL_BAT' and self._bat_state != 'SHUTDOWN':
            self._bat_state = 'CRITICAL'
            self._initiate_shutdown()

    # ── Apagado seguro ───────────────────────────────────────────────

    def _initiate_shutdown(self):
        if self._shutdown_sent:
            return
        self._shutdown_sent = True
        # Parar actuadores y notificar ambas ESP32
        self._send_a(b'C,0.000,0.000\n')
        self._send_a(b'SHUTDOWN\n')
        self._send_s(b'SHUTDOWN\n')
        self.get_logger().warn(f'Apagado del SO en {SHUTDOWN_GRACE_S:.0f}s...')
        if self._auto_sd:
            self._shutdown_timer = threading.Timer(
                SHUTDOWN_GRACE_S, self._execute_system_shutdown)
            self._shutdown_timer.daemon = True
            self._shutdown_timer.start()

    def _execute_system_shutdown(self):
        self.get_logger().error('Ejecutando shutdown del SO...')
        self._bat_state = 'SHUTDOWN'
        self._publish_sys('SHUTTING_DOWN')
        try:
            subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=True)
        except subprocess.CalledProcessError as exc:
            self.get_logger().error(f'Shutdown fallido: {exc}')

    # ── Cleanup ─────────────────────────────────────────────────────

    def destroy_node(self):
        self._alive = False
        if self._shutdown_timer and self._shutdown_timer.is_alive():
            self._shutdown_timer.cancel()
        for ser, name in [(self._ser_a, 'A'), (self._ser_s, 'S')]:
            try:
                if name == 'A':
                    ser.write(b'C,0.000,0.000\n')
                ser.close()
            except Exception:
                pass
        super().destroy_node()


# ──────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = ESP32DualBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()