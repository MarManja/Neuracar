#!/usr/bin/env python3
"""
ESP32 Bridge Node — Neuracar  v1.5
=====================================
Actualizado para ESP32 firmware v1.0 (compatible Teensy v6.x):
  - Trama B ampliada: V, I, P, SoC, warn, crit (INA226)
  - Mensaje X nuevo: alerta hardware undervoltaje INA226
  - Umbrales batería alineados con firmware (LIPO_WARN_V / LIPO_CRIT_V)
  - gear_ratio default corregido a 9.2459 (= 189356/5/4096 del firmware)
  - SoC viene del Teensy (Coulomb counter + tabla LiPo fusionados)
  - Handlers para STAs nuevos: ENC_SYNC, INA226_OK, ERR_INA226

Suscribe:
  /neuracar/user_command   (geometry_msgs/Vector3Stamped)

Publica:
  /neuracar/imu            (sensor_msgs/Imu)
  /neuracar/encoder        (sensor_msgs/JointState)
  /neuracar/battery        (sensor_msgs/BatteryState)
  /neuracar/status         (std_msgs/String)
  /neuracar/system_status  (std_msgs/String)

Protocolo serial @ 921600 baud — mensajes del firmware v4.0:
  Recibe:
    E,{angleDeg:.2f},{rpm:.2f}\\n                       @ 50 Hz
    I,{hdg},{roll},{pitch},{ax},{ay},{az},{gx},{gy},{gz}\\n @ 50 Hz
    B,{V:.3f},{I:.3f},{P:.3f},{soc:.3f},{warn},{crit}\\n  @ 10 Hz
    X,{RAZON},{V:.3f}\\n                                 alerta HW INA226
    STA,{codigo}\\n                                      eventos
  Envía:
    C,{throttle:.3f},{steering:.3f}\\n
    SHUTDOWN\\n
    CLRESTOP\\n
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
                     'Ejecuta: pip install pyserial --break-system-packages')


# ─── Umbrales batería — deben coincidir con el firmware v4.0 ──────────
# Fuente: LIPO_WARN_V y LIPO_CRIT_V del firmware (voltaje RAW LiPo 3S)
BAT_LOW_V      = 11.10  # V  (3.70 V/celda — advertencia software)
BAT_CRITICAL_V = 10.50  # V  (3.50 V/celda — alerta hardware INA226)
BAT_FULL_V     = 12.60  # V  (4.20 V/celda — 100 %)
BAT_EMPTY_V    = 10.50  # V  (0 % — límite crítico)

SHUTDOWN_GRACE_S = 5.0


class ESP32BridgeNode(Node):

    def __init__(self):
        super().__init__('esp32_bridge_node')

        # ── Parámetros ──────────────────────────────────────────────
        self.declare_parameter('port',          '/dev/esp32')
        self.declare_parameter('baudrate',      921600)
        self.declare_parameter('wheel_radius',  0.033)  # ESP32 fw usa 0.0508m (2")
        # gear_ratio: 189356/5/4096 = 9.2459 — calculado en el firmware v4.0
        # Actualizar si cambias la relación mecánica del chassis
        self.declare_parameter('gear_ratio',    9.2459)
        # watchdog_s: tiempo sin comando antes de parar el motor
        # Protege si el nodo de control muere. USB local → latencia < 1ms, no interfiere.
        self.declare_parameter('watchdog_s',    0.5)
        # auto_shutdown: False en desarrollo para no apagar la Jetson con batería baja
        # ros2 launch neuracar_bringup sensors.launch.py auto_shutdown:=false
        self.declare_parameter('auto_shutdown', True)

        self._port    = self.get_parameter('port').value
        self._baud    = self.get_parameter('baudrate').value
        self._R       = self.get_parameter('wheel_radius').value
        self._ratio   = self.get_parameter('gear_ratio').value
        self._wd_s    = self.get_parameter('watchdog_s').value
        self._auto_sd = self.get_parameter('auto_shutdown').value

        # ── Estado ──────────────────────────────────────────────────
        self._bat_state      = 'NORMAL'   # NORMAL | LOW | CRITICAL | SHUTDOWN
        self._shutdown_sent  = False
        # Timer que ejecuta "sudo shutdown -h now" en Linux de la Jetson
        self._shutdown_timer = None

        # ── Serial ──────────────────────────────────────────────────
        try:
            self._ser = serial.Serial(self._port, self._baud, timeout=0.1)
            self.get_logger().info(
                f'ESP32 conectada: {self._port} @ {self._baud} baud')
        except serial.SerialException as exc:
            self.get_logger().fatal(f'No se pudo abrir {self._port}: {exc}')
            raise

        # ── Publishers ──────────────────────────────────────────────
        self.pub_imu    = self.create_publisher(Imu,          '/neuracar/imu',          10)
        self.pub_enc    = self.create_publisher(JointState,   '/neuracar/encoder',       10)
        self.pub_bat    = self.create_publisher(BatteryState, '/neuracar/battery',       10)
        self.pub_status = self.create_publisher(String,       '/neuracar/status',        10)
        self.pub_sys    = self.create_publisher(String,       '/neuracar/system_status', 10)

        # ── Subscriber ──────────────────────────────────────────────
        self.sub_cmd = self.create_subscription(
            Vector3Stamped, '/neuracar/user_command', self._cmd_callback, 10)

        # ── Watchdog ────────────────────────────────────────────────
        self._last_cmd_time = self.get_clock().now()
        self.create_timer(0.1, self._watchdog_cb)

        # ── Hilo lector serial ──────────────────────────────────────
        self._alive  = True
        self._reader = threading.Thread(target=self._serial_reader, daemon=True)
        self._reader.start()

        self.get_logger().info('ESP32 bridge v1.4 iniciado (fw v4.0).')
        self._publish_sys('READY')

    # ── Helpers ─────────────────────────────────────────────────────
    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _send(self, data: bytes):
        try:
            self._ser.write(data)
        except serial.SerialException as exc:
            self.get_logger().error(f'Error escritura serial: {exc}')

    def _publish_sys(self, status: str):
        msg = String()
        msg.data = status
        self.pub_sys.publish(msg)

    # ── Subscriber callback ─────────────────────────────────────────
    def _cmd_callback(self, msg: Vector3Stamped):
        self._last_cmd_time = self.get_clock().now()
        if self._bat_state in ('CRITICAL', 'SHUTDOWN'):
            self._send(b'C,0.000,0.000\n')
            return
        t = max(-1.0, min(1.0, msg.vector.x))
        s = max(-1.0, min(1.0, msg.vector.y))
        # "C" = Command — prefijo que la Teensy reconoce en parseLine()
        self._send(f'C,{t:.3f},{s:.3f}\n'.encode())

    # ── Watchdog ────────────────────────────────────────────────────
    def _watchdog_cb(self):
        elapsed = (self.get_clock().now() - self._last_cmd_time).nanoseconds * 1e-9
        if elapsed > self._wd_s:
            self._send(b'C,0.000,0.000\n')

    # ── Lector serial ───────────────────────────────────────────────
    def _serial_reader(self):
        while self._alive:
            try:
                raw  = self._ser.readline()
                if not raw:
                    continue
                line = raw.decode('utf-8', errors='ignore').strip()
                if line:
                    self._parse_line(line)
            except serial.SerialException as exc:
                if self._alive:
                    self.get_logger().error(f'Error lectura serial: {exc}')
                break
            except Exception as exc:
                self.get_logger().warn(f'Parse error: {exc}  raw={raw!r}')

    # ── Parser ──────────────────────────────────────────────────────
    def _parse_line(self, line: str):
        parts    = line.split(',')
        msg_type = parts[0]
        try:
            if msg_type == 'E' and len(parts) in (3, 4):
                # ESP32 fw v1.0 manda 4 campos: angle, rpm, v_linear
                # Teensy fw v6.x mandaba 3 campos: angle, rpm
                v_lin = float(parts[3]) if len(parts) == 4 else None
                self._handle_encoder(float(parts[1]), float(parts[2]), v_lin)

            elif msg_type == 'I' and len(parts) == 10:
                self._handle_imu(*[float(p) for p in parts[1:]])

            elif msg_type == 'B' and len(parts) == 7:
                # B,{V:.3f},{I:.3f},{P:.3f},{soc:.3f},{warn},{crit}
                self._handle_battery(
                    float(parts[1]),   # voltage  [V]
                    float(parts[2]),   # current  [A]
                    float(parts[3]),   # power    [W]
                    float(parts[4]) / 100.0,  # soc: ESP32 manda 0-100, bridge usa 0-1
                    int(parts[5]),     # warn     0|1
                    int(parts[6]),     # crit     0|1
                )

            elif msg_type == 'X' and len(parts) == 3:
                # X,{RAZON},{V:.3f} — alerta hardware INA226 (ISR pin ALERT)
                self._handle_hw_alert(parts[1], float(parts[2]))

            elif msg_type == 'STA':
                self._handle_status(','.join(parts[1:]))

        except (ValueError, IndexError) as exc:
            self.get_logger().warn(f'Mensaje malformado "{line}": {exc}')

    # ── Handler: Encoder → JointState ───────────────────────────────
    def _handle_encoder(self, angle_deg: float, motor_rpm: float,
                         v_linear: float = None):
        """
        División de responsabilidades:
          Microcontrolador → calcula RPM con hardware counter (precisión µs)
          ROS2             → convierte a velocidad rueda y lineal (cinemática)

        ESP32 fw v1.0 envía v_linear calculado directamente en el firmware
        (campo extra en trama E). Se usa eso si está disponible para evitar
        discrepancias con el WHEEL_RADIUS_M del firmware (0.0508 m).
        Si no viene (Teensy legacy), se recalcula aquí con el parámetro local.

        gear_ratio = 9.2459 = 189356/5/4096
        wheel_radius: ajustar si cambia el chassis
        """
        msg              = JointState()
        msg.header.stamp = self._stamp()
        msg.name         = ['drive_motor']

        wheel_rpm   = motor_rpm / self._ratio
        wheel_rad_s = wheel_rpm * (2.0 * math.pi / 60.0)

        # Usar v_linear del firmware si viene (ESP32); si no, calcular local
        linear_m_s = v_linear if v_linear is not None else wheel_rad_s * self._R

        msg.position = [math.radians(angle_deg)]  # rad (ESP32: wrapeado 0-2π)
        msg.velocity = [wheel_rad_s]               # rad/s eje rueda
        msg.effort   = [linear_m_s]               # m/s velocidad lineal

        self.pub_enc.publish(msg)

    # ── Handler: IMU → sensor_msgs/Imu ──────────────────────────────
    def _handle_imu(self, heading, roll, pitch, ax, ay, az, gx, gy, gz):
        """
        Montaje BNO055 (Wire2, SDA=25 SCL=24):
          Posición: adelante del vehículo
          X = derecha  |  Y = adelante  |  Z = arriba
          (regla mano derecha: X×Y=Z → derecha×adelante=arriba ✓)

        frame_id = 'imu_link'
        TF estático necesario: imu_link → base_link (-90° en Z)
          ros2 run tf2_ros static_transform_publisher 0 0 0 -1.5708 0 0 base_link imu_link

        Covarianzas BNO055 NDOF (fusión interna):
          Orientación:   ±1°  → (π/180)²  ≈ 3e-4 rad²
          Giroscopio:  ~0.01°/s/√Hz → ≈ 1e-4 (rad/s)²
          Aceleración: ~0.3mg/√Hz   → ≈ 2e-3 (m/s²)²
        """
        msg                 = Imu()
        msg.header.stamp    = self._stamp()
        msg.header.frame_id = 'imu_link'

        # BNO055 VECTOR_EULER heading: 0-360° en sentido HORARIO (CW).
        # ROS convention: yaw positivo = sentido ANTIHORARIO (CCW).
        # Negamos para convertir CW→CCW.
        # Validar: girar el carro a la izquierda debe dar orientation.z positivo.
        yaw = -math.radians(heading)
        r   = math.radians(roll)
        p   = math.radians(pitch)

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

        c3 = [3e-4, 0.0, 0.0,  0.0, 3e-4, 0.0,  0.0, 0.0, 3e-4]
        c1 = [1e-4, 0.0, 0.0,  0.0, 1e-4, 0.0,  0.0, 0.0, 1e-4]
        c2 = [2e-3, 0.0, 0.0,  0.0, 2e-3, 0.0,  0.0, 0.0, 2e-3]
        msg.orientation_covariance         = c3
        msg.angular_velocity_covariance    = c1
        msg.linear_acceleration_covariance = c2

        self.pub_imu.publish(msg)

    # ── Handler: Batería → BatteryState ─────────────────────────────
    def _handle_battery(self, voltage: float, current: float, power: float,
                         soc: float, warn: int, crit: int):
        """
        INA226 mide en el lado INPUT del XL4015 (voltaje RAW LiPo 3S).
        soc: State of Charge calculado en Teensy (Coulomb counter + tabla LiPo).
        warn/crit: flags directos del firmware — usarlos evita recalcular umbrales.
        """
        msg                         = BatteryState()
        msg.header.stamp            = self._stamp()
        msg.voltage                 = voltage    # V — LiPo raw (antes del XL4015)
        msg.current                 = current    # A — positivo = descarga
        msg.design_capacity         = 3.3        # Ah (3300 mAh)
        msg.percentage              = float(max(0.0, min(1.0, soc)))  # del Teensy
        msg.power_supply_status     = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
        msg.present                 = True
        msg.power_supply_health = (
            BatteryState.POWER_SUPPLY_HEALTH_GOOD
            if not crit else BatteryState.POWER_SUPPLY_HEALTH_DEAD
        )

        self.pub_bat.publish(msg)

        # Actualizar estado usando los flags del firmware (evita duplicar umbrales)
        if warn and self._bat_state == 'NORMAL':
            self._bat_state = 'LOW'
            self._publish_sys('LOW_BAT')
            self.get_logger().warn(
                f'Batería baja: {voltage:.2f}V  SoC={soc*100:.0f}%  '
                f'P={power:.1f}W')

    # ── Handler: Alerta hardware INA226 (mensaje X) ─────────────────
    def _handle_hw_alert(self, reason: str, voltage: float):
        """
        El INA226 dispara su pin ALERT cuando V < LIPO_CRIT_V (10.50V).
        La ISR de la Teensy captura esto y envía "X,{RAZON},{V}" de inmediato.
        Es más rápido que esperar el ciclo de batería a 10Hz.
        """
        msg      = String()
        msg.data = f'HW_ALERT,{reason},{voltage:.3f}'
        self.pub_status.publish(msg)
        self._publish_sys('CRITICAL_BAT')

        self.get_logger().error(
            f'¡Alerta HW INA226! Razón={reason}  V={voltage:.2f}V  '
            f'→ apagado inmediato')

        if self._bat_state != 'SHUTDOWN':
            self._bat_state = 'CRITICAL'
            self._initiate_shutdown()

    # ── Handler: eventos STA ────────────────────────────────────────
    def _handle_status(self, status: str):
        msg      = String()
        msg.data = status
        self.pub_status.publish(msg)

        # Logs con nivel adecuado
        if status.startswith('ERR'):
            self.get_logger().error(f'[Teensy] {status}')
        elif status.startswith('ESTOP'):
            self.get_logger().warn(f'[Teensy] {status}')
        else:
            self.get_logger().info(f'[Teensy] {status}')

        # ── Máquina de estados batería ──────────────────────────────
        if status == 'LOW_BAT' and self._bat_state == 'NORMAL':
            self._bat_state = 'LOW'
            self._publish_sys('LOW_BAT')
            self.get_logger().warn('Batería baja — guarda tu trabajo.')

        elif status == 'CRITICAL_BAT' and self._bat_state != 'SHUTDOWN':
            self._bat_state = 'CRITICAL'
            self._publish_sys('CRITICAL_BAT')
            self.get_logger().error('Batería crítica — iniciando apagado seguro.')
            self._initiate_shutdown()

        # fw v6.1: emergencyStop("CRITICAL_BAT") genera "STA,ESTOP,CRITICAL_BAT"
        # en lugar de "STA,CRITICAL_BAT" directo — capturamos ambas formas.
        elif status == 'ESTOP,CRITICAL_BAT' and self._bat_state != 'SHUTDOWN':
            self._bat_state = 'CRITICAL'
            self._publish_sys('CRITICAL_BAT')
            self.get_logger().error('Batería crítica (vía ESTOP) — apagado seguro.')
            self._initiate_shutdown()

        elif status == 'OVERCURRENT':
            self._publish_sys('OVERCURRENT')
            self.get_logger().error('Sobrecorriente en batería lógica.')

        # STAs informativos — fw v6.1
        elif status.startswith('ENC_SYNC'):
            self.get_logger().info(f'Encoder sincronizado: {status}')
        elif status == 'ENC_ZEROED':
            self.get_logger().info('Encoder reseteado a cero')
        elif status.startswith('INA226_OK'):
            self.get_logger().info(f'INA226 listo: {status}')
        elif status == 'ERR_INA226':
            self.get_logger().error('INA226 no encontrado — sin monitoreo de batería')
            self._publish_sys('ERR_INA226')
        elif status == 'BAT_NORMAL' and self._bat_state == 'LOW':
            self._bat_state = 'NORMAL'
            self._publish_sys('BAT_NORMAL')
            self.get_logger().info('Batería recuperada — estado normal.')
        elif status == 'PONG':
            self.get_logger().debug('Teensy responde PING→PONG ok')
        elif status.startswith('READY'):
            self.get_logger().info(f'Teensy lista: {status}')
        elif status.startswith('WARN'):
            self.get_logger().warn(f'[Teensy] {status}')
        elif status.startswith('UNKNOWN_CMD'):
            self.get_logger().warn(f'Comando no reconocido por Teensy: {status}')

    # ── Protocolo de apagado seguro ─────────────────────────────────
    def _initiate_shutdown(self):
        """
        Secuencia:
          1. C,0,0       → Teensy para el motor
          2. SHUTDOWN    → Teensy actualiza OLED y entra en silencio
          3. Espera 5s   → ROS2 termina sus nodos limpiamente
          4. sudo shutdown -h now → Linux (SO) apaga la Jetson

        Requiere permiso sudo sin contraseña:
          echo "$USER ALL=(ALL) NOPASSWD: /sbin/shutdown" |
          sudo tee /etc/sudoers.d/neuracar-shutdown
        """
        if self._shutdown_sent:
            return
        self._shutdown_sent = True

        self._send(b'C,0.000,0.000\n')
        self._send(b'SHUTDOWN\n')

        self.get_logger().warn(
            f'Apagado del SO en {SHUTDOWN_GRACE_S:.0f}s...')

        if self._auto_sd:
            self._shutdown_timer = threading.Timer(
                SHUTDOWN_GRACE_S, self._execute_system_shutdown)
            self._shutdown_timer.daemon = True
            self._shutdown_timer.start()

    def _execute_system_shutdown(self):
        self.get_logger().error('Ejecutando shutdown del SO (Linux Jetson)...')
        self._bat_state = 'SHUTDOWN'
        self._publish_sys('SHUTTING_DOWN')
        try:
            subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=True)
        except subprocess.CalledProcessError as exc:
            self.get_logger().error(
                f'Shutdown fallido: {exc}\n'
                'Verifica /etc/sudoers.d/neuracar-shutdown')

    # ── Cleanup ─────────────────────────────────────────────────────
    def destroy_node(self):
        self._alive = False
        if self._shutdown_timer and self._shutdown_timer.is_alive():
            self._shutdown_timer.cancel()
        try:
            self._ser.write(b'C,0.000,0.000\n')
            self._ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ESP32BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()