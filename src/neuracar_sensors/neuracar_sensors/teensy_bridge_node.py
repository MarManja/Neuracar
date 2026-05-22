#!/usr/bin/env python3
"""
Teensy Bridge Node — Neuracar  v1.3
=====================================
Puente bidireccional Teensy 4.1 ROS2.

Cambios v1.3:
  - Umbrales de batería corregidos para XL4015 a 10.9V
  - Orientación del IMU documentada (X=derecha, Y=adelante, Z=arriba)

Suscribe:
  /neuracar/user_command   (geometry_msgs/Vector3Stamped)
      vector.x = throttle [-1, 1]   positivo = avance
      vector.y = steering [-1, 1]   positivo = izquierda

Publica:
  /neuracar/imu            (sensor_msgs/Imu)
  /neuracar/encoder        (sensor_msgs/JointState)
  /neuracar/battery        (sensor_msgs/BatteryState)
  /neuracar/status         (std_msgs/String)   eventos de la Teensy
  /neuracar/system_status  (std_msgs/String)   estado global

Protocolo serial ASCII @ 921600 baud:
  Envía a Teensy:  "C,{throttle:.3f},{steering:.3f}\\n"
                    └─ "C" = Command (prefijo que la Teensy reconoce en parseLine)
                   "SHUTDOWN\\n"   confirmar apagado
  Recibe de Teensy: "E,{angle_deg},{motor_rpm}\\n"
                    "I,{hdg},{roll},{pitch},{ax},{ay},{az},{gx},{gy},{gz}\\n"
                    "B,{voltage},{current}\\n"
                    "STA,{codigo}\\n"
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
    raise SystemExit(
        'pyserial no instalado.\n'
        'Ejecuta: pip install pyserial --break-system-packages'
    )


# ─── Umbrales de batería lógica ───────────────────────────────────────
# Circuito: Batería 3S (10.0-12.6V) → ACS712/FZ0430 (miden aquí) → XL4015 → 10.9V → Jetson
#
# El XL4015 es un buck (step-down). Necesita Vin > Vout + dropout (~1.1V a plena carga).
# Para salida 10.9V: necesita batería > 12.0V para regular correctamente.
#
# ¿Por qué no daña la Jetson cuando el buck pierde regulación?
#   Al perder regulación la salida sigue a la batería menos el dropout (≈ batería - 0.5V).
#   La Jetson acepta 9-20V, así que tolera bajadas hasta ~9V sin daño.
#   El riesgo real es batería < 10.9V (buck físicamente no puede dar ese voltaje).
#
# Umbrales elegidos con margen de seguridad para completar el apagado (5s):
BAT_LOW_V      = 11.2   # V → advertencia  (buck empieza a tener poco headroom)
BAT_CRITICAL_V = 11.0   # V → apagado      (0.1V antes del límite físico del buck)
BAT_FULL_V     = 12.6   # V → 100 %        (3S cargada: 4.2V × 3)
BAT_EMPTY_V    = 10.9   # V → 0 %          (límite físico del XL4015 a 10.9V salida)

# Tiempo de espera antes de ejecutar shutdown en el SO (Sistema Operativo = Linux de la Jetson)
SHUTDOWN_GRACE_S = 5.0


class TeensyBridgeNode(Node):

    def __init__(self):
        super().__init__('teensy_bridge_node')

        # ── Parámetros ──────────────────────────────────────────────
        self.declare_parameter('port',          '/dev/teensy')
        self.declare_parameter('baudrate',      921600)
        # wheel_radius: radio de la rueda en metros — MEDIR EN TU CHASSIS
        self.declare_parameter('wheel_radius',  0.033)
        # gear_ratio: vueltas del encoder del motor por cada vuelta de la rueda
        # Medir con encoder_test.ino: escribe "gear", gira 1 vuelta de la rueda, escribe "done"
        self.declare_parameter('gear_ratio',    9.5)
        # watchdog_s: si el nodo de control deja de publicar durante este tiempo,
        # el bridge envía C,0,0 a la Teensy para parar el motor.
        # Protege si ROS2 o el nodo de control se caen.
        # La latencia USB serial es < 1ms (conexión local), NO afecta el control normal.
        self.declare_parameter('watchdog_s',    0.5)
        # auto_shutdown: False durante desarrollo para no apagar la Jetson con batería baja
        # Usar: ros2 launch neuracar_bringup sensors.launch.py auto_shutdown:=false
        self.declare_parameter('auto_shutdown', True)

        self._port    = self.get_parameter('port').value
        self._baud    = self.get_parameter('baudrate').value
        self._R       = self.get_parameter('wheel_radius').value
        self._ratio   = self.get_parameter('gear_ratio').value
        self._wd_s    = self.get_parameter('watchdog_s').value
        self._auto_sd = self.get_parameter('auto_shutdown').value

        # ── Estado del sistema ───────────────────────────────────────
        self._bat_state     = 'NORMAL'  # NORMAL | LOW | CRITICAL | SHUTDOWN
        self._shutdown_sent = False
        # Timer Python que, tras SHUTDOWN_GRACE_S segundos, ejecuta
        # "sudo shutdown -h now" en el SO (Sistema Operativo = Linux de la Jetson)
        self._shutdown_timer = None

        # ── Conexión serial ─────────────────────────────────────────
        try:
            self._ser = serial.Serial(self._port, self._baud, timeout=0.1)
            self.get_logger().info(
                f'Teensy conectada: {self._port} @ {self._baud} baud'
            )
        except serial.SerialException as exc:
            self.get_logger().fatal(f'No se pudo abrir {self._port}: {exc}')
            raise

        # ── Publishers ──────────────────────────────────────────────
        self.pub_imu    = self.create_publisher(Imu,          '/neuracar/imu',           10)
        self.pub_enc    = self.create_publisher(JointState,   '/neuracar/encoder',        10)
        self.pub_bat    = self.create_publisher(BatteryState, '/neuracar/battery',        10)
        self.pub_status = self.create_publisher(String,       '/neuracar/status',         10)
        self.pub_sys    = self.create_publisher(String,       '/neuracar/system_status',  10)

        # ── Subscriber ──────────────────────────────────────────────
        self.sub_cmd = self.create_subscription(
            Vector3Stamped,
            '/neuracar/user_command',
            self._cmd_callback,
            10,
        )

        # ── Watchdog ────────────────────────────────────────────────
        # Corre cada 100ms. Si pasaron más de watchdog_s (0.5s) sin comando
        # envía parada de emergencia a la Teensy.
        # NO interfiere con el control normal (latencia USB < 1ms).
        # Protege únicamente cuando el nodo de control muere completamente.
        self._last_cmd_time = self.get_clock().now()
        self.create_timer(0.1, self._watchdog_cb)

        # ── Hilo lector serial ──────────────────────────────────────
        self._alive  = True
        self._reader = threading.Thread(target=self._serial_reader, daemon=True)
        self._reader.start()

        self.get_logger().info('Teensy bridge v1.3 iniciado.')
        self._publish_sys_status('READY')

    # ──────────────────────────────────────────────────────────────────
    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _send(self, data: bytes):
        try:
            self._ser.write(data)
        except serial.SerialException as exc:
            self.get_logger().error(f'Error escritura serial: {exc}')

    def _publish_sys_status(self, status: str):
        msg = String()
        msg.data = status
        self.pub_sys.publish(msg)

    # ──────────────────────────────────────────────────────────────────
    #  Subscriber: Jetson → Teensy
    # ──────────────────────────────────────────────────────────────────
    def _cmd_callback(self, msg: Vector3Stamped):
        self._last_cmd_time = self.get_clock().now()

        # Durante apagado solo enviamos parada — no comandos de movimiento
        if self._bat_state in ('CRITICAL', 'SHUTDOWN'):
            self._send(b'C,0.000,0.000\n')
            return

        throttle = max(-1.0, min(1.0, msg.vector.x))
        steering = max(-1.0, min(1.0, msg.vector.y))
        # "C" = Command: prefijo que la Teensy reconoce en parseLine()
        # con  if line[0] == 'C' and line[1] == ','
        self._send(f'C,{throttle:.3f},{steering:.3f}\n'.encode())

    # ──────────────────────────────────────────────────────────────────
    #  Watchdog
    # ──────────────────────────────────────────────────────────────────
    def _watchdog_cb(self):
        elapsed = (
            self.get_clock().now() - self._last_cmd_time
        ).nanoseconds * 1e-9
        if elapsed > self._wd_s:
            self._send(b'C,0.000,0.000\n')

    # ──────────────────────────────────────────────────────────────────
    #  Lector serial: Teensy a ROS2
    # ──────────────────────────────────────────────────────────────────
    def _serial_reader(self):
        while self._alive:
            try:
                raw = self._ser.readline()
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

    # ──────────────────────────────────────────────────────────────────
    #  Parser
    # ──────────────────────────────────────────────────────────────────
    def _parse_line(self, line: str):
        parts    = line.split(',')
        msg_type = parts[0]

        try:
            if msg_type == 'E' and len(parts) == 3:
                self._handle_encoder(float(parts[1]), float(parts[2]))

            elif msg_type == 'I' and len(parts) == 10:
                vals = [float(p) for p in parts[1:]]
                self._handle_imu(*vals)

            elif msg_type == 'B' and len(parts) == 3:
                self._handle_battery(float(parts[1]), float(parts[2]))

            elif msg_type == 'STA':
                self._handle_status(','.join(parts[1:]))

        except (ValueError, IndexError) as exc:
            self.get_logger().warn(f'Mensaje malformado "{line}": {exc}')

    # ──────────────────────────────────────────────────────────────────
    #  Encoder → JointState
    #
    #  División de responsabilidades (igual que QCar):
    #    Teensy  → calcula RPM del motor (precisión de microsegundos con hardware counter)
    #    ROS2    → convierte RPM a velocidad de rueda y lineal (parámetros cinemáticos)
    #
    #  Cálculo equivalente al QCar:
    #    wheel_rpm  = motor_rpm / gear_ratio
    #    wheel_rad_s = wheel_rpm × 2π / 60
    #    linear_m_s  = wheel_rad_s × wheel_radius
    # ──────────────────────────────────────────────────────────────────
    def _handle_encoder(self, angle_deg: float, motor_rpm: float):
        msg              = JointState()
        msg.header.stamp = self._stamp()
        msg.name         = ['drive_motor']

        # Posición continua acumulada [rad] — no wrappea en 2π (útil para odometría)
        angle_rad  = math.radians(angle_deg)

        # Cinemática: motor → rueda
        wheel_rpm   = motor_rpm / self._ratio
        wheel_rad_s = wheel_rpm * (2.0 * math.pi / 60.0)   # rad/s eje rueda
        linear_m_s  = wheel_rad_s * self._R                 # m/s vehículo

        msg.position = [angle_rad]    # rad — eje motor, acumulado
        msg.velocity = [wheel_rad_s]  # rad/s — eje rueda
        msg.effort   = [linear_m_s]   # m/s — velocidad lineal (convenio Neuracar)

        self.pub_enc.publish(msg)

    # ──────────────────────────────────────────────────────────────────
    #  IMU → sensor_msgs/Imu
    #
    #  Montaje físico del BNO055:
    #    Posición: adelante del vehículo
    #    Ejes físicos:  X = derecha  |  Y = adelante  |  Z = arriba
    #    (por regla de la mano derecha: X×Y=Z  derecha × adelante = arriba )
    #
    #  El BNO055 así montado ya entrega los ángulos correctos sin remapeo.
    #  frame_id = 'imu_link' — para que nav2 lo interprete correctamente
    #  necesitarás un static_transform_publisher de imu_link → base_link.
    #  Si el IMU está perfectamente centrado y orientado como ROS REP-103
    #  (X=adelante, Y=izquierda, Z=arriba) ese TF será la identidad.
    #  En tu montaje (X=derecha, Y=adelante) hay una rotación de -90° en Z.
    #
    #  Covarianzas BNO055 en modo NDOF (fusión interna activada):
    #    Orientación: precisión ±1° → varianza = (1°·π/180)² ≈ 3×10⁻⁴ rad²  
    #    Giroscopio:  ruido típico 0.01°/s/√Hz @ 100Hz → ≈ 1×10⁻⁴ (rad/s)²  
    #    Aceleración: ruido típico 0.3mg/√Hz         → ≈ 2×10⁻³ (m/s²)²     
    #    Son estimaciones razonables para el Stanley Controller.
    #    Si quieres valores exactos: coloca el sensor quieto 30s, mide
    #    la desviación estándar de cada eje y eleva al cuadrado.
    # ──────────────────────────────────────────────────────────────────
    def _handle_imu(
        self,
        heading: float, roll: float, pitch: float,
        ax: float, ay: float, az: float,
        gx: float, gy: float, gz: float,
    ):
        msg                 = Imu()
        msg.header.stamp    = self._stamp()
        msg.header.frame_id = 'imu_link'

        # BNO055 VECTOR_EULER: x=heading(yaw 0-360°), y=roll, z=pitch
        # Conversión Euler ZYX → cuaternión (convención estándar robótica)
        yaw = math.radians(heading)
        r   = math.radians(roll)
        p   = math.radians(pitch)

        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(p   * 0.5), math.sin(p   * 0.5)
        cr, sr = math.cos(r   * 0.5), math.sin(r   * 0.5)

        msg.orientation.w = cr * cp * cy + sr * sp * sy
        msg.orientation.x = sr * cp * cy - cr * sp * sy
        msg.orientation.y = cr * sp * cy + sr * cp * sy
        msg.orientation.z = cr * cp * sy - sr * sp * cy

        # BNO055 VECTOR_GYROSCOPE en rad/s
        msg.angular_velocity.x = gx
        msg.angular_velocity.y = gy
        msg.angular_velocity.z = gz

        # BNO055 VECTOR_LINEARACCEL en m/s² (sin gravedad)
        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az

        # Matrices de covarianza diagonales 3×3 (ver documentación arriba)
        ori_c  = [3e-4, 0, 0,  0, 3e-4, 0,  0, 0, 3e-4]
        gyro_c = [1e-4, 0, 0,  0, 1e-4, 0,  0, 0, 1e-4]
        acc_c  = [2e-3, 0, 0,  0, 2e-3, 0,  0, 0, 2e-3]
        msg.orientation_covariance         = ori_c
        msg.angular_velocity_covariance    = gyro_c
        msg.linear_acceleration_covariance = acc_c

        self.pub_imu.publish(msg)

    # ──────────────────────────────────────────────────────────────────
    #  Batería → BatteryState
    #
    #  Los sensores miden el voltaje RAW de la batería (antes del XL4015).
    #  Rango útil con XL4015 a 10.9V: 11.0V – 12.6V
    #  Porcentaje: 0% = 10.9V (límite físico buck), 100% = 12.6V
    # ──────────────────────────────────────────────────────────────────
    def _handle_battery(self, voltage: float, current: float):
        msg                         = BatteryState()
        msg.header.stamp            = self._stamp()
        msg.voltage                 = voltage   # V — batería raw (antes del buck)
        msg.current                 = current   # A — positivo = descarga
        msg.design_capacity         = 3.3       # Ah (3300 mAh)
        msg.power_supply_status     = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
        msg.present                 = True

        pct             = (voltage - BAT_EMPTY_V) / (BAT_FULL_V - BAT_EMPTY_V)
        msg.percentage  = float(max(0.0, min(1.0, pct)))
        msg.power_supply_health = (
            BatteryState.POWER_SUPPLY_HEALTH_GOOD
            if voltage >= BAT_LOW_V
            else BatteryState.POWER_SUPPLY_HEALTH_DEAD
        )
        self.pub_bat.publish(msg)

        if self._bat_state == 'LOW':
            self.get_logger().warn(
                f'Batería baja: {voltage:.2f} V  ({msg.percentage*100:.0f}%)'
                f' — XL4015 necesita >{BAT_CRITICAL_V+0.1:.1f}V para regular'
            )

    # ──────────────────────────────────────────────────────────────────
    #  Handler STA — máquina de estados + protocolo apagado
    # ──────────────────────────────────────────────────────────────────
    def _handle_status(self, status: str):
        msg      = String()
        msg.data = status
        self.pub_status.publish(msg)
        self.get_logger().info(f'[Teensy] {status}')

        if status == 'LOW_BAT' and self._bat_state == 'NORMAL':
            self._bat_state = 'LOW'
            self._publish_sys_status('LOW_BAT')
            self.get_logger().warn(
                f'¡Batería baja ({BAT_LOW_V}V)! '
                'El XL4015 pronto perderá regulación. Recarga pronto.'
            )

        elif status == 'CRITICAL_BAT' and self._bat_state != 'SHUTDOWN':
            self._bat_state = 'CRITICAL'
            self._publish_sys_status('CRITICAL_BAT')
            self.get_logger().error(
                f'¡Batería crítica ({BAT_CRITICAL_V}V)! '
                'Iniciando apagado seguro antes del límite del XL4015...'
            )
            self._initiate_shutdown()

        elif status == 'OVERCURRENT':
            self._publish_sys_status('OVERCURRENT')
            self.get_logger().error('¡Sobrecorriente en batería lógica! Motor detenido.')

    # ──────────────────────────────────────────────────────────────────
    #  Protocolo de apagado seguro
    #
    #  Secuencia:
    #    1. Enviar C,0,0    → Teensy para el motor inmediatamente
    #    2. Enviar SHUTDOWN  → Teensy actualiza LCD y entra en silencio
    #    3. Esperar 5s       → ROS2 termina sus nodos limpiamente
    #    4. sudo shutdown -h now → Linux (SO) apaga la Jetson
    #
    #  Requiere permiso sudo sin contraseña (configurar una vez):
    #    echo "$USER ALL=(ALL) NOPASSWD: /sbin/shutdown" |
    #    sudo tee /etc/sudoers.d/neuracar-shutdown
    # ──────────────────────────────────────────────────────────────────
    def _initiate_shutdown(self):
        if self._shutdown_sent:
            return
        self._shutdown_sent = True

        self._send(b'C,0.000,0.000\n')  # parar motor
        self._send(b'SHUTDOWN\n')        # avisar a Teensy

        self.get_logger().warn(
            f'Apagado del SO en {SHUTDOWN_GRACE_S:.0f}s...'
        )

        if self._auto_sd:
            self._shutdown_timer = threading.Timer(
                SHUTDOWN_GRACE_S, self._execute_system_shutdown
            )
            self._shutdown_timer.daemon = True
            self._shutdown_timer.start()

    def _execute_system_shutdown(self):
        self.get_logger().error('Ejecutando shutdown del SO (Linux)...')
        self._bat_state = 'SHUTDOWN'
        self._publish_sys_status('SHUTTING_DOWN')
        try:
            subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=True)
        except subprocess.CalledProcessError as exc:
            self.get_logger().error(
                f'No se pudo ejecutar shutdown: {exc}\n'
                'Verifica /etc/sudoers.d/neuracar-shutdown'
            )

    # ──────────────────────────────────────────────────────────────────
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
    node = TeensyBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()