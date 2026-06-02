#!/usr/bin/env python3
"""
esp32_actuadores_node.py — Neuracar v2.0
==========================================
Patrón latest-only: un hilo TX dedicado manda solo el comando
más reciente a frecuencia fija. Nunca acumula comandos en cola.

Cambios vs v1.0:
  - _cmd_cb() solo GUARDA el último comando (no manda nada)
  - Hilo _tx_loop() manda al serial a TX_HZ fijo
  - QoS depth=1 en el subscriber: ROS2 solo guarda el último mensaje
  - reset_input_buffer() al iniciar para limpiar basura del CH340
  - Watchdog integrado en el hilo TX (sin timer separado)
"""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String

try:
    import serial
except ImportError:
    raise SystemExit('pip install pyserial --break-system-packages')

# Frecuencia a la que el hilo TX manda comandos al ESP32-A
# 20 Hz es suficiente — el ESP32-A tiene watchdog de 500ms
# Bajar esto reduce drásticamente la acumulación en el buffer del CH340
TX_HZ = 20


class ActuadoresBridgeNode(Node):

    def __init__(self):
        super().__init__('esp32_actuadores_node')

        self.declare_parameter('port',       '/dev/esp32a')
        self.declare_parameter('baudrate',   921600)
        self.declare_parameter('watchdog_s', 0.5)

        port       = self.get_parameter('port').value
        baud       = self.get_parameter('baudrate').value
        self._wd_s = self.get_parameter('watchdog_s').value

        # Último comando — solo este se manda, los anteriores se descartan
        self._lock      = threading.Lock()
        self._throttle  = 0.0
        self._steering  = 0.0
        self._last_cmd_t = time.monotonic()
        self._blocked   = False

        try:
            self._ser = serial.Serial(
                port, baud,
                timeout=0.01,
                write_timeout=0.02)
            # Limpiar buffer del CH340 al arrancar — elimina basura acumulada
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            self.get_logger().info(f'ESP32-A conectada: {port} @ {baud}')
        except serial.SerialException as exc:
            self.get_logger().fatal(f'No se pudo abrir {port}: {exc}')
            raise

        self.pub_status = self.create_publisher(String, '/neuracar/status', 10)

        # QoS depth=1: ROS2 solo guarda el mensaje más reciente en la cola
        # Si llegan 10 mensajes antes de que el callback corra, solo procesa el último
        qos_latest = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        cb_group = ReentrantCallbackGroup()

        self.sub_cmd = self.create_subscription(
            Vector3Stamped,
            '/neuracar/user_command',
            self._cmd_cb,
            qos_latest,
            callback_group=cb_group)

        self.sub_sys = self.create_subscription(
            String,
            '/neuracar/system_status',
            self._sys_cb,
            10,
            callback_group=cb_group)

        # Hilo TX dedicado — manda solo el último comando a TX_HZ
        self._alive = True
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._tx_thread.start()

        # Hilo RX — solo STA ocasionales del ESP32-A
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

        self.get_logger().info(f'Actuadores bridge v2.0 listo. TX @ {TX_HZ} Hz')

    # ── Callbacks ROS2 — solo guardan el estado, NO mandan serial ────

    def _cmd_cb(self, msg: Vector3Stamped):
        t = max(-1.0, min(1.0, float(msg.vector.x)))
        s = max(-1.0, min(1.0, float(msg.vector.y)))
        with self._lock:
            self._throttle   = t
            self._steering   = s
            self._last_cmd_t = time.monotonic()

    def _sys_cb(self, msg: String):
        if msg.data in ('CRITICAL_BAT', 'SHUTTING_DOWN'):
            with self._lock:
                self._blocked   = True
                self._throttle  = 0.0
                self._steering  = 0.0
            self._serial_write(b'C,0.000,0.000\n')
            self._serial_write(b'SHUTDOWN\n')
            self.get_logger().warn(f'Actuadores bloqueados: {msg.data}')

    # ── Hilo TX — único punto que escribe al serial ───────────────────

    def _tx_loop(self):
        interval = 1.0 / TX_HZ
        while self._alive:
            loop_start = time.monotonic()

            with self._lock:
                throttle  = self._throttle
                steering  = self._steering
                elapsed   = loop_start - self._last_cmd_t
                blocked   = self._blocked

            if blocked:
                cmd = b'C,0.000,0.000\n'
            elif elapsed > self._wd_s:
                # Watchdog: sin comando reciente → neutro
                cmd = b'C,0.000,0.000\n'
                with self._lock:
                    self._throttle = 0.0
                    self._steering = 0.0
            else:
                cmd = f'C,{throttle:.3f},{steering:.3f}\n'.encode()

            self._serial_write(cmd)

            # Dormir el tiempo restante para mantener TX_HZ exacto
            elapsed_loop = time.monotonic() - loop_start
            sleep_t = interval - elapsed_loop
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ── Hilo RX — solo lee STA del ESP32-A ───────────────────────────

    def _rx_loop(self):
        while self._alive:
            try:
                raw = self._ser.readline()
                if not raw:
                    continue
                line = raw.decode('utf-8', errors='ignore').strip()
                if line.startswith('STA,'):
                    status = line[4:]
                    m = String(); m.data = f'ESP32A,{status}'
                    self.pub_status.publish(m)
                    if 'ERR' in status:
                        self.get_logger().error(f'[ESP32-A] {status}')
                    elif 'ESTOP' in status:
                        self.get_logger().warn(f'[ESP32-A] {status}')
                    else:
                        self.get_logger().info(f'[ESP32-A] {status}')
            except serial.SerialException as exc:
                if self._alive:
                    self.get_logger().error(f'RX error: {exc}')
                break
            except Exception:
                pass

    # ── Serial write con protección ───────────────────────────────────

    def _serial_write(self, data: bytes):
        try:
            self._ser.write(data)
        except serial.SerialTimeoutException:
            pass  # CH340 ocupado — el siguiente ciclo TX lo reintenta
        except serial.SerialException as exc:
            self.get_logger().error(f'TX error: {exc}')

    # ── Cleanup ───────────────────────────────────────────────────────

    def destroy_node(self):
        self._alive = False
        try:
            self._ser.write(b'C,0.000,0.000\n')
            self._ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ActuadoresBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()