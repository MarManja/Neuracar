"""
esp32_actuadores_node.py — NeuraCar
══════════════════════════════════════════════════════════════════
Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026

Serial bridge for the actuation ESP32 (port /dev/esp32a, 921600 baud).
Implements a latest-only TX pattern: a dedicated thread transmits at
20 Hz, always sending the most recent command and discarding any
intermediate values. This prevents CH340 buffer accumulation that
caused servo hunting in earlier revisions.

A Python-side watchdog zeroes the command if no
/neuracar/user_command message arrives within watchdog_s seconds,
ensuring the ESC and servo return to neutral on communication loss.

Serial frame sent:
  C,{throttle:.3f},{steering:.3f}   @ 20 Hz   values in [-1, 1]
  SHUTDOWN                          on ROS shutdown

Subscriptions:
  /neuracar/user_command  geometry_msgs/Vector3Stamped
                          vector.x = throttle [-1, 1]
                          vector.y = steering [-1, 1]

Parameters:
  port       (str,   /dev/esp32a): Serial port — must be upper-left
                                   USB-A port on Jetson carrier board
  baudrate   (int,   921600):      Serial baud rate
  watchdog_s (float, 0.50):        Timeout before zeroing command [s]
══════════════════════════════════════════════════════════════════
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
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            self.get_logger().info(f'ESP32-A conectada: {port} @ {baud}')
        except serial.SerialException as exc:
            self.get_logger().fatal(f'No se pudo abrir {port}: {exc}')
            raise

        self.pub_status = self.create_publisher(String, '/neuracar/status', 10)

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

        self._alive = True
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._tx_thread.start()

        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

        self.get_logger().info(f'Actuadores bridge listo. TX @ {TX_HZ} Hz')


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
                cmd = b'C,0.000,0.000\n'
                with self._lock:
                    self._throttle = 0.0
                    self._steering = 0.0
            else:
                cmd = f'C,{throttle:.3f},{steering:.3f}\n'.encode()

            self._serial_write(cmd)

            elapsed_loop = time.monotonic() - loop_start
            sleep_t = interval - elapsed_loop
            if sleep_t > 0:
                time.sleep(sleep_t)

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


    def _serial_write(self, data: bytes):
        try:
            self._ser.write(data)
        except serial.SerialTimeoutException:
            pass 
        except serial.SerialException as exc:
            self.get_logger().error(f'TX error: {exc}')


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