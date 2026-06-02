#!/usr/bin/env python3
"""
esp32_actuadores_bridge.py — Neuracar v1.0
============================================
Nodo exclusivo para ESP32-A (actuadores).
Recibe comandos de /neuracar/user_command y los manda por serial.
Sin sensores, sin publishers pesados — latencia mínima.

Suscribe:
  /neuracar/user_command   (geometry_msgs/Vector3Stamped)
    vector.x = throttle ∈ [-1, 1]
    vector.y = steering  ∈ [-1, 1]

Publica:
  /neuracar/status         (std_msgs/String)  — solo eventos STA del ESP32-A
"""

import threading

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String

try:
    import serial
except ImportError:
    raise SystemExit('pip install pyserial --break-system-packages')


class ActuadoresBridgeNode(Node):

    def __init__(self):
        super().__init__('esp32_actuadores_bridge')

        self.declare_parameter('port',       '/dev/esp32a')
        self.declare_parameter('baudrate',   921600)
        self.declare_parameter('watchdog_s', 0.5)

        port         = self.get_parameter('port').value
        baud         = self.get_parameter('baudrate').value
        self._wd_s   = self.get_parameter('watchdog_s').value

        self._blocked = False   # True si batería crítica viene del nodo de sensores

        try:
            # Write timeout corto: si el CH340 no acepta en 20ms, descartar
            # Read timeout mínimo: el ESP32-A casi no habla
            self._ser = serial.Serial(port, baud,
                                      timeout=0.01,
                                      write_timeout=0.02)
            self.get_logger().info(f'ESP32-A conectada: {port} @ {baud}')
        except serial.SerialException as exc:
            self.get_logger().fatal(f'No se pudo abrir {port}: {exc}')
            raise

        # Publisher solo para eventos STA del ESP32-A
        self.pub_status = self.create_publisher(String, '/neuracar/status', 10)

        # Subscriber con callback group reentrant para no bloquear el watchdog
        cb_group = ReentrantCallbackGroup()

        self.sub_cmd = self.create_subscription(
            Vector3Stamped,
            '/neuracar/user_command',
            self._cmd_cb,
            10,
            callback_group=cb_group)

        # Escuchar system_status para bloquearse si batería crítica
        self.sub_sys = self.create_subscription(
            String,
            '/neuracar/system_status',
            self._sys_cb,
            10,
            callback_group=cb_group)

        self._last_cmd_time = self.get_clock().now()

        # Watchdog timer
        self.create_timer(0.1, self._watchdog_cb, callback_group=cb_group)

        # Hilo lector serial del ESP32-A (solo STA ocasionales)
        self._alive  = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self.get_logger().info('Actuadores bridge listo.')

    # ── Subscriber: comandos de control ─────────────────────────────

    def _cmd_cb(self, msg: Vector3Stamped):
        self._last_cmd_time = self.get_clock().now()
        if self._blocked:
            self._send(b'C,0.000,0.000\n')
            return
        t = max(-1.0, min(1.0, float(msg.vector.x)))
        s = max(-1.0, min(1.0, float(msg.vector.y)))
        self._send(f'C,{t:.3f},{s:.3f}\n'.encode())

    # ── Subscriber: system_status (batería crítica) ──────────────────

    def _sys_cb(self, msg: String):
        if msg.data in ('CRITICAL_BAT', 'SHUTTING_DOWN'):
            self._blocked = True
            self._send(b'C,0.000,0.000\n')
            self._send(b'SHUTDOWN\n')
            self.get_logger().warn(f'Actuadores bloqueados: {msg.data}')

    # ── Watchdog ────────────────────────────────────────────────────

    def _watchdog_cb(self):
        elapsed = (self.get_clock().now() - self._last_cmd_time).nanoseconds * 1e-9
        if elapsed > self._wd_s:
            self._send(b'C,0.000,0.000\n')

    # ── Serial TX ───────────────────────────────────────────────────

    def _send(self, data: bytes):
        try:
            self._ser.write(data)
        except serial.SerialTimeoutException:
            # CH340 ocupado — descartar este frame, el siguiente llegará en 20ms
            pass
        except serial.SerialException as exc:
            self.get_logger().error(f'Error escritura serial: {exc}')

    # ── Serial RX (solo STA del ESP32-A) ────────────────────────────

    def _read_loop(self):
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
                # Ignorar silenciosamente cualquier otra cosa
            except serial.SerialException as exc:
                if self._alive:
                    self.get_logger().error(f'Serial RX error: {exc}')
                break
            except Exception:
                pass

    # ── Cleanup ──────────────────────────────────────────────────────

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