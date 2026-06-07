#!/usr/bin/env python3
"""
calibrar_velocidad_recta.py — Neuracar
========================================
Calibra la LUT throttle→velocidad en línea recta.
Diseñado para espacio limitado (~5m) con tiempo para
girar el carro manualmente entre pruebas.

MODO DE OPERACIÓN:
  Para cada throttle en la lista:
    1. Avisa que pongas el carro listo (pausa configurable)
    2. Arranca el motor durante RUN_S segundos
    3. Para y da tiempo para girar el carro (TURN_S segundos)
    4. Mide velocidad promedio en la segunda mitad del run

  Al terminar imprime la LUT lista para copiar a velocity_pid_node.py

USO:
  # Terminal 1 — sensores corriendo
  ros2 launch neuracar_bringup sensors.launch.py camera:=false lidar:=false

  # Terminal 2
  python3 calibrar_velocidad_recta.py

CONFIGURACIÓN:
  Edita la sección CONFIG antes de correr.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32

import math
import threading
import time
from datetime import datetime


# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════

THROTTLES = [0.50, 0.58, 0.62, 0.65, 0.70, 0.75, 0.85, 1.00]

RUN_S   = 2.0    # segundos corriendo por punto
TURN_S  = 20.0   # segundos para girar/cargar entre puntos
READY_S = 10.0   # pausa inicial

STEERING = 0.0   # línea recta

# ══════════════════════════════════════════════════════════════════


class CalibradorNode(Node):

    def __init__(self):
        super().__init__('calibrador_velocidad')

        self.pub_cmd = self.create_publisher(
            Vector3Stamped, '/neuracar/user_command', 10)

        self.lock  = threading.Lock()
        self.speed = 0.0

        self.create_subscription(
            Float32, '/neuracar/wheel_speed', self._speed_cb, 10)

        self.get_logger().info('Calibrador velocidad recta iniciado')

    def _speed_cb(self, msg: Float32):
        with self.lock:
            self.speed = float(msg.data)

    def send(self, throttle: float):
        msg = Vector3Stamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.vector.x        = float(throttle)
        msg.vector.y        = float(STEERING)
        self.pub_cmd.publish(msg)

    def stop(self):
        msg = Vector3Stamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.vector.x        = 0.0
        msg.vector.y        = 0.0
        self.pub_cmd.publish(msg)

    def medir(self, throttle: float) -> dict:
        samples_all  = []
        samples_half = []
        t0   = time.monotonic()
        half = RUN_S / 2.0

        while time.monotonic() - t0 < RUN_S:
            self.send(throttle)
            elapsed = time.monotonic() - t0
            with self.lock:
                v = self.speed
            samples_all.append(v)
            if elapsed >= half:
                samples_half.append(v)
            time.sleep(0.02)

        self.stop()

        data = samples_half if samples_half else samples_all
        mean = sum(data) / len(data)
        std  = math.sqrt(sum((v - mean)**2 for v in data) / len(data))

        return {
            'throttle': throttle,
            'mean':     round(mean, 4),
            'std':      round(std,  4),
            'n':        len(data),
        }


def main():
    rclpy.init()
    node = CalibradorNode()
    spin_t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_t.start()

    print('\n' + '='*58)
    print('  CALIBRADOR VELOCIDAD EN RECTA — Neuracar')
    print('='*58)
    print(f'  {len(THROTTLES)} puntos × {RUN_S}s run + {TURN_S}s giro')
    total = (READY_S + len(THROTTLES) * (RUN_S + TURN_S)) / 60
    print(f'  Tiempo estimado: {total:.1f} minutos')
    print('='*58)
    print('\nInstrucciones:')
    print('  • Pon el carro apuntando hacia el espacio libre')
    print(f'  • Tienes {TURN_S:.0f}s para girar/cargar entre cada punto')
    print('  • Ctrl+C para terminar y guardar lo que hay')
    print(f'\nIniciando en {READY_S:.0f} segundos...\n')

    results = []

    try:
        time.sleep(READY_S)
    except KeyboardInterrupt:
        node.stop()
        node.destroy_node()
        return

    try:
        for idx, thr in enumerate(THROTTLES):
            print(f'\n[{idx+1}/{len(THROTTLES)}] throttle={thr:.3f}')
            print(f'  Baja el carro al suelo y apunta...')

            for i in range(3, 0, -1):
                print(f'  {i}...', end=' ', flush=True)
                time.sleep(1.0)
            print('¡ARRANCANDO!')

            r = node.medir(thr)
            results.append(r)

            # Detectar salto
            if len(results) >= 2:
                dv = results[-1]['mean'] - results[-2]['mean']
                if dv > 0.4:
                    print(f'  ⚠ SALTO: {results[-2]["mean"]:.3f}→{results[-1]["mean"]:.3f} m/s')

            print(f'  → velocidad real = {r["mean"]:.3f} m/s  (std={r["std"]:.3f})')

            if idx < len(THROTTLES) - 1:
                print(f'  Gira/carga el carro ({TURN_S:.0f}s)')
                t_turn = time.monotonic()
                while time.monotonic() - t_turn < TURN_S:
                    remaining = int(TURN_S - (time.monotonic() - t_turn))
                    print(f'  {remaining}s restantes...', end='\r', flush=True)
                    time.sleep(1.0)
                print(' ' * 30, end='\r')
                print('  ¡Listo!')

    except KeyboardInterrupt:
        print('\n\nCtrl+C — guardando resultados...')
        node.stop()

    node.stop()

    # ── Imprimir LUT ──────────────────────────────────────────────
    print('\n' + '='*58)
    print('  RESULTADO — LUT calibrada en línea recta')
    print('='*58)

    jump_lo = jump_hi = jump_thr_lo = jump_thr_hi = None
    for i in range(len(results) - 1):
        dv = results[i+1]['mean'] - results[i]['mean']
        if dv > 0.4:
            jump_lo     = results[i]['mean']
            jump_hi     = results[i+1]['mean']
            jump_thr_lo = results[i]['throttle']
            jump_thr_hi = results[i+1]['throttle']

    print('\nTabla de mediciones:')
    print('  throttle  →  velocidad  (std)')
    for r in results:
        salto = ' ← SALTO' if (jump_hi and abs(r['mean'] - jump_hi) < 0.05) else ''
        print(f'  {r["throttle"]:.3f}    →  {r["mean"]:.3f} m/s'
              f'  ({r["std"]:.3f}){salto}')

    if jump_lo:
        print(f'\n  ⚠ ZONA NO USABLE: {jump_lo:.3f} - {jump_hi:.3f} m/s')

    print('\n' + '-'*58)
    print('  Copia esto a velocity_pid_node.py:\n')
    print('_LUT = [')
    print('    (0.00, 0.000),   # motor parado')
    for r in results:
        if r['mean'] > 0.05:
            note = ''
            if jump_hi and abs(r['mean'] - jump_hi) < 0.05:
                note = '   # post-salto'
            elif jump_lo and abs(r['mean'] - jump_lo) < 0.05:
                note = '   # pre-salto'
            print(f'    ({r["mean"]:.3f}, {r["throttle"]:.3f}),{note}')
    print(']')

    if jump_lo:
        print(f'\n# En lut_feedforward zona del salto:')
        print(f'# if {jump_lo:.3f} < v_abs < {jump_hi:.3f}:')
        print(f'#     return sign * {jump_thr_hi:.3f}')

    print('='*58)

    import csv, os
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(
        os.path.expanduser('~/Workspaces/Neuracar'),
        f'lut_recta_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['throttle', 'mean', 'std', 'n'])
        w.writeheader()
        w.writerows(results)
    print(f'CSV guardado: {csv_path}')

    try: node.destroy_node()
    except Exception: pass
    try: rclpy.shutdown()
    except Exception: pass


if __name__ == '__main__':
    main()