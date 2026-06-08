#!/usr/bin/env python3
"""
calibrar_velocidad_recta.py — Neuracar v2.0
=============================================
Calibra la LUT throttle→velocidad en línea recta.
v2.0: más puntos en la zona del salto, LUT generada
automáticamente limpia y monotónica.

MODO DE OPERACIÓN:
  Para cada throttle en la lista:
    1. Cuenta regresiva de 3 segundos
    2. Arranca el motor durante RUN_S segundos
    3. Para y da TURN_S segundos para girar el carro
    4. Mide velocidad promedio en la segunda mitad del run

  Al terminar:
    - Imprime la tabla de mediciones
    - Detecta automáticamente el salto del ESC
    - Genera _LUT_STRAIGHT limpia y monotónica
    - Calcula stable_min_v y stable_min_throttle
    - Guarda CSV

Ctrl+C en cualquier momento — guarda lo que hay.
"""

import csv
import math
import os
import threading
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32


# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════

# Más puntos en la zona del salto (0.60-0.70) para ser precisos
THROTTLES = [
    0.50,                          # arranque — ¿mueve?
    0.58, 0.60, 0.61, 0.62, 0.63, # zona pre-salto
    0.64, 0.65, 0.66, 0.67, 0.68, # zona del salto
    0.70, 0.75,                    # post-salto
    0.80, 0.85, 0.90, 0.95, 1.00  # crucero
]

RUN_S   = 2.0    # segundos corriendo por punto
TURN_S  = 20.0   # segundos para girar el carro entre puntos
READY_S = 10.0   # pausa inicial

STEERING = 0.0   # línea recta

# Umbral para detectar salto abrupto (m/s entre puntos consecutivos)
JUMP_THRESHOLD = 0.25

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

        self.get_logger().info('Calibrador velocidad recta v2.0 iniciado')

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
        """Arranca RUN_S segundos, promedia la segunda mitad."""
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


def build_lut(results: list) -> dict:
    """
    A partir de las mediciones brutas, genera automáticamente:
      - LUT limpia y monotónica (sin puntos duplicados ni invertidos)
      - Zona del salto (jump_lo, jump_hi, jump_thr)
      - stable_min_v y stable_min_throttle

    Proceso:
      1. Filtrar puntos con velocidad > 0.05 m/s
      2. Detectar saltos > JUMP_THRESHOLD
      3. En zona pre-salto: usar el último punto estable
      4. En zona post-salto: usar el primer punto estable
      5. Para el resto: promediar puntos con velocidades similares
         (dentro de 0.05 m/s entre sí) para reducir ruido
      6. Ordenar y verificar monotonía
    """
    valid = [r for r in results if r['mean'] > 0.05]
    if not valid:
        return {'lut': [], 'jump_lo': None, 'jump_hi': None,
                'jump_thr_lo': None, 'jump_thr_hi': None}

    # Detectar salto
    jump_lo = jump_hi = jump_thr_lo = jump_thr_hi = None
    jump_idx = None
    for i in range(len(valid) - 1):
        dv = valid[i+1]['mean'] - valid[i]['mean']
        if dv > JUMP_THRESHOLD:
            jump_lo      = valid[i]['mean']
            jump_hi      = valid[i+1]['mean']
            jump_thr_lo  = valid[i]['throttle']
            jump_thr_hi  = valid[i+1]['throttle']
            jump_idx     = i
            break

    # Separar en pre-salto y post-salto
    if jump_idx is not None:
        pre  = valid[:jump_idx+1]
        post = valid[jump_idx+1:]
    else:
        pre  = []
        post = valid

    def cluster_average(pts, tol=0.05):
        """
        Agrupa puntos con velocidades similares (dentro de tol m/s)
        y promedia throttle y velocidad dentro de cada grupo.
        Devuelve un punto representativo por grupo.
        """
        if not pts:
            return []
        clusters = []
        current  = [pts[0]]
        for p in pts[1:]:
            if abs(p['mean'] - current[-1]['mean']) <= tol:
                current.append(p)
            else:
                clusters.append(current)
                current = [p]
        clusters.append(current)
        result = []
        for cl in clusters:
            avg_v   = sum(p['mean']     for p in cl) / len(cl)
            avg_thr = sum(p['throttle'] for p in cl) / len(cl)
            result.append({
                'throttle': round(avg_thr, 4),
                'mean':     round(avg_v,   4),
            })
        return result

    # Limpiar pre-salto: solo el último punto (más representativo)
    pre_clean = []
    if pre:
        # Usar el punto de máxima velocidad del pre-salto
        best_pre = max(pre, key=lambda r: r['mean'])
        pre_clean = [{'throttle': best_pre['throttle'],
                      'mean':     best_pre['mean']}]

    # Limpiar post-salto: agrupar y promediar
    post_clean = cluster_average(post, tol=0.05)

    # LUT completa: motor parado + pre-salto + post-salto
    lut = [{'throttle': 0.000, 'mean': 0.00}]  # motor parado
    lut.extend(pre_clean)
    lut.extend(post_clean)

    # Asegurar monotonía estricta en velocidad
    lut_mono = [lut[0]]
    for pt in lut[1:]:
        if pt['mean'] > lut_mono[-1]['mean'] + 0.01:
            lut_mono.append(pt)

    # stable_min_v = primera velocidad estable post-salto
    if jump_hi is not None:
        stable_min_v   = jump_hi
        stable_min_thr = jump_thr_hi
    elif post_clean:
        stable_min_v   = post_clean[0]['mean']
        stable_min_thr = post_clean[0]['throttle']
    else:
        stable_min_v   = 0.82
        stable_min_thr = 0.650

    return {
        'lut':          lut_mono,
        'jump_lo':      jump_lo,
        'jump_hi':      jump_hi,
        'jump_thr_lo':  jump_thr_lo,
        'jump_thr_hi':  jump_thr_hi,
        'stable_min_v': round(stable_min_v,   3),
        'stable_min_thr': round(stable_min_thr, 3),
    }


def print_results(results: list, lut_data: dict):
    """Imprime la tabla de mediciones y la LUT generada."""

    jump_lo  = lut_data.get('jump_lo')
    jump_hi  = lut_data.get('jump_hi')
    jump_thr = lut_data.get('jump_thr_hi')

    print('\n' + '='*62)
    print('  TABLA DE MEDICIONES BRUTAS')
    print('='*62)
    print(f'  {"throttle":>8}  →  {"velocidad":>9}  (std)')
    for r in results:
        salto = ''
        if jump_hi and abs(r['mean'] - jump_hi) < 0.05:
            salto = '  ← post-salto'
        elif jump_lo and abs(r['mean'] - jump_lo) < 0.05:
            salto = '  ← pre-salto'
        elif jump_lo and jump_hi and jump_lo < r['mean'] < jump_hi:
            salto = '  (zona inestable)'
        print(f'  {r["throttle"]:>8.3f}  →  {r["mean"]:>7.4f} m/s'
              f'  ({r["std"]:.3f}){salto}')

    if jump_lo:
        print(f'\n  ⚠ SALTO DETECTADO: {jump_lo:.3f} → {jump_hi:.3f} m/s')
        print(f'    throttle {lut_data["jump_thr_lo"]:.3f} → {jump_thr:.3f}')
        print(f'    ZONA NO USABLE: {jump_lo:.3f} - {jump_hi:.3f} m/s')

    print('\n' + '='*62)
    print('  LUT GENERADA — lista para copiar a velocity_pid_node.py')
    print('='*62)
    print()
    print('_LUT_STRAIGHT: LUT = [')
    for pt in lut_data['lut']:
        note = ''
        if pt['mean'] == 0.0:
            note = '   # motor parado'
        elif jump_hi and abs(pt['mean'] - jump_hi) < 0.05:
            note = '   # post-salto (stable_min)'
        elif jump_lo and abs(pt['mean'] - jump_lo) < 0.05:
            note = '   # pre-salto'
        print(f'    ({pt["mean"]:.3f}, {pt["throttle"]:.3f}),{note}')
    print(']')

    print()
    print('# Parámetros para feedforward_from_lut (velocity_pid_node.py):')
    print(f'# stable_min_v         = {lut_data["stable_min_v"]}')
    print(f'# stable_min_throttle  = {lut_data["stable_min_thr"]}')
    print()
    print('# En __init__ del nodo:')
    print(f"# self.declare_parameter('straight_stable_min_v',        "
          f"{lut_data['stable_min_v']})")
    print(f"# self.declare_parameter('straight_stable_min_throttle', "
          f"{lut_data['stable_min_thr']})")
    print('='*62)


def main():
    rclpy.init()
    node = CalibradorNode()
    spin_t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_t.start()

    print('\n' + '='*62)
    print('  CALIBRADOR VELOCIDAD EN RECTA v2.0 — Neuracar')
    print('='*62)
    print(f'  {len(THROTTLES)} puntos × {RUN_S}s run + {TURN_S}s giro')
    total = (READY_S + len(THROTTLES) * (RUN_S + TURN_S)) / 60
    print(f'  Tiempo estimado: {total:.1f} minutos')
    print(f'  Zona densa: 0.58-0.70 (detección precisa del salto)')
    print('='*62)
    print('\nInstrucciones:')
    print('  • Pon el carro apuntando hacia espacio libre (~5m)')
    print(f'  • Tienes {TURN_S:.0f}s para girar/recargar entre puntos')
    print('  • Ctrl+C en cualquier momento — guarda lo que hay')
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

            # Feedback inmediato del salto
            if len(results) >= 2:
                dv = results[-1]['mean'] - results[-2]['mean']
                if dv > JUMP_THRESHOLD:
                    print(f'  ⚠ SALTO: '
                          f'{results[-2]["mean"]:.4f}→{results[-1]["mean"]:.4f} m/s')

            print(f'  → real = {r["mean"]:.4f} m/s  (std={r["std"]:.3f})')

            if idx < len(THROTTLES) - 1:
                print(f'  Gira/carga el carro ({TURN_S:.0f}s)')
                t_turn = time.monotonic()
                while time.monotonic() - t_turn < TURN_S:
                    rem = int(TURN_S - (time.monotonic() - t_turn))
                    print(f'  {rem}s...', end='\r', flush=True)
                    time.sleep(1.0)
                print(' ' * 20, end='\r')
                print('  ¡Listo!')

    except KeyboardInterrupt:
        print('\n\nCtrl+C — procesando resultados...')
        node.stop()

    node.stop()

    # ── Generar LUT automáticamente ───────────────────────────────
    lut_data = build_lut(results)
    print_results(results, lut_data)

    # ── Guardar CSV ───────────────────────────────────────────────
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir  = os.path.expanduser('~/Workspaces/Neuracar')
    csv_path = os.path.join(out_dir, f'lut_recta_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['throttle', 'mean', 'std', 'n'])
        w.writeheader()
        w.writerows(results)
    print(f'\nCSV guardado: {csv_path}')

    # ── Guardar LUT generada también ─────────────────────────────
    lut_path = os.path.join(out_dir, f'lut_recta_{ts}_clean.csv')
    with open(lut_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['mean', 'throttle'])
        w.writeheader()
        w.writerows(lut_data['lut'])
    print(f'LUT limpia: {lut_path}')

    try: node.destroy_node()
    except Exception: pass
    try: rclpy.shutdown()
    except Exception: pass


if __name__ == '__main__':
    main()