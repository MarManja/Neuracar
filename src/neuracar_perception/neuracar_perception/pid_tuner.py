#!/usr/bin/env python3
"""
pid_tuner.py — Neuracar PID Auto-Tuner
========================================
Corre el PID con diferentes combinaciones de gains, mide la
velocidad real durante N segundos y genera un reporte CSV + texto.

No necesitas el dashboard — solo correr este script y leer el reporte.

USO:
  # 1. Asegúrate de que el bridge de sensores está corriendo:
  #    ros2 launch neuracar_bringup sensors.launch.py camera:=false lidar:=false
  #
  # 2. En otra terminal, publica steering centro:
  #    ros2 topic pub /neuracar/cmd_steering std_msgs/msg/Float32 "{data: 0.0}" -r 50
  #
  # 3. Corre este script:
  #    python3 pid_tuner.py

  El script:
    - Para cada combinación de gains en GAIN_SETS
    - Para cada velocidad objetivo en SETPOINTS
    - Publica el setpoint durante SETTLE_S segundos (espera convergencia)
    - Luego mide durante MEASURE_S segundos
    - Calcula: promedio, std, error medio, overshoot, tiempo de convergencia
    - Guarda reporte en pid_tuning_results.csv y pid_tuning_report.txt

IMPORTANTE: el carro debe estar al aire con la rueda girando libre.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32

import csv
import math
import os
import subprocess
import time
import threading
from datetime import datetime
from collections import deque

# ══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — edita aquí
# ══════════════════════════════════════════════════════════════════

# Velocidades objetivo a probar [m/s]
SETPOINTS = [0.3, 0.5, 1.0, 1.5, 2.0]

# Combinaciones de gains a probar
# Formato: {'kp': X, 'ki': Y, 'kd': Z, 'max_integral': W}
GAIN_SETS = [
    {'kp': 0.04, 'ki': 0.08, 'kd': 0.01, 'max_integral': 0.05},   # baseline
    {'kp': 0.04, 'ki': 0.12, 'kd': 0.01, 'max_integral': 0.08},   # más ki
    {'kp': 0.06, 'ki': 0.10, 'kd': 0.01, 'max_integral': 0.08},   # más kp
    {'kp': 0.04, 'ki': 0.08, 'kd': 0.02, 'max_integral': 0.05},   # más kd
    {'kp': 0.05, 'ki': 0.15, 'kd': 0.01, 'max_integral': 0.10},   # ki agresivo
]

SETTLE_S  = 4.0   # segundos esperando convergencia antes de medir
MEASURE_S = 5.0   # segundos de medición
PAUSE_S   = 2.0   # pausa entre pruebas (motor a neutro)

# Parámetros fijos del PID
MAX_THROTTLE = 0.72
ALPHA        = 0.3
MAX_RATE     = 2.0
V_DEADBAND   = 0.05

# ══════════════════════════════════════════════════════════════════


class PIDTunerNode(Node):

    def __init__(self):
        super().__init__('pid_tuner')

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Publicadores
        self.pub_vel = self.create_publisher(
            Float32, '/neuracar/cmd_velocity', 10)
        self.pub_str = self.create_publisher(
            Float32, '/neuracar/cmd_steering', 10)

        # Subscriber velocidad real
        self.wheel_speed = 0.0
        self.speed_lock  = threading.Lock()
        self.create_subscription(
            Float32, '/neuracar/wheel_speed', self._speed_cb, 10)

        # Buffer de mediciones
        self._measuring   = False
        self._samples     = []
        self._conv_time   = None   # tiempo hasta convergencia
        self._sp_current  = 0.0

        self.get_logger().info('PID Tuner iniciado')

    def _speed_cb(self, msg: Float32):
        v = float(msg.data)
        with self.speed_lock:
            self.wheel_speed = v
            if self._measuring:
                self._samples.append((time.monotonic(), v))
                # Detectar convergencia: error < 5% del setpoint por 0.5s
                if (self._conv_time is None and
                        self._sp_current > 0.1 and
                        abs(v - self._sp_current) / self._sp_current < 0.08):
                    self._conv_time = time.monotonic()

    def set_velocity(self, v: float):
        msg = Float32(); msg.data = float(v)
        self.pub_vel.publish(msg)
        msg2 = Float32(); msg2.data = 0.0
        self.pub_str.publish(msg2)

    def stop(self):
        self.set_velocity(0.0)

    def measure(self, setpoint: float) -> dict:
        """
        Publica setpoint, espera convergencia y mide durante MEASURE_S.
        Devuelve estadísticas de la prueba.
        """
        with self.speed_lock:
            self._samples    = []
            self._conv_time  = None
            self._sp_current = setpoint

        # Fase de settle — publicar setpoint y esperar
        t_start = time.monotonic()
        while time.monotonic() - t_start < SETTLE_S:
            self.set_velocity(setpoint)
            time.sleep(0.02)

        # Fase de medición
        with self.speed_lock:
            self._measuring = True
            meas_start      = time.monotonic()
            conv_ref        = self._conv_time

        while time.monotonic() - meas_start < MEASURE_S:
            self.set_velocity(setpoint)
            time.sleep(0.02)

        with self.speed_lock:
            self._measuring = False
            samples         = list(self._samples)
            conv_time       = self._conv_time

        self.stop()

        if not samples:
            return {'error': 'sin datos'}

        speeds  = [s[1] for s in samples]
        times   = [s[0] for s in samples]
        t0      = times[0]

        mean_v  = sum(speeds) / len(speeds)
        std_v   = math.sqrt(sum((v - mean_v)**2 for v in speeds) / len(speeds))
        mean_err = setpoint - mean_v
        max_v   = max(speeds)
        min_v   = min(speeds)
        overshoot = max(0.0, max_v - setpoint)

        # Tiempo hasta convergencia desde inicio del settle
        conv_s = (conv_time - (meas_start - SETTLE_S)
                  if conv_time else None)

        # Porcentaje de muestras dentro de ±8 cm/s (umbral "ajustando")
        pct_ok = sum(1 for v in speeds
                     if abs(v - setpoint) < 0.08) / len(speeds) * 100

        return {
            'setpoint':    setpoint,
            'mean':        round(mean_v, 4),
            'std':         round(std_v, 4),
            'mean_error':  round(mean_err, 4),
            'max':         round(max_v, 4),
            'min':         round(min_v, 4),
            'overshoot':   round(overshoot, 4),
            'conv_time_s': round(conv_s, 2) if conv_s else 'no convergió',
            'pct_within_8cm': round(pct_ok, 1),
            'n_samples':   len(samples),
        }


def run_pid_node(gains: dict) -> subprocess.Popen:
    """Lanza el nodo PID con los gains dados como proceso separado."""
    cmd = [
        'ros2', 'run', 'neuracar_perception', 'velocity_pid_node',
        '--ros-args',
        '-p', f'kp:={gains["kp"]}',
        '-p', f'ki:={gains["ki"]}',
        '-p', f'kd:={gains["kd"]}',
        '-p', f'max_integral:={gains["max_integral"]}',
        '-p', f'max_throttle:={MAX_THROTTLE}',
        '-p', f'alpha:={ALPHA}',
        '-p', f'max_rate:={MAX_RATE}',
        '-p', f'v_deadband:={V_DEADBAND}',
    ]
    return subprocess.Popen(cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def main():
    rclpy.init()
    node = PIDTunerNode()

    # Hilo ROS2
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Esperar a que los topics estén disponibles
    print('\n' + '='*60)
    print('  NEURACAR PID AUTO-TUNER')
    print('='*60)
    print(f'  {len(GAIN_SETS)} combinaciones × {len(SETPOINTS)} velocidades')
    print(f'  {SETTLE_S}s settle + {MEASURE_S}s medición por prueba')
    total_min = (len(GAIN_SETS) * len(SETPOINTS) *
                 (SETTLE_S + MEASURE_S + PAUSE_S)) / 60
    print(f'  Tiempo estimado: {total_min:.1f} minutos')
    print('='*60)
    print('\nAsegúrate de que:')
    print('  1. El bridge de sensores está corriendo')
    print('  2. El carro está al aire con rueda libre')
    print('  3. Batería cargada')
    print('\nIniciando en 5 segundos... (Ctrl+C para cancelar)\n')
    time.sleep(5)

    results  = []
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir  = os.path.expanduser('~/Workspaces/Neuracar')
    csv_path = os.path.join(out_dir, f'pid_tuning_{ts}.csv')
    txt_path = os.path.join(out_dir, f'pid_tuning_{ts}.txt')

    total    = len(GAIN_SETS) * len(SETPOINTS)
    done     = 0

    for gi, gains in enumerate(GAIN_SETS):
        print(f'\n{"─"*60}')
        print(f'  GAINS {gi+1}/{len(GAIN_SETS)}: '
              f'kp={gains["kp"]} ki={gains["ki"]} '
              f'kd={gains["kd"]} max_int={gains["max_integral"]}')
        print(f'{"─"*60}')

        # Lanzar el nodo PID con estos gains
        pid_proc = run_pid_node(gains)
        time.sleep(1.5)   # esperar a que arranque

        for sp in SETPOINTS:
            done += 1
            print(f'  [{done}/{total}] Setpoint {sp:.1f} m/s ... ', end='', flush=True)

            stats = node.measure(sp)
            stats['kp']  = gains['kp']
            stats['ki']  = gains['ki']
            stats['kd']  = gains['kd']
            stats['max_integral'] = gains['max_integral']
            results.append(stats)

            if 'error' in stats:
                print(f'ERROR: {stats["error"]}')
            else:
                status = ('✓' if abs(stats['mean_error']) < 0.08
                          else '~' if abs(stats['mean_error']) < 0.15
                          else '✗')
                print(f'{status} real={stats["mean"]:.3f} '
                      f'err={stats["mean_error"]:+.3f} '
                      f'std={stats["std"]:.3f} '
                      f'ok={stats["pct_within_8cm"]:.0f}% '
                      f'conv={stats["conv_time_s"]}s')

            time.sleep(PAUSE_S)

        # Matar el nodo PID
        pid_proc.terminate()
        pid_proc.wait(timeout=3)
        time.sleep(1.0)

    # ── Guardar CSV ───────────────────────────────────────────────
    if results and 'error' not in results[0]:
        fieldnames = ['kp', 'ki', 'kd', 'max_integral', 'setpoint',
                      'mean', 'std', 'mean_error', 'max', 'min',
                      'overshoot', 'conv_time_s', 'pct_within_8cm', 'n_samples']
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            w.writerows(results)
        print(f'\nCSV guardado: {csv_path}')

    # ── Generar reporte texto ─────────────────────────────────────
    with open(txt_path, 'w') as f:
        f.write('NEURACAR PID TUNING REPORT\n')
        f.write(f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Settle: {SETTLE_S}s  Measure: {MEASURE_S}s\n')
        f.write('='*70 + '\n\n')

        # Ranking por promedio de pct_within_8cm sobre todos los setpoints
        from collections import defaultdict
        gain_scores = defaultdict(list)
        for r in results:
            if 'error' not in r:
                key = (r['kp'], r['ki'], r['kd'], r['max_integral'])
                gain_scores[key].append(r['pct_within_8cm'])

        ranked = sorted(gain_scores.items(),
                        key=lambda x: sum(x[1])/len(x[1]),
                        reverse=True)

        f.write('RANKING DE GAINS (por % tiempo dentro de ±8cm/s)\n')
        f.write('-'*70 + '\n')
        for rank, (key, scores) in enumerate(ranked, 1):
            kp, ki, kd, mi = key
            avg = sum(scores) / len(scores)
            f.write(f'  #{rank}: kp={kp} ki={ki} kd={kd} max_int={mi}'
                    f' → promedio {avg:.1f}%\n')

        f.write('\n\nDETALLE POR GAINS Y SETPOINT\n')
        f.write('-'*70 + '\n')
        for gi, gains in enumerate(GAIN_SETS):
            key = (gains['kp'], gains['ki'], gains['kd'], gains['max_integral'])
            f.write(f'\nGains {gi+1}: kp={gains["kp"]} ki={gains["ki"]} '
                    f'kd={gains["kd"]} max_int={gains["max_integral"]}\n')
            for r in results:
                if 'error' in r: continue
                if (r['kp'] == gains['kp'] and r['ki'] == gains['ki'] and
                        r['kd'] == gains['kd']):
                    status = ('✓ CONVERGIDO' if abs(r['mean_error']) < 0.08
                              else '~ AJUSTANDO' if abs(r['mean_error']) < 0.15
                              else '✗ ERROR ALTO')
                    f.write(f'  {r["setpoint"]:.1f} m/s → '
                            f'real={r["mean"]:.3f} '
                            f'err={r["mean_error"]:+.3f} '
                            f'std={r["std"]:.3f} '
                            f'ok={r["pct_within_8cm"]:.0f}% '
                            f'conv={r["conv_time_s"]}s '
                            f'→ {status}\n')

        # Mejor combinación por setpoint
        f.write('\n\nMEJOR GAIN POR VELOCIDAD\n')
        f.write('-'*70 + '\n')
        for sp in SETPOINTS:
            sp_results = [r for r in results
                          if 'error' not in r and r['setpoint'] == sp]
            if not sp_results:
                continue
            best = min(sp_results, key=lambda r: abs(r['mean_error']))
            f.write(f'  {sp:.1f} m/s → kp={best["kp"]} ki={best["ki"]} '
                    f'kd={best["kd"]} '
                    f'(err={best["mean_error"]:+.3f} '
                    f'ok={best["pct_within_8cm"]:.0f}%)\n')

    print(f'Reporte guardado: {txt_path}')
    print('\n¡Tuning completado!')

    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()