#!/usr/bin/env python3
"""
pid_tuner_pista.py — Neuracar PID Tuner en Pista
==================================================
Calibra la LUT y los gains del PID con el carro en pista real,
girando en círculo con steering fijo (no necesita trayectoria grabada).

MODO DE OPERACIÓN:
  1. El carro gira en círculo con steering=1.0 (o el valor que configures)
  2. Para cada throttle fijo: mide velocidad real → construye LUT
  3. Para cada combinación de gains: mide convergencia → elige mejores
  4. Genera reporte con LUT calibrada y gains óptimos listos para copiar

USO:
  # Terminal 1 — sensores corriendo
  ros2 launch neuracar_bringup sensors.launch.py camera:=false lidar:=false

  # Terminal 2 — este script
  python3 pid_tuner_pista.py

  El script controla el carro solo. No necesitas otras terminales.
  Asegúrate de tener espacio libre alrededor del carro (radio ~0.5m).

CONFIGURACIÓN:
  Edita la sección CONFIG antes de correr.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Bool, Float32

import csv
import math
import os
import time
import threading
from datetime import datetime
from collections import deque

# ══════════════════════════════════════════════════════════════════
#  CONFIG — edita aquí antes de correr
# ══════════════════════════════════════════════════════════════════

# Steering fijo para el círculo — 1.0 = máximo giro
# Ajusta si quieres un círculo más grande (0.7-0.8) o más cerrado (1.0)
STEERING = 1.0

# ── FASE 1: Calibración LUT ───────────────────────────────────────
# Throttles a medir para construir la curva real del motor en pista
LUT_THROTTLES = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55,
                 0.58, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

# Tiempo de estabilización por punto (segundos)
# Más tiempo = medición más precisa pero más lenta
LUT_SETTLE_S  = 3.0   # esperar a que la velocidad se estabilice
LUT_MEASURE_S = 4.0   # medir durante este tiempo y promediar

# ── FASE 2: Tuning de gains ───────────────────────────────────────
# Velocidades a probar (solo las que caen fuera del salto del ESC)
TUNE_SETPOINTS = [1.0, 1.5, 2.0]

# Combinaciones de gains a probar
GAIN_SETS = [
    # ── Baseline del tuner anterior ───────────────────────────────
    {'kp': 0.04, 'ki': 0.08, 'kd': 0.01, 'max_integral': 0.05},
    {'kp': 0.04, 'ki': 0.12, 'kd': 0.01, 'max_integral': 0.08},
    {'kp': 0.05, 'ki': 0.15, 'kd': 0.01, 'max_integral': 0.10},
    {'kp': 0.06, 'ki': 0.15, 'kd': 0.01, 'max_integral': 0.10},
    {'kp': 0.05, 'ki': 0.20, 'kd': 0.02, 'max_integral': 0.12},

    # ── kp bajo, barrido de ki ─────────────────────────────────────
    # El FF hace el trabajo pesado — kp solo corrige residual
    {'kp': 0.03, 'ki': 0.10, 'kd': 0.01, 'max_integral': 0.08},
    {'kp': 0.03, 'ki': 0.15, 'kd': 0.01, 'max_integral': 0.10},
    {'kp': 0.03, 'ki': 0.20, 'kd': 0.01, 'max_integral': 0.12},
    {'kp': 0.03, 'ki': 0.25, 'kd': 0.01, 'max_integral': 0.15},

    # ── kp medio, barrido de ki ───────────────────────────────────
    {'kp': 0.05, 'ki': 0.10, 'kd': 0.01, 'max_integral': 0.08},
    {'kp': 0.05, 'ki': 0.25, 'kd': 0.01, 'max_integral': 0.15},
    {'kp': 0.05, 'ki': 0.30, 'kd': 0.02, 'max_integral': 0.15},

    # ── kp alto, más agresivo ─────────────────────────────────────
    {'kp': 0.08, 'ki': 0.15, 'kd': 0.01, 'max_integral': 0.10},
    {'kp': 0.08, 'ki': 0.20, 'kd': 0.02, 'max_integral': 0.12},
    {'kp': 0.10, 'ki': 0.20, 'kd': 0.02, 'max_integral': 0.12},

    # ── kd alto — más amortiguación contra oscilación ─────────────
    {'kp': 0.05, 'ki': 0.15, 'kd': 0.05, 'max_integral': 0.10},
    {'kp': 0.05, 'ki': 0.20, 'kd': 0.05, 'max_integral': 0.12},
    {'kp': 0.06, 'ki': 0.15, 'kd': 0.05, 'max_integral': 0.10},

    # ── max_integral alto — más acción integral acumulada ─────────
    {'kp': 0.04, 'ki': 0.15, 'kd': 0.01, 'max_integral': 0.20},
    {'kp': 0.05, 'ki': 0.15, 'kd': 0.01, 'max_integral': 0.20},
    {'kp': 0.05, 'ki': 0.20, 'kd': 0.01, 'max_integral': 0.20},
]

TUNE_SETTLE_S  = 5.0   # más tiempo en pista — la inercia real tarda más
TUNE_MEASURE_S = 8.0   # más ciclos de medición
PAUSE_S        = 3.0   # pausa entre pruebas (motor a neutro)

MAX_THROTTLE = 1.0
# ══════════════════════════════════════════════════════════════════


class PistaTunerNode(Node):

    def __init__(self):
        super().__init__('pid_tuner_pista')

        # Publisher directo al ESP32 (bypasea el PID para fase LUT)
        self.pub_cmd = self.create_publisher(
            Vector3Stamped, '/neuracar/user_command', 10)

        # Publishers para el PID (fase tuning)
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.pub_vel = self.create_publisher(Float32, '/neuracar/cmd_velocity', qos)
        self.pub_str = self.create_publisher(Float32, '/neuracar/cmd_steering',  qos)

        # Subscriber velocidad real
        self.lock        = threading.Lock()
        self.speed       = 0.0
        self.speed_hist  = deque(maxlen=500)
        self.measuring   = False
        self.sp_current  = 0.0
        self.conv_time   = None

        # Parada de emergencia por LiDAR
        self.obstacle       = False
        self.obstacle_stops = 0

        self.create_subscription(
            Float32, '/neuracar/wheel_speed', self._speed_cb, 10)

        self.create_subscription(
            Bool, '/neuracar/lidar/obstacle_alert_rear', self._obs_cb, 10)

        self.get_logger().info('PID Tuner Pista iniciado — LiDAR activo')

    def _speed_cb(self, msg: Float32):
        v = float(msg.data)
        with self.lock:
            self.speed = v
            if self.measuring:
                self.speed_hist.append((time.monotonic(), v))
                if (self.conv_time is None and
                        self.sp_current > 0.1 and
                        abs(v - self.sp_current) / self.sp_current < 0.08):
                    self.conv_time = time.monotonic()

    def _obs_cb(self, msg: Bool):
        """Recibe alerta ya procesada del obstacle_detector_node."""
        if msg.data and not self.obstacle:
            self.obstacle = True
            self.obstacle_stops += 1
            self.get_logger().error(
                f'¡OBSTÁCULO! Parada de emergencia #{self.obstacle_stops}')
            self.stop()
        elif not msg.data:
            self.obstacle = False

    def send_direct(self, throttle: float, steering: float):
        """Manda throttle/steering directo al ESP32 — bypasea el PID."""
        msg = Vector3Stamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.vector.x        = float(throttle)
        msg.vector.y        = float(steering)
        self.pub_cmd.publish(msg)

    def send_pid(self, velocity: float, steering: float):
        """Manda setpoint al PID — usa para fase de tuning."""
        v = Float32(); v.data = float(velocity); self.pub_vel.publish(v)
        s = Float32(); s.data = float(steering); self.pub_str.publish(s)

    def stop(self):
        # Solo throttle a 0 — steering se mantiene en STEERING
        # para que el carro no cambie de trayectoria entre pruebas
        self.send_direct(0.0, STEERING)
        v = Float32(); v.data = 0.0; self.pub_vel.publish(v)
        s = Float32(); s.data = float(STEERING); self.pub_str.publish(s)

    def measure_direct(self, throttle: float) -> dict:
        """
        FASE 1 — Manda throttle fijo y mide velocidad real.
        Para inmediatamente si el LiDAR detecta obstáculo.
        """
        self.obstacle = False   # reset al inicio de cada punto

        # Settle
        t0 = time.monotonic()
        while time.monotonic() - t0 < LUT_SETTLE_S:
            if self.obstacle:
                self.stop()
                return {'throttle': throttle, 'mean': 0.0,
                        'std': 0.0, 'aborted': True}
            self.send_direct(throttle, STEERING)
            time.sleep(0.02)

        # Medir
        samples = []
        t1 = time.monotonic()
        while time.monotonic() - t1 < LUT_MEASURE_S:
            if self.obstacle:
                self.stop()
                # Si ya tenemos muestras suficientes, usar las que hay
                if len(samples) > 20:
                    break
                return {'throttle': throttle, 'mean': 0.0,
                        'std': 0.0, 'aborted': True}
            self.send_direct(throttle, STEERING)
            with self.lock:
                samples.append(self.speed)
            time.sleep(0.02)

        self.stop()

        if not samples:
            return {'throttle': throttle, 'mean': 0.0, 'std': 0.0}

        half  = samples[len(samples)//2:]
        mean2 = sum(half) / len(half)
        std   = math.sqrt(sum((v - mean2)**2 for v in samples) / len(samples))

        return {
            'throttle': throttle,
            'mean':     round(mean2, 4),
            'std':      round(std,   4),
            'n':        len(samples),
        }

    def measure_pid(self, setpoint: float) -> dict:
        """
        FASE 2 — Manda setpoint al PID y mide convergencia.
        Para inmediatamente si el LiDAR detecta obstáculo.
        """
        self.obstacle = False   # reset al inicio de cada prueba

        with self.lock:
            self.speed_hist.clear()
            self.conv_time  = None
            self.sp_current = setpoint
            self.measuring  = True

        # Settle
        t0 = time.monotonic()
        while time.monotonic() - t0 < TUNE_SETTLE_S:
            if self.obstacle:
                with self.lock: self.measuring = False
                self.stop()
                return {'error': 'obstáculo', 'setpoint': setpoint}
            self.send_pid(setpoint, STEERING)
            time.sleep(0.02)

        # Medir
        t1 = time.monotonic()
        while time.monotonic() - t1 < TUNE_MEASURE_S:
            if self.obstacle:
                with self.lock: self.measuring = False
                self.stop()
                # Usar muestras parciales si hay suficientes
                with self.lock:
                    samples = [s[1] for s in self.speed_hist]
                if len(samples) < 20:
                    return {'error': 'obstáculo', 'setpoint': setpoint}
                break
            self.send_pid(setpoint, STEERING)
            time.sleep(0.02)

        with self.lock:
            self.measuring = False
            samples   = [s[1] for s in self.speed_hist]
            conv_time = self.conv_time

        self.stop()

        if not samples:
            return {'error': 'sin datos', 'setpoint': setpoint}

        half      = samples[len(samples)//2:]
        mean      = sum(half) / len(half)
        std       = math.sqrt(sum((v - mean)**2 for v in half) / len(half))
        mean_err  = setpoint - mean
        max_v     = max(samples)
        overshoot = max(0.0, max_v - setpoint)
        pct_ok    = sum(1 for v in half
                        if abs(v - setpoint) < 0.08) / len(half) * 100
        conv_s    = (conv_time - (time.monotonic() - TUNE_SETTLE_S - TUNE_MEASURE_S)
                     if conv_time else None)

        return {
            'setpoint':       setpoint,
            'mean':           round(mean,      4),
            'std':            round(std,       4),
            'mean_error':     round(mean_err,  4),
            'overshoot':      round(overshoot, 4),
            'pct_within_8cm': round(pct_ok,    1),
            'conv_time_s':    round(conv_s, 2) if conv_s else 'no convergió',
        }


def run_pid_node(gains: dict, lut_points: list) -> object:
    """Lanza velocity_pid_node con gains dados."""
    import subprocess
    cmd = [
        'ros2', 'run', 'neuracar_perception', 'velocity_pid_node',
        '--ros-args',
        '-p', f'kp:={gains["kp"]}',
        '-p', f'ki:={gains["ki"]}',
        '-p', f'kd:={gains["kd"]}',
        '-p', f'max_integral:={gains["max_integral"]}',
        '-p', f'max_throttle:={MAX_THROTTLE}',
    ]
    return subprocess.Popen(cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)



def save_results(txt_path, csv_path, lut_results, tune_results,
                 obstacle_stops, partial=False):
    """Guarda reporte y CSV. Llamado tras cada set de gains y al terminar."""
    jump_lo = jump_hi = jump_thr_lo = jump_thr_hi = None
    for i in range(len(lut_results) - 1):
        if lut_results[i+1].get('mean', 0) - lut_results[i].get('mean', 0) > 0.4:
            jump_lo     = lut_results[i]['mean']
            jump_hi     = lut_results[i+1]['mean']
            jump_thr_lo = lut_results[i]['throttle']
            jump_thr_hi = lut_results[i+1]['throttle']

    with open(txt_path, 'w') as f:
        estado = '(PARCIAL)' if partial else '(COMPLETO)'
        f.write(f'NEURACAR PID TUNER EN PISTA {estado}\n')
        f.write(f'Fecha  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Batería: NiMH 7S 8.4V 3000mAh  Steering: {STEERING}\n')
        f.write(f'Paradas por obstáculo LiDAR: {obstacle_stops}\n')
        f.write('='*62 + '\n\n')

        f.write('LUT CALIBRADA EN PISTA\n')
        f.write('-'*62 + '\n')
        valid_lut = [r for r in lut_results
                     if r.get('mean', 0) > 0.05 and not r.get('aborted')]
        if valid_lut:
            f.write('_LUT = [\n')
            f.write('    (0.00, 0.000),   # motor parado\n')
            for r in valid_lut:
                note = '  # salto' if (jump_hi and
                        abs(r['mean'] - jump_hi) < 0.05) else ''
                f.write(f'    ({r["mean"]:.3f}, {r["throttle"]:.3f}),{note}\n')
            f.write(']\n\n')
            if jump_lo:
                f.write(f'# ZONA NO USABLE: {jump_lo:.2f} - {jump_hi:.2f} m/s\n')
                f.write(f'# salto entre throttle {jump_thr_lo} y {jump_thr_hi}\n\n')
        else:
            f.write('  Sin datos LUT\n\n')

        valid = [r for r in tune_results if 'error' not in r]
        if valid:
            from collections import defaultdict
            scores = defaultdict(list)
            for r in valid:
                key = (r['kp'], r['ki'], r['kd'], r['max_integral'])
                scores[key].append(r['pct_within_8cm'])
            ranked = sorted(scores.items(),
                            key=lambda x: sum(x[1])/len(x[1]), reverse=True)

            f.write('RANKING GAINS (% dentro de ±8cm/s)\n')
            f.write('-'*62 + '\n')
            for rank, (key, sc) in enumerate(ranked, 1):
                kp, ki, kd, mi = key
                f.write(f'  #{rank}: kp={kp} ki={ki} kd={kd} ')
                f.write(f'max_int={mi} → {sum(sc)/len(sc):.1f}%\n')

            f.write('\nMEJOR GAIN POR VELOCIDAD\n')
            f.write('-'*62 + '\n')
            for sp in TUNE_SETPOINTS:
                sp_r = [r for r in valid if r['setpoint'] == sp]
                if not sp_r: continue
                best = min(sp_r, key=lambda r: abs(r['mean_error']))
                f.write(f'  {sp:.1f} m/s → kp={best["kp"]} ki={best["ki"]} ')
                f.write(f'kd={best["kd"]} ')
                f.write(f'(err={best["mean_error"]:+.3f} ')
                f.write(f'ok={best["pct_within_8cm"]:.0f}%)\n')

            f.write('\nGAIN SCHEDULE SUGERIDO\n')
            f.write('-'*62 + '\n')
            f.write('_GAIN_SCHEDULE = [\n')
            f.write('    # v_m/s   kp     ki     kd     max_int\n')
            for sp in TUNE_SETPOINTS:
                sp_r = [r for r in valid if r['setpoint'] == sp]
                if not sp_r: continue
                best = min(sp_r, key=lambda r: abs(r['mean_error']))
                f.write(f'    ({sp:.1f},   {best["kp"]:.3f},  ')
                f.write(f'{best["ki"]:.3f},  {best["kd"]:.3f},   ')
                f.write(f'{best["max_integral"]:.3f}),\n')
            f.write(']\n')
        else:
            f.write('Sin datos de gains aún\n')

    if valid_lut := [r for r in tune_results if 'error' not in r]:
        fields = ['kp', 'ki', 'kd', 'max_integral', 'setpoint',
                  'mean', 'std', 'mean_error', 'overshoot',
                  'pct_within_8cm', 'conv_time_s']
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader(); w.writerows(valid_lut)


def main():
    rclpy.init()
    node = PistaTunerNode()
    spin_t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_t.start()

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir  = os.path.expanduser('~/Workspaces/Neuracar')
    txt_path = os.path.join(out_dir, f'pid_pista5_{ts}.txt')
    csv_path = os.path.join(out_dir, f'pid_pista5_{ts}.csv')
    lut_results  = []
    tune_results = []
    pid_proc     = None

    print('\n' + '='*62)
    print('  NEURACAR PID TUNER EN PISTA')
    print('='*62)
    print(f'  Steering fijo: {STEERING}  (círculo)')
    t1 = len(LUT_THROTTLES)*(LUT_SETTLE_S+LUT_MEASURE_S+PAUSE_S)/60
    t2 = len(GAIN_SETS)*len(TUNE_SETPOINTS)*(TUNE_SETTLE_S+TUNE_MEASURE_S+PAUSE_S)/60
    print(f'  Fase 1 LUT: {len(LUT_THROTTLES)} puntos ≈ {t1:.1f} min')
    print(f'  Fase 2 Gains: {len(GAIN_SETS)} sets × {len(TUNE_SETPOINTS)} vel ≈ {t2:.1f} min')
    print(f'  Total estimado: {t1+t2:.1f} minutos')
    print('='*62)
    print('\nAsegúrate de:')
    print('  1. Sensors bridge + obstacle_detector_node corriendo')
    print('  2. velocity_pid_node NO corriendo')
    print('  3. Espacio libre para círculos')
    print('  4. Batería NiMH cargada')
    print('  Si hay obstáculo: para y espera hasta que se despeje.')
    print('  Ctrl+C: guarda resultados parciales y termina.')
    print('\nIniciando en 8 segundos...\n')

    try:
        time.sleep(8)
    except KeyboardInterrupt:
        try: node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass
        return

    def wait_clear(label=''):
        """Espera hasta que no haya obstáculo."""
        if node.obstacle:
            print(f'  ⏸  {label}Obstáculo — esperando despeje...',
                  end='', flush=True)
            while node.obstacle:
                node.stop(); time.sleep(0.2)
            print(' reanudando.')
            time.sleep(1.5)

    # ── FASE 1 ────────────────────────────────────────────────────
    print('\n' + '─'*62)
    print('  FASE 1: Calibración LUT')
    print('─'*62)

    try:
        for thr in LUT_THROTTLES:
            wait_clear()
            print(f'  throttle={thr:.3f} ... ', end='', flush=True)
            r = node.measure_direct(thr)

            if r.get('aborted'):
                # Obstáculo durante medición — esperar y repetir
                wait_clear('tras obstáculo — ')
                r = node.measure_direct(thr)

            lut_results.append(r)
            print(f'real={r["mean"]:.3f} m/s  std={r["std"]:.3f}')
            time.sleep(PAUSE_S)

    except KeyboardInterrupt:
        print('\nCtrl+C en fase 1 — guardando parcial...')
        node.stop()
        save_results(txt_path, csv_path, lut_results, tune_results,
                     node.obstacle_stops, partial=True)
        print(f'Guardado: {txt_path}')
        try: node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass
        return

    # Guardado tras fase 1
    save_results(txt_path, csv_path, lut_results, tune_results,
                 node.obstacle_stops, partial=True)
    print(f'\n  Fase 1 OK — guardado: {txt_path}')

    # Detectar salto
    for i in range(len(lut_results)-1):
        dv = lut_results[i+1].get('mean',0) - lut_results[i].get('mean',0)
        if dv > 0.4:
            print(f'  ⚠ Salto: {lut_results[i]["mean"]:.2f}→')
            print(f'    {lut_results[i+1]["mean"]:.2f} m/s ')
            print(f'    (throttle {lut_results[i]["throttle"]}→')
            print(f'    {lut_results[i+1]["throttle"]})')

    # ── FASE 2 ────────────────────────────────────────────────────
    print('\n' + '─'*62)
    print('  FASE 2: Tuning de gains')
    print('─'*62)

    total_tests = len(GAIN_SETS) * len(TUNE_SETPOINTS)
    done        = 0

    try:
        for gi, gains in enumerate(GAIN_SETS):
            print(f'\n  Gains {gi+1}/{len(GAIN_SETS)}: ')
            print(f'  kp={gains["kp"]} ki={gains["ki"]} ')
            print(f'  kd={gains["kd"]} max_int={gains["max_integral"]}')

            pid_proc = run_pid_node(gains, lut_results)
            time.sleep(2.0)

            for sp in TUNE_SETPOINTS:
                done += 1
                wait_clear()
                print(f'  [{done}/{total_tests}] sp={sp:.1f} m/s ... ',
                      end='', flush=True)

                r = node.measure_pid(sp)

                # Si hubo obstáculo, esperar y repetir una vez
                if r.get('error') == 'obstáculo':
                    wait_clear('tras obstáculo — ')
                    r = node.measure_pid(sp)

                r.update(gains)
                tune_results.append(r)

                if 'error' in r:
                    print(f'ERROR: {r["error"]}')
                else:
                    ok = ('✓' if abs(r['mean_error']) < 0.08
                          else '~' if abs(r['mean_error']) < 0.15
                          else '✗')
                    print(f'{ok} real={r["mean"]:.3f} ')
                    print(f'    err={r["mean_error"]:+.3f} ')
                    print(f'    ok={r["pct_within_8cm"]:.0f}%')

                time.sleep(PAUSE_S)

            pid_proc.terminate()
            pid_proc.wait(timeout=3)
            pid_proc = None
            time.sleep(1.0)

            # Guardar tras cada set de gains completo
            save_results(txt_path, csv_path, lut_results, tune_results,
                         node.obstacle_stops, partial=True)
            print(f'  💾 Guardado ({gi+1}/{len(GAIN_SETS)} gains)')

    except KeyboardInterrupt:
        print('\nCtrl+C — guardando...')
        node.stop()
        if pid_proc:
            try: pid_proc.terminate(); pid_proc.wait(timeout=3)
            except Exception: pass

    # ── Guardado final ────────────────────────────────────────────
    node.stop()
    save_results(txt_path, csv_path, lut_results, tune_results,
                 node.obstacle_stops, partial=False)

    print(f'\nReporte: {txt_path}')
    print(f'CSV    : {csv_path}')
    print(f'Paradas por obstáculo: {node.obstacle_stops}')
    print('\n¡Tuning completado!')
    print('Copia _LUT y _GAIN_SCHEDULE del reporte a velocity_pid_node.py')

    try: node.destroy_node()
    except Exception: pass
    try: rclpy.shutdown()
    except Exception: pass


if __name__ == '__main__':
    main()