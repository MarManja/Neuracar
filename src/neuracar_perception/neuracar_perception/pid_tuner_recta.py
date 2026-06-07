#!/usr/bin/env python3
"""
pid_tuner_recta.py — Neuracar
================================
Calibra los gains kp/ki/kd del velocity_pid_node en línea recta.

ESTRATEGIA — barrido secuencial en 3 fases (no grid completo):
  Fase 1: Encuentra el mejor kp  (ki=0, kd=0) — 6 pruebas
  Fase 2: Con ese kp, encuentra el mejor ki   — 6 pruebas
  Fase 3: Con kp+ki, afina kd                 — 4 pruebas
  Total: ~16 pruebas × ~25s = ~7 minutos

Por qué secuencial y no grid:
  Un grid de 6kp×6ki×4kd = 144 pruebas.
  El barrido secuencial aprovecha que kp, ki, kd tienen roles
  independientes: kp → velocidad de respuesta,
  ki → elimina error residual, kd → amortigua oscilación.
  Con 16 pruebas bien elegidas encuentra valores muy cercanos al óptimo.

MODO DE OPERACIÓN (igual que calibrar_velocidad_recta.py):
  - Avisa antes de cada prueba con cuenta regresiva de 3 segundos
  - El carro corre RUN_S segundos en línea recta
  - Para y da TURN_S segundos para girar el carro
  - Ctrl+C en cualquier momento — guarda lo que hay

USO:
  # Terminal 1 — sensores corriendo (sin velocity_pid_node — este lo lanza)
  ros2 launch neuracar_bringup sensors.launch.py camera:=false

  # Terminal 2
  python3 pid_tuner_recta.py

CONFIGURACIÓN: edita la sección CONFIG antes de correr.
"""

import csv
import math
import os
import subprocess
import threading
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32

# ══════════════════════════════════════════════════════════════════
#  CONFIG — edita aquí
# ══════════════════════════════════════════════════════════════════

# Velocidades de setpoint a probar en cada prueba de gains
# Usa solo velocidades estables (fuera de la zona del salto)
SETPOINTS = [0.82, 1.10, 1.38]   # m/s

# ── Fase 1: barrido de kp (ki=0, kd=0) ───────────────────────────
KP_VALUES = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]

# ── Fase 2: barrido de ki (con mejor kp de fase 1) ───────────────
KI_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# ── Fase 3: barrido de kd (con mejor kp+ki de fase 2) ────────────
KD_VALUES = [0.00, 0.01, 0.02, 0.05]

# Tiempo que corre el motor por prueba (segundos)
# A 1.4 m/s en 3s recorre ~4.2m — ajusta si tu espacio es menor
RUN_S    = 3.0

# Tiempo de settle antes de medir (dentro del RUN_S)
# La segunda mitad del run se usa para medir (más estabilizado)
SETTLE_F = 0.5   # fracción del run usada para medir (última mitad)

# Tiempo para girar el carro entre pruebas
TURN_S   = 15.0

# Pausa inicial antes de la primera prueba
READY_S  = 10.0

# Steering recto
STEERING = 0.0

# Parámetros fijos del PID durante el tuning
MAX_THROTTLE = 1.0
ALPHA        = 0.3
MAX_RATE     = 2.0
V_DEADBAND   = 0.05
MAX_INTEGRAL = 0.20   # límite anti-windup

# Umbral de "convergido" — error < X m/s se considera bueno
GOOD_ERR_MS  = 0.08   # 8 cm/s

# ══════════════════════════════════════════════════════════════════


class TunerNode(Node):

    def __init__(self):
        super().__init__('pid_tuner_recta')

        # Publisher directo al ESP32 (bypasea el PID para parar limpio)
        self.pub_stop = self.create_publisher(
            Vector3Stamped, '/neuracar/user_command', 10)

        # Publishers para el PID
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.pub_vel = self.create_publisher(
            Float32, '/neuracar/cmd_velocity', qos)
        self.pub_str = self.create_publisher(
            Float32, '/neuracar/cmd_steering', qos)

        # Estado
        self.lock  = threading.Lock()
        self.speed = 0.0

        self.create_subscription(
            Float32, '/neuracar/wheel_speed', self._speed_cb, 10)

        self.get_logger().info('PID Tuner Recta iniciado')

    def _speed_cb(self, msg: Float32):
        with self.lock:
            self.speed = float(msg.data)

    def send_pid(self, velocity: float):
        v = Float32(); v.data = float(velocity); self.pub_vel.publish(v)
        s = Float32(); s.data = float(STEERING); self.pub_str.publish(s)

    def stop(self):
        """Para el carro limpiamente — neutro directo al ESP32."""
        msg = Vector3Stamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.vector.x = 0.0
        msg.vector.y = 0.0
        self.pub_stop.publish(msg)
        # También mandar 0 al PID por si está corriendo
        v = Float32(); v.data = 0.0; self.pub_vel.publish(v)
        s = Float32(); s.data = 0.0; self.pub_str.publish(s)

    def medir(self, setpoint: float) -> dict:
        """
        Envía setpoint al PID durante RUN_S segundos.
        Mide la velocidad real en la segunda mitad del run.
        Devuelve estadísticas de convergencia.
        """
        samples_all  = []
        samples_meas = []
        t0           = time.monotonic()
        meas_start   = RUN_S * (1.0 - SETTLE_F)
        conv_time    = None

        while time.monotonic() - t0 < RUN_S:
            self.send_pid(setpoint)
            elapsed = time.monotonic() - t0

            with self.lock:
                v = self.speed
            samples_all.append(v)

            if elapsed >= meas_start:
                samples_meas.append(v)
                # Detectar convergencia: error < GOOD_ERR_MS por primera vez
                if conv_time is None and abs(v - setpoint) < GOOD_ERR_MS:
                    conv_time = elapsed

            time.sleep(0.02)

        self.stop()

        data = samples_meas if samples_meas else samples_all
        if not data:
            return {'setpoint': setpoint, 'mean': 0.0, 'std': 0.0,
                    'error': 99.0, 'pct_ok': 0.0, 'conv_s': None}

        mean     = sum(data) / len(data)
        std      = math.sqrt(sum((v - mean)**2 for v in data) / len(data))
        err      = abs(setpoint - mean)
        pct_ok   = sum(1 for v in data
                       if abs(v - setpoint) < GOOD_ERR_MS) / len(data) * 100

        return {
            'setpoint': setpoint,
            'mean':     round(mean,   4),
            'std':      round(std,    4),
            'error':    round(err,    4),
            'pct_ok':   round(pct_ok, 1),
            'conv_s':   round(conv_time, 2) if conv_time else None,
        }


def launch_pid(kp, ki, kd) -> subprocess.Popen:
    """Lanza velocity_pid_node con gains dados."""
    cmd = [
        'ros2', 'run', 'neuracar_perception', 'velocity_pid_node',
        '--ros-args',
        '-p', f'gain_scheduling:=false',
        '-p', f'kp:={kp}',
        '-p', f'ki:={ki}',
        '-p', f'kd:={kd}',
        '-p', f'max_integral:={MAX_INTEGRAL}',
        '-p', f'max_throttle:={MAX_THROTTLE}',
        '-p', f'alpha:={ALPHA}',
        '-p', f'max_rate:={MAX_RATE}',
        '-p', f'v_deadband:={V_DEADBAND}',
    ]
    return subprocess.Popen(cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def kill_pid(proc):
    if proc:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass


def countdown(node: TunerNode, setpoint: float):
    """Cuenta regresiva de 3 segundos antes de arrancar."""
    print(f'  Baja el carro al suelo, apunta a espacio libre...')
    print(f'  Setpoint: {setpoint} m/s')
    for i in range(3, 0, -1):
        print(f'  {i}...', end=' ', flush=True)
        time.sleep(1.0)
    print('¡ARRANCANDO!')


def turn_pause(node: TunerNode):
    """Pausa con cuenta regresiva para girar el carro."""
    print(f'  Para y gira el carro ({TURN_S:.0f}s)')
    node.stop()
    t = time.monotonic()
    while time.monotonic() - t < TURN_S:
        rem = int(TURN_S - (time.monotonic() - t))
        print(f'  {rem}s...', end='\r', flush=True)
        time.sleep(1.0)
    print(' ' * 20, end='\r')
    print('  ¡Listo para siguiente!')


def score(results: list) -> float:
    """
    Puntuación de una combinación de gains.
    Combina pct_ok (peso mayor) y error medio.
    Mayor es mejor.
    """
    if not results:
        return -999.0
    pct  = sum(r['pct_ok'] for r in results) / len(results)
    err  = sum(r['error']  for r in results) / len(results)
    return pct - err * 100   # penaliza error medio en cm/s


def run_gains(node: TunerNode, kp, ki, kd,
              first: bool = False) -> list:
    """
    Prueba una combinación de gains en todos los SETPOINTS.
    Devuelve lista de resultados por setpoint.
    """
    proc = launch_pid(kp, ki, kd)
    time.sleep(1.5)   # esperar arranque del nodo

    results = []
    for idx, sp in enumerate(SETPOINTS):
        if not first or idx > 0:
            turn_pause(node)
        countdown(node, sp)
        r = node.medir(sp)
        results.append(r)
        ok = '✓' if r['error'] < GOOD_ERR_MS else \
             '~' if r['error'] < 0.15 else '✗'
        conv = f'{r["conv_s"]}s' if r['conv_s'] else 'no conv'
        print(f'  {ok} sp={sp:.2f} real={r["mean"]:.3f} '
              f'err={r["error"]:.3f} ok={r["pct_ok"]:.0f}% conv={conv}')

    kill_pid(proc)
    time.sleep(1.0)
    return results


def save_report(txt_path, csv_path, all_results, best_kp, best_ki, best_kd):
    with open(txt_path, 'w') as f:
        f.write('NEURACAR PID TUNER RECTA\n')
        f.write(f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Setpoints: {SETPOINTS}\n')
        f.write('='*58 + '\n\n')
        f.write('RESULTADO FINAL\n')
        f.write('-'*58 + '\n')
        f.write(f'  kp = {best_kp}\n')
        f.write(f'  ki = {best_ki}\n')
        f.write(f'  kd = {best_kd}\n\n')
        f.write('Copia a _GAIN_SCHEDULE en velocity_pid_node.py:\n')
        f.write('_GAIN_SCHEDULE = [\n')
        f.write('    # v_m/s   kp     ki     kd     max_int\n')
        for sp in SETPOINTS:
            f.write(f'    ({sp:.2f},  {best_kp:.3f},  '
                    f'{best_ki:.3f},  {best_kd:.3f},   '
                    f'{MAX_INTEGRAL:.3f}),\n')
        f.write(']\n\n')
        f.write('DETALLE POR FASE\n')
        f.write('-'*58 + '\n')
        for phase, gains, results_list in all_results:
            f.write(f'\n{phase}: kp={gains[0]} ki={gains[1]} kd={gains[2]}\n')
            for r in results_list:
                f.write(f'  sp={r["setpoint"]:.2f} → real={r["mean"]:.3f} '
                        f'err={r["error"]:.3f} ok={r["pct_ok"]:.0f}%\n')

    rows = []
    for phase, gains, results_list in all_results:
        for r in results_list:
            rows.append({
                'phase': phase,
                'kp': gains[0], 'ki': gains[1], 'kd': gains[2],
                **r
            })
    if rows:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)


def main():
    rclpy.init()
    node = TunerNode()
    spin_t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_t.start()

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir  = os.path.expanduser('~/Workspaces/Neuracar')
    txt_path = os.path.join(out_dir, f'pid_recta_{ts}.txt')
    csv_path = os.path.join(out_dir, f'pid_recta_{ts}.csv')

    n_pruebas = (len(KP_VALUES) + len(KI_VALUES) + len(KD_VALUES)) * len(SETPOINTS)
    t_total   = (len(KP_VALUES) + len(KI_VALUES) + len(KD_VALUES)) * \
                len(SETPOINTS) * (RUN_S + TURN_S) / 60

    print('\n' + '='*58)
    print('  NEURACAR PID TUNER EN RECTA')
    print('='*58)
    print(f'  Setpoints: {SETPOINTS} m/s')
    print(f'  Fase 1 kp: {KP_VALUES}')
    print(f'  Fase 2 ki: {KI_VALUES}')
    print(f'  Fase 3 kd: {KD_VALUES}')
    print(f'  {n_pruebas} pruebas × {RUN_S+TURN_S:.0f}s = ~{t_total:.0f} min')
    print('='*58)
    print('\nAsegúrate de:')
    print('  1. Sensors + perception launch corriendo')
    print('  2. velocity_pid_node NO corriendo (este lo lanza)')
    print('  3. Espacio libre de ~4m al frente')
    print('  4. Batería NiMH cargada')
    print(f'\nIniciando en {READY_S:.0f} segundos... (Ctrl+C para cancelar)\n')

    try:
        time.sleep(READY_S)
    except KeyboardInterrupt:
        node.stop(); node.destroy_node(); return

    all_results  = []   # para el reporte
    best_kp = KP_VALUES[len(KP_VALUES)//2]   # default por si se interrumpe
    best_ki = KI_VALUES[0]
    best_kd = KD_VALUES[0]
    pid_proc = None

    try:
        # ══════════════════════════════════════════════════════════
        #  FASE 1 — Barrido de kp (ki=0, kd=0)
        # ══════════════════════════════════════════════════════════
        print('\n' + '─'*58)
        print('  FASE 1: Barrido de kp  (ki=0, kd=0)')
        print('─'*58)

        phase1_scores = {}
        for gi, kp in enumerate(KP_VALUES):
            print(f'\n  kp={kp}  ki=0  kd=0  '
                  f'[{gi+1}/{len(KP_VALUES)}]')
            res = run_gains(node, kp=kp, ki=0.0, kd=0.0,
                            first=(gi == 0))
            sc  = score(res)
            phase1_scores[kp] = sc
            all_results.append((f'F1_kp{kp}', (kp, 0.0, 0.0), res))
            print(f'  → score={sc:.1f}')

        best_kp = max(phase1_scores, key=phase1_scores.get)
        print(f'\n  ✓ Mejor kp = {best_kp}  (score={phase1_scores[best_kp]:.1f})')

        # Guardar parcial
        save_report(txt_path, csv_path, all_results, best_kp, best_ki, best_kd)
        print(f'  💾 Guardado parcial: {txt_path}')

        # ══════════════════════════════════════════════════════════
        #  FASE 2 — Barrido de ki (con mejor kp)
        # ══════════════════════════════════════════════════════════
        print('\n' + '─'*58)
        print(f'  FASE 2: Barrido de ki  (kp={best_kp}, kd=0)')
        print('─'*58)

        phase2_scores = {}
        for gi, ki in enumerate(KI_VALUES):
            print(f'\n  kp={best_kp}  ki={ki}  kd=0  '
                  f'[{gi+1}/{len(KI_VALUES)}]')
            res = run_gains(node, kp=best_kp, ki=ki, kd=0.0)
            sc  = score(res)
            phase2_scores[ki] = sc
            all_results.append((f'F2_ki{ki}', (best_kp, ki, 0.0), res))
            print(f'  → score={sc:.1f}')

        best_ki = max(phase2_scores, key=phase2_scores.get)
        print(f'\n  ✓ Mejor ki = {best_ki}  (score={phase2_scores[best_ki]:.1f})')

        save_report(txt_path, csv_path, all_results, best_kp, best_ki, best_kd)
        print(f'  💾 Guardado parcial: {txt_path}')

        # ══════════════════════════════════════════════════════════
        #  FASE 3 — Barrido de kd (con mejor kp+ki)
        # ══════════════════════════════════════════════════════════
        print('\n' + '─'*58)
        print(f'  FASE 3: Barrido de kd  (kp={best_kp}, ki={best_ki})')
        print('─'*58)

        phase3_scores = {}
        for gi, kd in enumerate(KD_VALUES):
            print(f'\n  kp={best_kp}  ki={best_ki}  kd={kd}  '
                  f'[{gi+1}/{len(KD_VALUES)}]')
            res = run_gains(node, kp=best_kp, ki=best_ki, kd=kd)
            sc  = score(res)
            phase3_scores[kd] = sc
            all_results.append((f'F3_kd{kd}', (best_kp, best_ki, kd), res))
            print(f'  → score={sc:.1f}')

        best_kd = max(phase3_scores, key=phase3_scores.get)
        print(f'\n  ✓ Mejor kd = {best_kd}  (score={phase3_scores[best_kd]:.1f})')

    except KeyboardInterrupt:
        print('\n\nCtrl+C — guardando resultados parciales...')
        node.stop()

    # ── Guardado final ────────────────────────────────────────────
    node.stop()
    save_report(txt_path, csv_path, all_results, best_kp, best_ki, best_kd)

    print('\n' + '='*58)
    print('  RESULTADO FINAL')
    print('='*58)
    print(f'  kp = {best_kp}')
    print(f'  ki = {best_ki}')
    print(f'  kd = {best_kd}')
    print(f'\n  Reporte: {txt_path}')
    print(f'  CSV    : {csv_path}')
    print('\n  Copia a _GAIN_SCHEDULE en velocity_pid_node.py:')
    print('  _GAIN_SCHEDULE = [')
    print('      # v_m/s   kp     ki     kd     max_int')
    for sp in SETPOINTS:
        print(f'      ({sp:.2f},  {best_kp:.3f},  '
              f'{best_ki:.3f},  {best_kd:.3f},   {MAX_INTEGRAL:.3f}),')
    print('  ]')

    try: node.destroy_node()
    except Exception: pass
    try: rclpy.shutdown()
    except Exception: pass


if __name__ == '__main__':
    main()