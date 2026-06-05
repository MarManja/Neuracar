#!/usr/bin/env python3
"""
pid_dashboard.py — Neuracar PID Tuning Dashboard
==================================================
Dashboard mínimo exclusivo para validar el PID de velocidad.
Sin cámara, sin lane, sin LiDAR — solo lo que necesitas para tuning.

Muestra en tiempo real:
  - Setpoint vs velocidad real (gráfica)
  - Throttle calculado por el PID (gráfica)
  - Error instantáneo con barra visual
  - Estado del lazo: INACTIVO / CONVERGIDO / AJUSTANDO / ERROR ALTO
  - Batería: voltaje y SoC

Topics:
  /neuracar/cmd_velocity  (std_msgs/Float32) — setpoint m/s
  /neuracar/wheel_speed   (std_msgs/Float32) — velocidad real encoder
  /neuracar/user_command  (geometry_msgs/Vector3Stamped) — throttle → ESP32
  /neuracar/battery       (sensor_msgs/BatteryState)

Uso:
  ros2 run neuracar_dashboard pid_dashboard
"""

import sys
import math
import threading
from collections import deque

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Float32

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QGridLayout, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

# ── Paleta ────────────────────────────────────────────────────────────
BG     = '#0D1117'
CARD   = '#161B22'
BORDER = '#30363D'
GREEN  = '#3FB950'
YELLOW = '#E3B341'
RED    = '#FF4444'
CYAN   = '#00E5CC'
WHITE  = '#E6EDF3'
DIM    = '#8B949E'
BLUE   = '#4C9EF0'
ORANGE = '#FF8C00'

HISTORY_S = 20    # segundos visibles en la gráfica
HISTORY_N = 1000  # puntos máximos (alimentado a 50Hz → 20s)


# ── ROS Backend ───────────────────────────────────────────────────────
class PIDBackend(Node):

    def __init__(self):
        super().__init__('pid_dashboard')

        # Estado PID
        self.lock         = threading.Lock()
        self.sp           = 0.0    # setpoint m/s
        self.real         = 0.0    # velocidad real m/s
        self.throttle     = 0.0    # throttle calculado [-1,1]
        self.error        = 0.0    # sp - real
        self.sp_t         = 0.0    # timestamp último setpoint
        self.real_t       = 0.0    # timestamp última medición

        # Batería
        self.bat_v        = 0.0
        self.bat_soc      = 0.0
        self.bat_present  = False

        # Historial para gráficas — alimentado desde wheel_speed @ 50Hz
        self.t_start      = None
        self.h_t          = deque(maxlen=HISTORY_N)
        self.h_sp         = deque(maxlen=HISTORY_N)
        self.h_real       = deque(maxlen=HISTORY_N)
        self.h_thr        = deque(maxlen=HISTORY_N)
        self.h_err        = deque(maxlen=HISTORY_N)

        # Suscripciones
        self.create_subscription(
            Float32, '/neuracar/cmd_velocity', self._sp_cb, 10)
        self.create_subscription(
            Float32, '/neuracar/wheel_speed', self._real_cb, 10)
        self.create_subscription(
            Vector3Stamped, '/neuracar/user_command', self._cmd_cb, 10)
        self.create_subscription(
            BatteryState, '/neuracar/battery', self._bat_cb, 10)

        self.get_logger().info('PID Dashboard backend iniciado')

    def _sp_cb(self, msg: Float32):
        with self.lock:
            self.sp    = float(msg.data)
            self.sp_t  = self.get_clock().now().nanoseconds / 1e9
            self.error = self.sp - self.real

    def _real_cb(self, msg: Float32):
        now = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            self.real   = float(msg.data)
            self.real_t = now
            self.error  = self.sp - self.real

            # Iniciar timer relativo al primer dato
            if self.t_start is None:
                self.t_start = now
            t = now - self.t_start

            # Agregar al historial — esto alimenta las gráficas
            self.h_t.append(t)
            self.h_sp.append(self.sp)
            self.h_real.append(self.real)
            self.h_thr.append(self.throttle)
            self.h_err.append(self.error)

    def _cmd_cb(self, msg: Vector3Stamped):
        with self.lock:
            self.throttle = float(msg.vector.x)

    def _bat_cb(self, msg: BatteryState):
        with self.lock:
            self.bat_v       = float(msg.voltage)
            self.bat_soc     = float(msg.percentage)
            self.bat_present = bool(msg.present)


# ── Dashboard GUI ─────────────────────────────────────────────────────
class PIDDashboard(QMainWindow):

    def __init__(self, node: PIDBackend):
        super().__init__()
        self.node = node
        self.setWindowTitle('Neuracar — PID Velocity Tuning')
        self.setMinimumSize(1100, 700)
        self._apply_style()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Columna izquierda: indicadores numéricos ──────────────────
        left = QWidget()
        left.setFixedWidth(280)
        lv = QVBoxLayout(left)
        lv.setSpacing(6)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self._build_pid_status())
        lv.addWidget(self._build_battery())
        lv.addWidget(self._build_tips())
        lv.addStretch()
        root.addWidget(left)

        # ── Columna derecha: gráficas ─────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(6)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self._build_plots())
        root.addWidget(right, 1)

        # Timer de refresco — independiente de los callbacks ROS
        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(50)   # 20 FPS — suficiente para ver el PID

    # ── Estilo ────────────────────────────────────────────────────────
    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background:{BG}; color:{WHITE};
                font-family:'DejaVu Sans'; font-size:11px; }}
            QGroupBox {{
                border:1px solid {BORDER}; border-radius:8px;
                margin-top:12px; padding:6px; padding-top:16px;
                font-weight:bold; font-size:10px; color:{DIM}; }}
            QGroupBox::title {{
                subcontrol-origin:margin; left:10px; color:{CYAN}; }}
            QProgressBar {{
                border:1px solid {BORDER}; border-radius:4px;
                background:{CARD}; text-align:center;
                font-size:11px; color:{WHITE}; }}
            QProgressBar::chunk {{ border-radius:4px; }}
        """)

    def _lbl(self, text, color=WHITE, size=11, bold=False):
        l = QLabel(text)
        l.setStyleSheet(
            f'color:{color};font-size:{size}px;'
            f'font-weight:{"bold" if bold else "normal"};')
        return l

    # ── Panel: Estado PID ─────────────────────────────────────────────
    def _build_pid_status(self):
        grp = QGroupBox('PID Velocidad')
        g   = QGridLayout(grp)
        g.setSpacing(8)
        g.setContentsMargins(8, 8, 8, 8)

        # Estado general
        g.addWidget(self._lbl('Estado:', DIM), 0, 0)
        self.l_state = self._lbl('INACTIVO', DIM, 14, True)
        g.addWidget(self.l_state, 0, 1)

        # Setpoint
        g.addWidget(self._lbl('Setpoint:', DIM), 1, 0)
        self.l_sp = self._lbl('0.000 m/s', GREEN, 16, True)
        g.addWidget(self.l_sp, 1, 1)

        # Real
        g.addWidget(self._lbl('Real:', DIM), 2, 0)
        self.l_real = self._lbl('0.000 m/s', BLUE, 16, True)
        g.addWidget(self.l_real, 2, 1)

        # Error
        g.addWidget(self._lbl('Error:', DIM), 3, 0)
        self.l_err = self._lbl('0.000 m/s', WHITE, 14, True)
        g.addWidget(self.l_err, 3, 1)

        # Throttle al ESP32
        g.addWidget(self._lbl('Throttle:', DIM), 4, 0)
        self.l_thr = self._lbl('0.000', ORANGE, 14, True)
        g.addWidget(self.l_thr, 4, 1)

        # Separador
        sep = QLabel(); sep.setFixedHeight(1)
        sep.setStyleSheet(f'background:{BORDER};')
        g.addWidget(sep, 5, 0, 1, 2)

        # Barra de seguimiento
        g.addWidget(self._lbl('Seguimiento:', DIM, 10), 6, 0, 1, 2)
        self.bar_follow = QProgressBar()
        self.bar_follow.setRange(0, 100)
        self.bar_follow.setValue(0)
        self.bar_follow.setFixedHeight(22)
        self.bar_follow.setFormat('%v%')
        g.addWidget(self.bar_follow, 7, 0, 1, 2)

        # Barra de error centrada
        g.addWidget(self._lbl('Error  ← 0 →  (+/- 0.3 m/s)', DIM, 9), 8, 0, 1, 2)
        self.bar_err = QLabel()
        self.bar_err.setFixedHeight(18)
        self.bar_err.setStyleSheet(
            f'background:{CARD};border:1px solid {BORDER};border-radius:4px;')
        g.addWidget(self.bar_err, 9, 0, 1, 2)

        return grp

    # ── Panel: Batería ────────────────────────────────────────────────
    def _build_battery(self):
        grp = QGroupBox('Batería NiMH')
        g   = QGridLayout(grp)
        g.setSpacing(6)

        g.addWidget(self._lbl('Voltaje:', DIM), 0, 0)
        self.l_bat_v = self._lbl('— V', WHITE, 13, True)
        g.addWidget(self.l_bat_v, 0, 1)

        g.addWidget(self._lbl('SoC:', DIM), 1, 0, 1, 2)
        self.bar_soc = QProgressBar()
        self.bar_soc.setRange(0, 100)
        self.bar_soc.setValue(0)
        self.bar_soc.setFixedHeight(20)
        self.bar_soc.setFormat('SoC: %v%')
        g.addWidget(self.bar_soc, 2, 0, 1, 2)

        return grp

    # ── Panel: Tips de tuning ─────────────────────────────────────────
    def _build_tips(self):
        grp = QGroupBox('Guía de tuning')
        v   = QVBoxLayout(grp)
        v.setSpacing(4)

        tips = [
            ('🟢 Convergido', '< 3 cm/s error', GREEN),
            ('🟡 Ajustando',  '< 8 cm/s error', YELLOW),
            ('🔴 Error alto', '≥ 8 cm/s error', RED),
            ('', '', DIM),
            ('Oscila →',     'bajar kp', DIM),
            ('Lento →',      'subir kp', DIM),
            ('Error fijo →', 'subir ki',  DIM),
            ('Overshoot →',  'subir kd',  DIM),
        ]
        for t1, t2, c in tips:
            if not t1:
                sep = QLabel(); sep.setFixedHeight(1)
                sep.setStyleSheet(f'background:{BORDER};margin:2px 0;')
                v.addWidget(sep)
                continue
            row = QWidget()
            rl  = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)
            l1 = QLabel(t1); l1.setStyleSheet(f'color:{c};font-size:10px;font-weight:bold;')
            l2 = QLabel(t2); l2.setStyleSheet(f'color:{DIM};font-size:10px;')
            rl.addWidget(l1); rl.addWidget(l2); rl.addStretch()
            v.addWidget(row)

        return grp

    # ── Gráficas pyqtgraph ────────────────────────────────────────────
    def _build_plots(self):
        grp = QGroupBox('Respuesta en tiempo real')
        v   = QVBoxLayout(grp)
        v.setContentsMargins(4, 16, 4, 4)

        pw = pg.GraphicsLayoutWidget()
        pw.setBackground(CARD)
        v.addWidget(pw)

        # ── Gráfica 1: Setpoint vs Real ───────────────────────────────
        self.p1 = pw.addPlot(row=0, col=0)
        self.p1.setTitle(
            '<span style="color:#8B949E;font-size:9pt">'
            'Velocidad: <span style="color:#3FB950">■ Setpoint</span> '
            '<span style="color:#4C9EF0">■ Real</span></span>')
        self.p1.setLabel('left', 'm/s', color=DIM, size='8pt')
        self.p1.showGrid(x=True, y=True, alpha=0.15)
        self.p1.addLine(y=0, pen=pg.mkPen(BORDER, width=1,
                                           style=Qt.DashLine))
        self.p1.setMinimumHeight(200)

        self.c_sp   = self.p1.plot(
            pen=pg.mkPen(GREEN,  width=2.5), name='Setpoint')
        self.c_real = self.p1.plot(
            pen=pg.mkPen(BLUE,   width=2.5), name='Real')

        # Región de tolerancia ±3cm/s alrededor del setpoint
        self.tol_region = pg.LinearRegionItem(
            values=[-0.03, 0.03],
            orientation='horizontal',
            brush=pg.mkBrush(GREEN + '18'),
            pen=pg.mkPen(GREEN + '40', width=1),
            movable=False)
        # La región se actualiza en _refresh según el setpoint actual

        # ── Gráfica 2: Throttle ───────────────────────────────────────
        self.p2 = pw.addPlot(row=1, col=0)
        self.p2.setTitle(
            '<span style="color:#8B949E;font-size:9pt">'
            '<span style="color:#FF8C00">■ Throttle → ESP32-A</span></span>')
        self.p2.setLabel('left', '[-1, 1]', color=DIM, size='8pt')
        self.p2.showGrid(x=True, y=True, alpha=0.15)
        self.p2.setYRange(-1.1, 1.1)
        self.p2.addLine(y=0, pen=pg.mkPen(BORDER, width=1,
                                           style=Qt.DashLine))
        self.p2.setMinimumHeight(140)

        self.c_thr = self.p2.plot(
            pen=pg.mkPen(ORANGE, width=2), name='Throttle')

        # ── Gráfica 3: Error ──────────────────────────────────────────
        self.p3 = pw.addPlot(row=2, col=0)
        self.p3.setTitle(
            '<span style="color:#8B949E;font-size:9pt">'
            '<span style="color:#E3B341">■ Error (setpoint − real)</span></span>')
        self.p3.setLabel('left', 'm/s', color=DIM, size='8pt')
        self.p3.showGrid(x=True, y=True, alpha=0.15)
        self.p3.setYRange(-0.35, 0.35)
        self.p3.addLine(y=0,     pen=pg.mkPen(BORDER, width=1,
                                               style=Qt.DashLine))
        self.p3.addLine(y= 0.03, pen=pg.mkPen(GREEN + '60', width=1,
                                               style=Qt.DashLine))
        self.p3.addLine(y=-0.03, pen=pg.mkPen(GREEN + '60', width=1,
                                               style=Qt.DashLine))
        self.p3.addLine(y= 0.08, pen=pg.mkPen(YELLOW + '60', width=1,
                                               style=Qt.DashLine))
        self.p3.addLine(y=-0.08, pen=pg.mkPen(YELLOW + '60', width=1,
                                               style=Qt.DashLine))
        self.p3.setMinimumHeight(120)

        self.c_err = self.p3.plot(
            pen=pg.mkPen(YELLOW, width=2), name='Error')

        # Enlazar ejes X para scroll sincronizado
        self.p2.setXLink(self.p1)
        self.p3.setXLink(self.p1)

        return grp

    # ── Refresco 20 FPS ───────────────────────────────────────────────
    def _refresh(self):
        n = self.node

        with n.lock:
            sp    = n.sp
            real  = n.real
            err   = n.error
            thr   = n.throttle
            t     = list(n.h_t)
            h_sp  = list(n.h_sp)
            h_r   = list(n.h_real)
            h_thr = list(n.h_thr)
            h_err = list(n.h_err)
            bat_v = n.bat_v
            bat_s = n.bat_soc
            bat_p = n.bat_present

        # ── Gráficas ──────────────────────────────────────────────────
        if len(t) >= 2:
            self.c_sp.setData(t,   h_sp)
            self.c_real.setData(t, h_r)
            self.c_thr.setData(t,  h_thr)
            self.c_err.setData(t,  h_err)

            # Scroll: mostrar últimos HISTORY_S segundos
            te = t[-1]
            x0 = max(0.0, te - HISTORY_S)
            self.p1.setXRange(x0, te + 0.2, padding=0)

        # ── Indicadores numéricos ─────────────────────────────────────

        # Estado del lazo
        if abs(sp) < 0.02:
            st, sc = 'INACTIVO',    DIM
        elif abs(err) < 0.03:
            st, sc = '✓ CONVERGIDO', GREEN
        elif abs(err) < 0.08:
            st, sc = '~ AJUSTANDO',  YELLOW
        else:
            st, sc = '✗ ERROR ALTO', RED

        self.l_state.setText(st)
        self.l_state.setStyleSheet(
            f'color:{sc};font-size:14px;font-weight:bold;')

        # Setpoint
        self.l_sp.setText(f'{sp:+.3f} m/s')

        # Real — color según error
        if abs(sp) < 0.02:
            rc = DIM
        elif abs(err) < 0.03:
            rc = GREEN
        elif abs(err) < 0.08:
            rc = YELLOW
        else:
            rc = RED
        self.l_real.setText(f'{real:+.3f} m/s')
        self.l_real.setStyleSheet(
            f'color:{rc};font-size:16px;font-weight:bold;')

        # Error
        ec = GREEN if abs(err) < 0.03 else YELLOW if abs(err) < 0.08 else RED
        self.l_err.setText(f'{err:+.3f} m/s')
        self.l_err.setStyleSheet(
            f'color:{ec};font-size:14px;font-weight:bold;')

        # Throttle
        self.l_thr.setText(f'{thr:+.3f}')

        # Barra de seguimiento 0-100%
        if abs(sp) > 0.02:
            pct = max(0, int((1.0 - min(abs(err) / 0.3, 1.0)) * 100))
        else:
            pct = 0
        self.bar_follow.setValue(pct)
        fc = GREEN if pct > 85 else YELLOW if pct > 60 else RED
        self.bar_follow.setStyleSheet(
            f'QProgressBar::chunk{{background:{fc};border-radius:4px;}}')

        # Barra de error centrada [-0.3, +0.3]
        bw  = max(self.bar_err.width(), 2)
        ec2 = max(-0.3, min(0.3, err))
        mid = bw // 2
        pos = int(mid + (ec2 / 0.3) * mid)
        pos = max(1, min(pos, bw - 1))
        x1  = min(mid, pos)
        x2  = max(mid, pos)
        bc  = GREEN if abs(err) < 0.03 else YELLOW if abs(err) < 0.08 else RED
        if x1 == x2:
            self.bar_err.setStyleSheet(
                f'background:{CARD};border:1px solid {BORDER};border-radius:4px;')
        else:
            self.bar_err.setStyleSheet(
                f'background:qlineargradient(x1:0,x2:1,'
                f'stop:{x1/bw:.3f} {CARD},'
                f'stop:{x1/bw:.3f} {bc},'
                f'stop:{x2/bw:.3f} {bc},'
                f'stop:{x2/bw:.3f} {CARD});'
                f'border:1px solid {BORDER};border-radius:4px;')

        # ── Batería ───────────────────────────────────────────────────
        if bat_p and bat_v > 1.0:
            bvc = RED if bat_v < 5.5 else YELLOW if bat_v < 6.5 else GREEN
            self.l_bat_v.setText(f'{bat_v:.2f} V')
            self.l_bat_v.setStyleSheet(
                f'color:{bvc};font-size:13px;font-weight:bold;')
            soc_pct = int(bat_s * 100)
            self.bar_soc.setValue(soc_pct)
            sc2 = RED if soc_pct < 20 else YELLOW if soc_pct < 40 else GREEN
            self.bar_soc.setStyleSheet(
                f'QProgressBar::chunk{{background:{sc2};border-radius:4px;}}')
        else:
            self.l_bat_v.setText('N/A')
            self.bar_soc.setValue(0)


# ── main ──────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = PIDBackend()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    app = QApplication(sys.argv)
    win = PIDDashboard(node)
    win.show()

    try:
        app.exec_()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()