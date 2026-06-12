"""
dashboard_node.py — NeuraCar
══════════════════════════════════════════════════════════════════
Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026

Real-time telemetry dashboard built with PyQt5 and pyqtgraph.
Displays reference vs measured trajectory, velocity PID response,
throttle, steering, lateral error, obstacle status, and odometry.
Runs independently from the autonomous stack — launch separately
and it will reconnect automatically when topics become available.

Requires: PyQt5, pyqtgraph
On Jetson without display: ssh -X devel-ds@ip and run from laptop
if ROS_DOMAIN_ID is shared.

Subscriptions:
  /neuracar/path_reference       nav_msgs/Path
  /neuracar/path_real            nav_msgs/Path
  /neuracar/odometry             nav_msgs/Odometry
  /neuracar/velocity             geometry_msgs/TwistStamped
  /neuracar/cmd_velocity         std_msgs/Float32
  /neuracar/wheel_speed          std_msgs/Float32
  /neuracar/user_command         geometry_msgs/Vector3Stamped
  /neuracar/cmd_steering         std_msgs/Float32
  /neuracar/lidar/obstacle_alert std_msgs/Bool
  /neuracar/status               std_msgs/String

Publications:
  None — visualization only.
══════════════════════════════════════════════════════════════════
"""
import sys
import math
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)

from geometry_msgs.msg import TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float32, String

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QGroupBox, QGridLayout,
    QSizePolicy, QSplitter, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import pyqtgraph as pg

# ── Paleta ────────────────────────────────────────────────────────────
BG     = '#0D1117'; CARD   = '#161B22'; BORDER = '#30363D'
GREEN  = '#3FB950'; YELLOW = '#E3B341'; RED    = '#FF4444'
CYAN   = '#00E5CC'; WHITE  = '#E6EDF3'; DIM    = '#8B949E'
BLUE   = '#4C9EF0'; ORANGE = '#FF8C00'

HISTORY_N = 500   
HISTORY_S = 15   


class DashboardBackend(Node):

    def __init__(self):
        super().__init__('neuracar_dashboard')

        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2)

        self.lock = threading.Lock()

        self.ref_xs:  list = []
        self.ref_ys:  list = []
        self.real_xs: list = []
        self.real_ys: list = []
        self.ref_received = False

        # Pose actual
        self.pos_x       = 0.0
        self.pos_y       = 0.0
        self.heading_deg = 0.0

        # Velocidad
        self.vel_linear  = 0.0
        self.vel_angular = 0.0
        self.pid_sp      = 0.0   
        self.pid_real    = 0.0    
        self.pid_error   = 0.0
        self.throttle    = 0.0
        self.steering    = 0.0
        self.steer_sp    = 0.0

        self.obstacle    = False
        self.obs_stops   = 0

        self.status_buf: deque = deque(maxlen=5)

        self.t_start   = None
        self.h_t       = deque(maxlen=HISTORY_N)
        self.h_sp      = deque(maxlen=HISTORY_N)
        self.h_real    = deque(maxlen=HISTORY_N)
        self.h_thr     = deque(maxlen=HISTORY_N)
        self.h_steer   = deque(maxlen=HISTORY_N)

        self._sp_t     = 0.0
        self._real_t   = 0.0

        self.create_subscription(
            Path, '/neuracar/path_reference', self._ref_cb, 10)
        self.create_subscription(
            Path, '/neuracar/path_real', self._real_path_cb, qos_be)

        self.create_subscription(
            Odometry, '/neuracar/odometry', self._odom_cb, qos_be)
        self.create_subscription(
            TwistStamped, '/neuracar/velocity', self._vel_cb, qos_be)

        self.create_subscription(
            Float32, '/neuracar/cmd_velocity', self._sp_cb, 10)
        self.create_subscription(
            Float32, '/neuracar/wheel_speed', self._wheel_cb, 10)
        self.create_subscription(
            Float32, '/neuracar/cmd_steering', self._steer_sp_cb, 10)

        self.create_subscription(
            Vector3Stamped, '/neuracar/user_command', self._cmd_cb, 10)

        self.create_subscription(
            Bool, '/neuracar/lidar/obstacle_alert', self._obs_cb, 10)

        self.create_subscription(
            String, '/neuracar/status', self._status_cb, 10)

        self.get_logger().info('Dashboard iniciado')

    def _ref_cb(self, msg: Path):
        xs = [p.pose.position.x for p in msg.poses]
        ys = [p.pose.position.y for p in msg.poses]
        with self.lock:
            self.ref_xs = xs
            self.ref_ys = ys
            self.ref_received = True

    def _real_path_cb(self, msg: Path):
        xs = [p.pose.position.x for p in msg.poses]
        ys = [p.pose.position.y for p in msg.poses]
        with self.lock:
            self.real_xs = xs
            self.real_ys = ys

    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y**2 + q.z**2)
        with self.lock:
            self.pos_x       = msg.pose.pose.position.x
            self.pos_y       = msg.pose.pose.position.y
            self.heading_deg = math.degrees(math.atan2(siny, cosy))

    def _vel_cb(self, msg: TwistStamped):
        with self.lock:
            self.vel_linear  = msg.twist.linear.x
            self.vel_angular = msg.twist.angular.z

    def _sp_cb(self, msg: Float32):
        now = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            self.pid_sp    = float(msg.data)
            self.pid_error = self.pid_sp - self.pid_real
            self._sp_t     = now

    def _wheel_cb(self, msg: Float32):
        now = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            self.pid_real  = float(msg.data)
            self.pid_error = self.pid_sp - self.pid_real
            self._real_t   = now

        if self.t_start is None:
            self.t_start = now
        t = now - self.t_start
        with self.lock:
            self.h_t.append(t)
            self.h_sp.append(self.pid_sp)
            self.h_real.append(self.pid_real)
            self.h_thr.append(self.throttle)
            self.h_steer.append(self.steer_sp)

    def _steer_sp_cb(self, msg: Float32):
        with self.lock:
            self.steer_sp = float(msg.data)

    def _cmd_cb(self, msg: Vector3Stamped):
        with self.lock:
            self.throttle = float(msg.vector.x)
            self.steering = float(msg.vector.y)

    def _obs_cb(self, msg: Bool):
        was = self.obstacle
        with self.lock:
            self.obstacle = bool(msg.data)
            if self.obstacle and not was:
                self.obs_stops += 1

    def _status_cb(self, msg: String):
        with self.lock:
            self.status_buf.append(msg.data)


class Dashboard(QMainWindow):

    def __init__(self, node: DashboardBackend):
        super().__init__()
        self.node = node
        self.setWindowTitle('Neuracar Dashboard')
        self.setMinimumSize(1400, 800)
        self._apply_style()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        left.setFixedWidth(300)
        lv = QVBoxLayout(left)
        lv.setSpacing(5)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self._build_pose())
        lv.addWidget(self._build_pid())
        lv.addWidget(self._build_actuators())
        lv.addWidget(self._build_lidar())
        lv.addWidget(self._build_status())
        lv.addStretch()
        splitter.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(5)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self._build_trajectory(), 3)   
        rv.addWidget(self._build_plots(), 2)         
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 8)
        splitter.setSizes([300, 1100])

        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(50)  

    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background:{BG}; color:{WHITE};
                font-family:'DejaVu Sans'; font-size:11px; }}
            QGroupBox {{
                border:1px solid {BORDER}; border-radius:6px;
                margin-top:10px; padding:4px; padding-top:14px;
                font-weight:bold; font-size:10px; color:{DIM}; }}
            QGroupBox::title {{
                subcontrol-origin:margin; left:8px; color:{CYAN}; }}
            QSplitter::handle {{ background:{BORDER}; }}
            QProgressBar {{
                border:1px solid {BORDER}; border-radius:3px;
                background:{CARD}; text-align:center;
                font-size:10px; color:{WHITE}; }}
            QProgressBar::chunk {{ border-radius:3px; }}
        """)

    def _lbl(self, text, color=WHITE, size=11, bold=False):
        l = QLabel(text)
        l.setStyleSheet(
            f'color:{color};font-size:{size}px;'
            f'font-weight:{"bold" if bold else "normal"};')
        return l

    def _build_pose(self):
        grp = QGroupBox('Odometría — pose actual')
        g   = QGridLayout(grp); g.setSpacing(4)
        g.addWidget(self._lbl('X:', DIM), 0, 0)
        self.l_x   = self._lbl('—', CYAN, 12, True); g.addWidget(self.l_x, 0, 1)
        g.addWidget(self._lbl('Y:', DIM), 1, 0)
        self.l_y   = self._lbl('—', CYAN, 12, True); g.addWidget(self.l_y, 1, 1)
        g.addWidget(self._lbl('Hdg:', DIM), 2, 0)
        self.l_hdg = self._lbl('—', YELLOW, 12, True); g.addWidget(self.l_hdg, 2, 1)
        g.addWidget(self._lbl('Vel:', DIM), 3, 0)
        self.l_vel = self._lbl('—', BLUE, 12, True); g.addWidget(self.l_vel, 3, 1)
        return grp

    def _build_pid(self):
        grp = QGroupBox('PID Velocidad')
        g   = QGridLayout(grp); g.setSpacing(6)
        g.addWidget(self._lbl('Setpoint:', DIM), 0, 0)
        self.l_sp   = self._lbl('— m/s', GREEN, 13, True); g.addWidget(self.l_sp, 0, 1)
        g.addWidget(self._lbl('Real:', DIM), 1, 0)
        self.l_real = self._lbl('— m/s', BLUE, 13, True); g.addWidget(self.l_real, 1, 1)
        g.addWidget(self._lbl('Error:', DIM), 2, 0)
        self.l_err  = self._lbl('— m/s', WHITE, 12, True); g.addWidget(self.l_err, 2, 1)
        g.addWidget(self._lbl('Estado:', DIM), 3, 0)
        self.l_pid_st = self._lbl('—', DIM, 12, True); g.addWidget(self.l_pid_st, 3, 1)
        g.addWidget(self._lbl('Seguimiento:', DIM, 10), 4, 0, 1, 2)
        self.bar_follow = QProgressBar()
        self.bar_follow.setRange(0, 100); self.bar_follow.setValue(0)
        self.bar_follow.setFixedHeight(18); self.bar_follow.setFormat('%v%')
        g.addWidget(self.bar_follow, 5, 0, 1, 2)
        return grp

    def _build_actuators(self):
        grp = QGroupBox('Actuadores → ESP32-A')
        g   = QGridLayout(grp); g.setSpacing(4)
        g.addWidget(self._lbl('Throttle:', DIM), 0, 0)
        self.l_thr  = self._lbl('—', ORANGE, 12, True); g.addWidget(self.l_thr, 0, 1)
        g.addWidget(self._lbl('Steering:', DIM), 1, 0)
        self.l_str  = self._lbl('—', YELLOW, 12, True); g.addWidget(self.l_str, 1, 1)
        g.addWidget(self._lbl('Steer SP:', DIM), 2, 0)
        self.l_ssp  = self._lbl('—', DIM, 11, True); g.addWidget(self.l_ssp, 2, 1)
        return grp

    def _build_lidar(self):
        grp = QGroupBox('LiDAR — obstacle_detector')
        g   = QGridLayout(grp); g.setSpacing(4)
        g.addWidget(self._lbl('Estado:', DIM), 0, 0)
        self.l_obs   = self._lbl('LIBRE', GREEN, 12, True); g.addWidget(self.l_obs, 0, 1)
        g.addWidget(self._lbl('Paradas:', DIM), 1, 0)
        self.l_stops = self._lbl('0', DIM, 11, True); g.addWidget(self.l_stops, 1, 1)
        return grp

    def _build_status(self):
        grp = QGroupBox('Firmware — últimos eventos')
        v   = QVBoxLayout(grp); v.setSpacing(2)
        self.status_labels = []
        for _ in range(4):
            l = self._lbl('', DIM, 9)
            l.setWordWrap(True)
            v.addWidget(l)
            self.status_labels.append(l)
        return grp

    def _build_trajectory(self):
        grp = QGroupBox(
            'Trayectoria — verde: referencia  |  azul: real  |  ● posición actual')
        v   = QVBoxLayout(grp); v.setContentsMargins(4, 16, 4, 4)

        pw = pg.GraphicsLayoutWidget()
        pw.setBackground(CARD)
        v.addWidget(pw)

        self.p_traj = pw.addPlot()
        self.p_traj.setLabel('left',   'Y [m]', color=DIM, size='8pt')
        self.p_traj.setLabel('bottom', 'X [m]', color=DIM, size='8pt')
        self.p_traj.setAspectLocked(True)
        self.p_traj.showGrid(x=True, y=True, alpha=0.15)
        self.p_traj.getAxis('bottom').setPen(BORDER)
        self.p_traj.getAxis('left').setPen(BORDER)

        self.c_ref  = self.p_traj.plot(
            pen=pg.mkPen(GREEN, width=2, style=Qt.DashLine),
            name='Referencia')

        self.c_real_path = self.p_traj.plot(
            pen=pg.mkPen(BLUE, width=2),
            name='Real')

        self.c_pos = self.p_traj.plot(
            pen=None,
            symbol='o',
            symbolSize=12,
            symbolBrush=YELLOW,
            symbolPen=pg.mkPen(WHITE, width=1))

        return grp

    def _build_plots(self):
        grp = QGroupBox('Velocidad en tiempo real')
        v   = QVBoxLayout(grp); v.setContentsMargins(4, 16, 4, 4)

        pw = pg.GraphicsLayoutWidget()
        pw.setBackground(CARD)
        v.addWidget(pw)

        self.p_vel = pw.addPlot(row=0, col=0)
        self.p_vel.setTitle(
            '<span style="color:#8B949E;font-size:9pt">'
            '<span style="color:#3FB950">■ Setpoint</span> '
            '<span style="color:#4C9EF0">■ Real</span></span>')
        self.p_vel.setLabel('left', 'm/s', color=DIM, size='8pt')
        self.p_vel.showGrid(x=True, y=True, alpha=0.15)
        self.p_vel.addLine(y=0, pen=pg.mkPen(BORDER, width=1,
                                              style=Qt.DashLine))
        self.c_vel_sp   = self.p_vel.plot(pen=pg.mkPen(GREEN, width=2))
        self.c_vel_real = self.p_vel.plot(pen=pg.mkPen(BLUE,  width=2))

        self.p_thr = pw.addPlot(row=0, col=1)
        self.p_thr.setTitle(
            '<span style="color:#FF8C00;font-size:9pt">■ Throttle → ESP32</span>')
        self.p_thr.setLabel('left', '[-1,1]', color=DIM, size='8pt')
        self.p_thr.setYRange(-1.1, 1.1)
        self.p_thr.showGrid(x=True, y=True, alpha=0.15)
        self.p_thr.setXLink(self.p_vel)
        self.c_thr_line = self.p_thr.plot(pen=pg.mkPen(ORANGE, width=2))

        return grp

    def _refresh(self):
        n = self.node

        with n.lock:
            ref_xs   = list(n.ref_xs)
            ref_ys   = list(n.ref_ys)
            real_xs  = list(n.real_xs)
            real_ys  = list(n.real_ys)
            pos_x    = n.pos_x
            pos_y    = n.pos_y
            hdg      = n.heading_deg
            sp       = n.pid_sp
            real     = n.pid_real
            err      = n.pid_error
            thr      = n.throttle
            steer    = n.steering
            ssp      = n.steer_sp
            vel_lin  = n.vel_linear
            ht       = list(n.h_t)
            hsp      = list(n.h_sp)
            hreal    = list(n.h_real)
            hthr     = list(n.h_thr)
            obs      = n.obstacle
            stops    = n.obs_stops
            statuses = list(n.status_buf)

        if ref_xs:
            self.c_ref.setData(ref_xs, ref_ys)
        if real_xs:
            self.c_real_path.setData(real_xs, real_ys)
        self.c_pos.setData([pos_x], [pos_y])

        if len(ht) >= 2:
            self.c_vel_sp.setData(ht,  hsp)
            self.c_vel_real.setData(ht, hreal)
            self.c_thr_line.setData(ht, hthr)
            te = ht[-1]
            x0 = max(0.0, te - HISTORY_S)
            self.p_vel.setXRange(x0, te + 0.5, padding=0)

        self.l_x.setText(f'{pos_x:.3f} m')
        self.l_y.setText(f'{pos_y:.3f} m')
        self.l_hdg.setText(f'{hdg:.1f}°')
        vc = GREEN if abs(vel_lin) < 0.05 else BLUE
        self.l_vel.setText(f'{vel_lin:.3f} m/s')
        self.l_vel.setStyleSheet(
            f'color:{vc};font-size:12px;font-weight:bold;')

        self.l_sp.setText(f'{sp:+.3f} m/s')
        rc = (GREEN if abs(err) < 0.03
              else YELLOW if abs(err) < 0.08 else RED)
        self.l_real.setText(f'{real:+.3f} m/s')
        self.l_real.setStyleSheet(
            f'color:{rc};font-size:13px;font-weight:bold;')
        ec = GREEN if abs(err) < 0.03 else YELLOW if abs(err) < 0.08 else RED
        self.l_err.setText(f'{err:+.3f} m/s')
        self.l_err.setStyleSheet(
            f'color:{ec};font-size:12px;font-weight:bold;')

        if abs(sp) < 0.02:
            pid_st, pid_sc = 'INACTIVO', DIM
        elif abs(err) < 0.03:
            pid_st, pid_sc = ' CONVERGIDO', GREEN
        elif abs(err) < 0.08:
            pid_st, pid_sc = ' AJUSTANDO', YELLOW
        else:
            pid_st, pid_sc = ' ERROR ALTO', RED
        self.l_pid_st.setText(pid_st)
        self.l_pid_st.setStyleSheet(
            f'color:{pid_sc};font-size:12px;font-weight:bold;')

        pct = max(0, int((1.0 - min(abs(err)/0.3, 1.0)) * 100)) \
              if abs(sp) > 0.02 else 0
        self.bar_follow.setValue(pct)
        fc = GREEN if pct > 85 else YELLOW if pct > 60 else RED
        self.bar_follow.setStyleSheet(
            f'QProgressBar::chunk{{background:{fc};border-radius:3px;}}')

        tc = GREEN if thr == 0 else ORANGE
        self.l_thr.setText(f'{thr:+.3f}')
        self.l_thr.setStyleSheet(
            f'color:{tc};font-size:12px;font-weight:bold;')
        sc = RED if abs(steer) > 0.3 else YELLOW if abs(steer) > 0.1 else GREEN
        self.l_str.setText(f'{steer:+.3f}')
        self.l_str.setStyleSheet(
            f'color:{sc};font-size:12px;font-weight:bold;')
        self.l_ssp.setText(f'{ssp:+.3f}')

        if obs:
            self.l_obs.setText('¡OBSTÁCULO!')
            self.l_obs.setStyleSheet(
                f'color:{RED};font-size:12px;font-weight:bold;')
        else:
            self.l_obs.setText('LIBRE')
            self.l_obs.setStyleSheet(
                f'color:{GREEN};font-size:12px;font-weight:bold;')
        self.l_stops.setText(str(stops))

        for i, lbl in enumerate(self.status_labels):
            if i < len(statuses):
                s = statuses[-(i+1)]
                c = RED if 'ERR' in s or 'ESTOP' in s else \
                    YELLOW if 'WARN' in s or 'LOW' in s else DIM
                lbl.setText(s)
                lbl.setStyleSheet(f'color:{c};font-size:9px;')
            else:
                lbl.setText('')


def main():
    rclpy.init()
    node = DashboardBackend()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    app = QApplication(sys.argv)
    win = Dashboard(node)
    win.show()

    try:
        app.exec_()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()