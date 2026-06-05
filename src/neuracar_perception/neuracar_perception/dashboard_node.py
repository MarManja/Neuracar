#!/usr/bin/env python3
"""
=======================================================================
 Neuracar Dashboard — v2.0
 Proyecto: Neuracar
-----------------------------------------------------------------------
 Cambios v2.0 vs v1.0:
   - Panel PID Tuning: setpoint vs velocidad real en tiempo real
     con indicadores visuales de estado del lazo de control
   - Suscripción a /neuracar/cmd_velocity (setpoint m/s del PID)
   - Suscripción a /neuracar/wheel_speed  (velocidad real encoder)
   - Suscripción a /neuracar/battery      (BatteryState)
   - Panel Batería: voltaje, corriente, SoC con colores de alerta
   - Las gráficas de telemetría ahora incluyen setpoint vs real
     para validar el PID visualmente en tiempo real

 Layout:
   IZQUIERDA: System Status + PID Tuning + Lane + LiDAR + Pose
   DERECHA:   Cámara frontal + Lane debug + Telemetría

 Topics:
   /camera/color/image_raw        sensor_msgs/Image
   /neuracar/lane_image           sensor_msgs/CompressedImage
   /neuracar/lane_error           geometry_msgs/Vector3Stamped
   /neuracar/lidar/obstacle_alert std_msgs/Bool
   /neuracar/velocity             geometry_msgs/TwistStamped
   /neuracar/odometry             nav_msgs/Odometry
   /neuracar/user_command         geometry_msgs/Vector3Stamped
   /neuracar/cmd_velocity         std_msgs/Float32   ← NUEVO v2.0
   /neuracar/wheel_speed          std_msgs/Float32   ← NUEVO v2.0
   /neuracar/battery              sensor_msgs/BatteryState ← NUEVO v2.0
=======================================================================
"""

import sys
import math
import threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
from collections import deque

from geometry_msgs.msg import TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage, BatteryState
from std_msgs.msg import Bool, Float32

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QGroupBox, QGridLayout,
    QSizePolicy, QSplitter, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import pyqtgraph as pg

# ── Paleta ───────────────────────────────────────────────────────────
BG     = '#0D1117'; CARD   = '#161B22'; BORDER = '#30363D'
GREEN  = '#3FB950'; YELLOW = '#E3B341'; RED    = '#FF4444'
CYAN   = '#00E5CC'; WHITE  = '#E6EDF3'; DIM    = '#8B949E'
M_BLUE = '#0072BD'; M_RED  = '#D95319'; M_YEL  = '#EDB120'
M_PURP = '#7E2F8E'; ORANGE = '#FF8C00'

CAM_W, CAM_H = 640, 480
HISTORY_S = 30    # segundos de historial en gráficas
HISTORY_N = 300   # puntos a 10 Hz


# ── Utilidades de imagen ──────────────────────────────────────────────
def _image_msg_to_bgr(msg: Image):
    try:
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        enc = msg.encoding.lower()
        if enc == 'rgb8':
            return raw.reshape(msg.height, msg.width, 3)[:, :, ::-1].copy()
        if enc == 'bgr8':
            return raw.reshape(msg.height, msg.width, 3)
        if enc in ('mono8', '8uc1'):
            g = raw.reshape(msg.height, msg.width)
            return np.stack([g, g, g], axis=-1)
        if enc == 'rgba8':
            return raw.reshape(msg.height, msg.width, 4)[:, :, :3][:, :, ::-1].copy()
    except Exception:
        pass
    return None


def _compressed_to_bgr(msg: CompressedImage):
    try:
        return cv2.imdecode(
            np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


# ── ROS Backend ───────────────────────────────────────────────────────
class ROSBackend(Node):

    def __init__(self):
        super().__init__('neuracar_dashboard')

        _be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2)

        # ── Imágenes ─────────────────────────────────────────────────
        self.create_subscription(Image, '/camera/color/image_raw',
                                 self._cam_cb, _be)
        self.create_subscription(CompressedImage, '/neuracar/lane_image',
                                 self._lane_img_cb, _be)

        # ── Sensores ──────────────────────────────────────────────────
        self.create_subscription(Vector3Stamped, '/neuracar/lane_error',
                                 self._lane_err_cb, 10)
        self.create_subscription(Bool, '/neuracar/lidar/obstacle_alert',
                                 self._obs_cb, 10)
        self.create_subscription(TwistStamped, '/neuracar/velocity',
                                 self._vel_cb, 10)
        self.create_subscription(Odometry, '/neuracar/odometry',
                                 self._odom_cb, 10)
        self.create_subscription(Vector3Stamped, '/neuracar/user_command',
                                 self._cmd_cb, 10)

        # ── PID topics — NUEVO v2.0 ───────────────────────────────────
        self.create_subscription(Float32, '/neuracar/cmd_velocity',
                                 self._setpoint_cb, 10)
        self.create_subscription(Float32, '/neuracar/wheel_speed',
                                 self._wheel_speed_cb, 10)

        # ── Batería — NUEVO v2.0 ──────────────────────────────────────
        self.create_subscription(BatteryState, '/neuracar/battery',
                                 self._bat_cb, 10)

        self.lock = threading.Lock()

        # Frames
        self.frame_cam  = None; self.fps_cam  = 0.0; self._t_cam  = 0.0
        self.frame_lane = None; self.fps_lane = 0.0; self._t_lane = 0.0

        # Lane
        self.lane_cte  = 0.0; self.lane_conf = 0.0; self.lane_t = 0.0

        # Obstáculo
        self.obstacle   = False; self.obs_stops = 0; self.obs_last_t = 0.0

        # Velocidad / pose
        self.vel_linear  = 0.0; self.vel_angular = 0.0
        self.pos_x       = 0.0; self.pos_y = 0.0; self.heading_deg = 0.0
        self.vel_t       = 0.0

        # Comando activo
        self.cmd_throttle = 0.0; self.cmd_steering = 0.0

        # PID — NUEVO v2.0
        self.pid_setpoint   = 0.0   # m/s deseados
        self.pid_measured   = 0.0   # m/s reales del encoder
        self.pid_error      = 0.0   # diferencia
        self.pid_throttle   = 0.0   # throttle calculado por PID
        self.pid_sp_t       = 0.0   # timestamp último setpoint

        # Batería — NUEVO v2.0
        self.bat_voltage  = 0.0
        self.bat_current  = 0.0
        self.bat_soc      = 0.0    # 0.0-1.0
        self.bat_present  = False
        self.bat_health   = 0      # 0=good, 1=dead

        # Historial (300 pts ~ 30 s a 10 Hz)
        N = HISTORY_N
        self.t_start   = None
        self.h_time    = deque(maxlen=N)
        self.h_vel     = deque(maxlen=N)    # velocidad real
        self.h_sp      = deque(maxlen=N)    # setpoint PID
        self.h_thr     = deque(maxlen=N)    # throttle calculado
        self.h_cte     = deque(maxlen=N)
        self.h_conf    = deque(maxlen=N)
        self.h_steer   = deque(maxlen=N)
        self.h_bat_v   = deque(maxlen=N)    # voltaje batería

        self.get_logger().info('Neuracar Dashboard v2.0 iniciado')

    # ── Image callbacks ───────────────────────────────────────────────
    def _cam_cb(self, msg: Image):
        bgr = _image_msg_to_bgr(msg)
        if bgr is None: return
        bgr = cv2.resize(bgr, (CAM_W, CAM_H), interpolation=cv2.INTER_NEAREST)
        now = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            self.frame_cam = bgr
            self.fps_cam   = 1.0 / max(now - self._t_cam, 0.001)
            self._t_cam    = now

    def _lane_img_cb(self, msg: CompressedImage):
        bgr = _compressed_to_bgr(msg)
        if bgr is None: return
        now = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            self.frame_lane = bgr
            self.fps_lane   = 1.0 / max(now - self._t_lane, 0.001)
            self._t_lane    = now

    # ── Data callbacks ────────────────────────────────────────────────
    def _lane_err_cb(self, msg: Vector3Stamped):
        self.lane_cte  = float(msg.vector.x)
        self.lane_conf = float(msg.vector.y)
        self.lane_t    = self.get_clock().now().nanoseconds / 1e9

    def _obs_cb(self, msg: Bool):
        was = self.obstacle; self.obstacle = bool(msg.data)
        if self.obstacle and not was: self.obs_stops += 1
        self.obs_last_t = self.get_clock().now().nanoseconds / 1e9

    def _vel_cb(self, msg: TwistStamped):
        self.vel_linear  = float(msg.twist.linear.x)
        self.vel_angular = float(msg.twist.angular.z)
        now = self.get_clock().now().nanoseconds / 1e9
        self.vel_t = now
        if self.t_start is None: self.t_start = now
        t = now - self.t_start
        with self.lock:
            self.h_time.append(t)
            self.h_vel.append(self.vel_linear)
            self.h_sp.append(self.pid_setpoint)
            self.h_thr.append(self.cmd_throttle)
            self.h_cte.append(self.lane_cte)
            self.h_conf.append(self.lane_conf)
            self.h_steer.append(self.cmd_steering)
            self.h_bat_v.append(self.bat_voltage)

    def _odom_cb(self, msg: Odometry):
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y**2 + q.z**2)
        self.heading_deg = math.degrees(math.atan2(siny, cosy))

    def _cmd_cb(self, msg: Vector3Stamped):
        if msg.header.frame_id == 'watchdog': return
        self.cmd_throttle = float(msg.vector.x)
        self.cmd_steering = float(msg.vector.y)

    # ── PID callbacks — NUEVO v2.0 ────────────────────────────────────
    def _setpoint_cb(self, msg: Float32):
        self.pid_setpoint = float(msg.data)
        self.pid_sp_t     = self.get_clock().now().nanoseconds / 1e9
        self.pid_error    = self.pid_setpoint - self.pid_measured

    def _wheel_speed_cb(self, msg: Float32):
        self.pid_measured = float(msg.data)
        self.pid_error    = self.pid_setpoint - self.pid_measured

    # ── Batería callback — NUEVO v2.0 ─────────────────────────────────
    def _bat_cb(self, msg: BatteryState):
        self.bat_voltage = float(msg.voltage)
        self.bat_current = float(msg.current)
        self.bat_soc     = float(msg.percentage)   # 0.0-1.0
        self.bat_present = bool(msg.present)
        self.bat_health  = msg.power_supply_health


# ── Dashboard GUI ─────────────────────────────────────────────────────
class Dashboard(QMainWindow):

    def __init__(self, node: ROSBackend):
        super().__init__()
        self.node = node
        self.setWindowTitle('Neuracar Dashboard v2.0')
        self.setMinimumSize(1600, 900)
        self._apply_style()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(6); root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── Columna izquierda — status + PID + lane + lidar + pose ───
        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setSpacing(4); lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self._build_system())
        lv.addWidget(self._build_pid())       # ← NUEVO v2.0
        lv.addWidget(self._build_battery())   # ← NUEVO v2.0
        lv.addWidget(self._build_lane())
        lv.addWidget(self._build_lidar())
        lv.addWidget(self._build_pose())
        splitter.addWidget(left)

        # ── Columna derecha — cámara + telemetría ────────────────────
        right = QWidget()
        rv    = QVBoxLayout(right)
        rv.setSpacing(4); rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self._cam_panel('Cámara frontal — /camera/color/image_raw', 'cam'), 2)
        rv.addWidget(self._cam_panel('Lane debug — /neuracar/lane_image', 'lane'), 1)
        rv.addWidget(self._build_plots(), 2)   # gráficas abajo derecha
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        splitter.setSizes([440, 1160])

        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)   # 30 FPS

    # ── Estilo global ─────────────────────────────────────────────────
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
        w = 'bold' if bold else 'normal'
        l.setStyleSheet(f'color:{color};font-size:{size}px;font-weight:{w};')
        return l

    # ── Panel: System Status ──────────────────────────────────────────
    def _build_system(self):
        grp = QGroupBox('System Status')
        g   = QGridLayout(grp); g.setSpacing(4)
        g.addWidget(self._lbl('ROS:', DIM), 0, 0)
        self.l_ros = self._lbl('OK', GREEN, 11, True); g.addWidget(self.l_ros, 0, 1)
        g.addWidget(self._lbl('Vel real:', DIM), 1, 0)
        self.l_vel = self._lbl('—', M_BLUE, 12, True); g.addWidget(self.l_vel, 1, 1)
        g.addWidget(self._lbl('Throttle:', DIM), 2, 0)
        self.l_thr = self._lbl('—', CYAN, 12, True); g.addWidget(self.l_thr, 2, 1)
        g.addWidget(self._lbl('Steering:', DIM), 3, 0)
        self.l_str = self._lbl('—', M_YEL, 12, True); g.addWidget(self.l_str, 3, 1)
        g.addWidget(self._lbl('FPS:', DIM), 4, 0)
        self.l_fps = self._lbl('—', DIM, 9); g.addWidget(self.l_fps, 4, 1)
        return grp

    # ── Panel: PID Tuning — NUEVO v2.0 ───────────────────────────────
    def _build_pid(self):
        grp = QGroupBox('PID Velocidad — velocity_pid_node')
        g   = QGridLayout(grp); g.setSpacing(6)

        # Setpoint vs medido
        g.addWidget(self._lbl('Setpoint:', DIM), 0, 0)
        self.l_pid_sp = self._lbl('— m/s', GREEN, 13, True)
        g.addWidget(self.l_pid_sp, 0, 1)

        g.addWidget(self._lbl('Real:', DIM), 1, 0)
        self.l_pid_real = self._lbl('— m/s', M_BLUE, 13, True)
        g.addWidget(self.l_pid_real, 1, 1)

        g.addWidget(self._lbl('Error:', DIM), 2, 0)
        self.l_pid_err = self._lbl('— m/s', WHITE, 12, True)
        g.addWidget(self.l_pid_err, 2, 1)

        # Estado del lazo
        g.addWidget(self._lbl('Estado:', DIM), 3, 0)
        self.l_pid_state = self._lbl('INACTIVO', DIM, 12, True)
        g.addWidget(self.l_pid_state, 3, 1)

        # Barra de seguimiento — qué tan cerca está real de setpoint
        g.addWidget(self._lbl('Seguimiento:', DIM), 4, 0, 1, 2)
        self.pid_bar = QProgressBar()
        self.pid_bar.setRange(0, 100)
        self.pid_bar.setValue(0)
        self.pid_bar.setFixedHeight(18)
        self.pid_bar.setFormat('%v%')
        g.addWidget(self.pid_bar, 5, 0, 1, 2)

        # Barra de error — centrada en 0
        g.addWidget(self._lbl('Error [-0.3, +0.3 m/s]:', DIM), 6, 0, 1, 2)
        self.error_bar = QLabel()
        self.error_bar.setFixedHeight(14)
        self.error_bar.setStyleSheet(
            f'background:{CARD};border:1px solid {BORDER};border-radius:3px;')
        g.addWidget(self.error_bar, 7, 0, 1, 2)

        return grp

    # ── Panel: Batería — NUEVO v2.0 ───────────────────────────────────
    def _build_battery(self):
        grp = QGroupBox('Batería NiMH — /neuracar/battery')
        g   = QGridLayout(grp); g.setSpacing(4)

        g.addWidget(self._lbl('Voltaje:', DIM), 0, 0)
        self.l_bat_v = self._lbl('— V', WHITE, 12, True)
        g.addWidget(self.l_bat_v, 0, 1)

        g.addWidget(self._lbl('Corriente:', DIM), 1, 0)
        self.l_bat_i = self._lbl('— A', WHITE, 11, True)
        g.addWidget(self.l_bat_i, 1, 1)

        g.addWidget(self._lbl('SoC:', DIM), 2, 0)
        self.bat_soc_bar = QProgressBar()
        self.bat_soc_bar.setRange(0, 100)
        self.bat_soc_bar.setValue(0)
        self.bat_soc_bar.setFixedHeight(18)
        self.bat_soc_bar.setFormat('SoC: %v%')
        g.addWidget(self.bat_soc_bar, 2, 1)

        return grp

    # ── Panel: Lane Detector ──────────────────────────────────────────
    def _build_lane(self):
        grp = QGroupBox('Lane Detector')
        g   = QGridLayout(grp); g.setSpacing(4)
        g.addWidget(self._lbl('Estado:', DIM), 0, 0)
        self.l_lstate = self._lbl('—', DIM, 12, True); g.addWidget(self.l_lstate, 0, 1)
        g.addWidget(self._lbl('CTE:', DIM), 1, 0)
        self.l_cte = self._lbl('—', YELLOW, 12, True); g.addWidget(self.l_cte, 1, 1)
        g.addWidget(self._lbl('Confianza:', DIM), 2, 0)
        self.l_conf = self._lbl('—', GREEN, 12, True); g.addWidget(self.l_conf, 2, 1)
        self.cte_bar = QLabel()
        self.cte_bar.setFixedHeight(14)
        self.cte_bar.setStyleSheet(
            f'background:{CARD};border:1px solid {BORDER};border-radius:3px;')
        g.addWidget(self.cte_bar, 3, 0, 1, 2)
        return grp

    # ── Panel: LiDAR ─────────────────────────────────────────────────
    def _build_lidar(self):
        grp = QGroupBox('LiDAR — obstacle_detector_node')
        g   = QGridLayout(grp); g.setSpacing(4)
        g.addWidget(self._lbl('Estado:', DIM), 0, 0)
        self.l_obs   = self._lbl('LIBRE', GREEN, 12, True); g.addWidget(self.l_obs, 0, 1)
        g.addWidget(self._lbl('Paradas:', DIM), 1, 0)
        self.l_stops = self._lbl('0', DIM, 11, True); g.addWidget(self.l_stops, 1, 1)
        return grp

    # ── Panel: Pose ───────────────────────────────────────────────────
    def _build_pose(self):
        grp = QGroupBox('Odometría — /neuracar/odometry')
        g   = QGridLayout(grp); g.setSpacing(4)
        g.addWidget(self._lbl('X:', DIM), 0, 0)
        self.l_x   = self._lbl('—', CYAN, 11, True); g.addWidget(self.l_x, 0, 1)
        g.addWidget(self._lbl('Y:', DIM), 1, 0)
        self.l_y   = self._lbl('—', CYAN, 11, True); g.addWidget(self.l_y, 1, 1)
        g.addWidget(self._lbl('Hdg:', DIM), 2, 0)
        self.l_hdg = self._lbl('—', M_YEL, 11, True); g.addWidget(self.l_hdg, 2, 1)
        return grp

    # ── Panel: Gráficas de telemetría ─────────────────────────────────
    def _build_plots(self):
        grp = QGroupBox('Telemetría en tiempo real')
        lay = QVBoxLayout(grp); lay.setSpacing(2)
        pw  = pg.GraphicsLayoutWidget()
        pw.setBackground(CARD)
        lay.addWidget(pw)

        def mp(row, title, ylabel, yrange=None):
            p = pw.addPlot(row=row, col=0, title=title)
            p.setLabel('left', ylabel, color=DIM, size='7pt')
            p.showGrid(x=True, y=True, alpha=0.2)
            p.getAxis('bottom').setPen(BORDER)
            if yrange: p.setYRange(*yrange)
            p.titleLabel.setText(title, color=DIM, size='8pt')
            return p

        # Gráfica 1: PID — setpoint vs real (la más importante para tuning)
        self.p_pid  = mp(0, 'PID Velocidad: setpoint (verde) vs real (azul)', 'm/s')
        self.c_sp   = self.p_pid.plot(pen=pg.mkPen(GREEN,  width=2), name='Setpoint')
        self.c_real = self.p_pid.plot(pen=pg.mkPen(M_BLUE, width=2), name='Real')
        self.p_pid.addLine(y=0, pen=pg.mkPen(BORDER, width=1, style=Qt.DashLine))

        # Gráfica 2: Throttle calculado por el PID
        self.p_thr  = mp(1, 'Throttle PID → ESP32-A', '[-1,1]', (-1.1, 1.1))
        self.c_thr  = self.p_thr.plot(pen=pg.mkPen(ORANGE, width=2))
        self.p_thr.addLine(y=0, pen=pg.mkPen(BORDER, width=1, style=Qt.DashLine))

        # Gráfica 3: CTE lane follower
        self.p_cte  = mp(2, 'CTE error lateral', 'norm', (-1.1, 1.1))
        self.c_cte  = self.p_cte.plot(pen=pg.mkPen(M_YEL, width=2))
        self.p_cte.addLine(y=0, pen=pg.mkPen(BORDER, width=1, style=Qt.DashLine))

        # Gráfica 4: Steering
        self.p_steer = mp(3, 'Steering enviado', '[-1,1]', (-1.1, 1.1))
        self.c_steer = self.p_steer.plot(pen=pg.mkPen(M_RED, width=2))
        self.p_steer.addLine(y=0, pen=pg.mkPen(BORDER, width=1, style=Qt.DashLine))

        return grp

    # ── Cámara panel ──────────────────────────────────────────────────
    def _cam_panel(self, title, key):
        grp = QGroupBox(title)
        lay = QVBoxLayout(grp); lay.setContentsMargins(2, 14, 2, 2)
        lbl = QLabel('SIN SEÑAL')
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            'background:#050D16;color:#2A4060;'
            'font-size:14px;font-weight:bold;')
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lbl.setMinimumHeight(180)
        lay.addWidget(lbl)
        if key == 'cam': self.v_cam  = lbl
        else:            self.v_lane = lbl
        return grp

    @staticmethod
    def _to_pixmap(bgr, tw, th):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qi   = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
        return QPixmap.fromImage(qi).scaled(
            tw, th, Qt.KeepAspectRatio, Qt.FastTransformation)

    # ── Refresh 30 FPS ────────────────────────────────────────────────
    def _refresh(self):
        n   = self.node
        now = n.get_clock().now().nanoseconds / 1e9

        with n.lock:
            cam  = n.frame_cam
            lane = n.frame_lane
            t    = list(n.h_time)
            hv   = list(n.h_vel)
            hsp  = list(n.h_sp)
            hthr = list(n.h_thr)
            hc   = list(n.h_cte)
            hco  = list(n.h_conf)
            hs   = list(n.h_steer)

        # ── Imágenes ──────────────────────────────────────────────────
        if cam is not None:
            self.v_cam.setPixmap(
                self._to_pixmap(cam, self.v_cam.width(), self.v_cam.height()))
        if lane is not None:
            self.v_lane.setPixmap(
                self._to_pixmap(lane, self.v_lane.width(), self.v_lane.height()))

        # ── Gráficas ──────────────────────────────────────────────────
        if t:
            self.c_real.setData(t, hv)
            self.c_sp.setData(t,   hsp)
            self.c_thr.setData(t,  hthr)
            self.c_cte.setData(t,  hc)
            self.c_steer.setData(t, hs)
            te = t[-1]
            for p in (self.p_pid, self.p_thr, self.p_cte, self.p_steer):
                p.setXRange(max(0, te - HISTORY_S), te + 0.5, padding=0)

        # ── System Status ─────────────────────────────────────────────
        ros_ok = (now - max(n._t_cam or 0, n._t_lane or 0, n.vel_t or 0, 1e-9)) < 3.0
        self.l_ros.setText('OK' if ros_ok else 'SIN SEÑAL')
        self.l_ros.setStyleSheet(
            f'color:{GREEN if ros_ok else RED};font-size:11px;font-weight:bold;')

        vc = GREEN if abs(n.vel_linear) < 0.05 else M_BLUE
        self.l_vel.setText(f'{n.vel_linear:.3f} m/s')
        self.l_vel.setStyleSheet(f'color:{vc};font-size:12px;font-weight:bold;')

        tc = GREEN if n.cmd_throttle == 0 else CYAN
        self.l_thr.setText(f'{n.cmd_throttle:+.3f}')
        self.l_thr.setStyleSheet(f'color:{tc};font-size:12px;font-weight:bold;')

        sc = (RED if abs(n.cmd_steering) > 0.3
              else YELLOW if abs(n.cmd_steering) > 0.1 else GREEN)
        self.l_str.setText(f'{n.cmd_steering:+.3f}')
        self.l_str.setStyleSheet(f'color:{sc};font-size:12px;font-weight:bold;')
        self.l_fps.setText(f'CAM:{n.fps_cam:.0f}  LANE:{n.fps_lane:.0f}')

        # ── Panel PID ─────────────────────────────────────────────────
        sp   = n.pid_setpoint
        real = n.pid_measured
        err  = n.pid_error

        # Setpoint
        self.l_pid_sp.setText(f'{sp:+.3f} m/s')
        self.l_pid_sp.setStyleSheet(
            f'color:{GREEN};font-size:13px;font-weight:bold;')

        # Velocidad real — color según qué tan cerca está del setpoint
        if abs(sp) < 0.02:
            rc = DIM
        elif abs(err) < 0.03:
            rc = GREEN    # dentro de ±3 cm/s — perfecto
        elif abs(err) < 0.08:
            rc = YELLOW   # dentro de ±8 cm/s — aceptable
        else:
            rc = RED      # error grande — PID necesita ajuste
        self.l_pid_real.setText(f'{real:+.3f} m/s')
        self.l_pid_real.setStyleSheet(
            f'color:{rc};font-size:13px;font-weight:bold;')

        # Error
        ec = GREEN if abs(err) < 0.03 else YELLOW if abs(err) < 0.08 else RED
        self.l_pid_err.setText(f'{err:+.3f} m/s')
        self.l_pid_err.setStyleSheet(
            f'color:{ec};font-size:12px;font-weight:bold;')

        # Estado del lazo
        if abs(sp) < 0.02:
            pid_st, pid_sc = 'INACTIVO', DIM
        elif abs(err) < 0.03:
            pid_st, pid_sc = '✓ CONVERGIDO', GREEN
        elif abs(err) < 0.08:
            pid_st, pid_sc = '~ AJUSTANDO', YELLOW
        else:
            pid_st, pid_sc = '✗ ERROR ALTO', RED
        self.l_pid_state.setText(pid_st)
        self.l_pid_state.setStyleSheet(
            f'color:{pid_sc};font-size:12px;font-weight:bold;')

        # Barra de seguimiento: 100% = error=0, 0% = error>=0.3 m/s
        if abs(sp) > 0.02:
            follow_pct = max(0, int((1.0 - min(abs(err) / 0.3, 1.0)) * 100))
        else:
            follow_pct = 0
        self.pid_bar.setValue(follow_pct)
        bar_color = GREEN if follow_pct > 85 else YELLOW if follow_pct > 60 else RED
        self.pid_bar.setStyleSheet(
            f'QProgressBar::chunk {{ background:{bar_color}; border-radius:3px; }}')

        # Barra de error centrada [-0.3, +0.3 m/s]
        bw  = max(self.error_bar.width(), 1)
        err_clamp = max(-0.3, min(0.3, err))
        center = bw // 2
        pos    = int(center + (err_clamp / 0.3) * center)
        pos    = max(2, min(pos, bw - 2))
        if err_clamp >= 0:
            x1, x2 = center, pos
        else:
            x1, x2 = pos, center
        ec2 = GREEN if abs(err) < 0.03 else YELLOW if abs(err) < 0.08 else RED
        self.error_bar.setStyleSheet(
            f'background: qlineargradient(x1:0, x2:1, '
            f'stop:{x1/bw:.3f} {CARD}, '
            f'stop:{x1/bw:.3f} {ec2}, '
            f'stop:{x2/bw:.3f} {ec2}, '
            f'stop:{x2/bw:.3f} {CARD});'
            f'border:1px solid {BORDER}; border-radius:3px;')

        # ── Panel Batería ─────────────────────────────────────────────
        if n.bat_present and n.bat_voltage > 1.0:
            bv = n.bat_voltage
            bi = n.bat_current
            bs = int(n.bat_soc * 100)

            bvc = (RED if bv < 5.5 else YELLOW if bv < 6.5 else GREEN)
            self.l_bat_v.setText(f'{bv:.2f} V')
            self.l_bat_v.setStyleSheet(
                f'color:{bvc};font-size:12px;font-weight:bold;')

            self.l_bat_i.setText(f'{bi:.2f} A')
            self.l_bat_i.setStyleSheet(
                f'color:{WHITE};font-size:11px;font-weight:bold;')

            self.bat_soc_bar.setValue(bs)
            soc_color = RED if bs < 20 else YELLOW if bs < 40 else GREEN
            self.bat_soc_bar.setStyleSheet(
                f'QProgressBar::chunk {{background:{soc_color};border-radius:3px;}}')
        else:
            self.l_bat_v.setText('N/A')
            self.l_bat_i.setText('N/A')
            self.bat_soc_bar.setValue(0)

        # ── Lane Status ───────────────────────────────────────────────
        age  = now - n.lane_t if n.lane_t > 0 else 99.0
        conf = n.lane_conf
        cte  = n.lane_cte

        if age > 1.5:   lt, lc = 'SIN SEÑAL', RED
        elif conf >= 0.7: lt, lc = 'DETECTADO', GREEN
        elif conf >= 0.25: lt, lc = 'PARCIAL', YELLOW
        else:            lt, lc = 'CIEGO', RED

        self.l_lstate.setText(lt)
        self.l_lstate.setStyleSheet(
            f'color:{lc};font-size:12px;font-weight:bold;')

        cc = RED if abs(cte) > 0.5 else YELLOW if abs(cte) > 0.25 else GREEN
        self.l_cte.setText(f'{cte:+.3f}')
        self.l_cte.setStyleSheet(f'color:{cc};font-size:12px;font-weight:bold;')
        self.l_conf.setText(f'{conf:.3f}')
        self.l_conf.setStyleSheet(f'color:{lc};font-size:12px;font-weight:bold;')

        bw  = max(self.cte_bar.width(), 1)
        pos = int((float(np.clip(cte, -1, 1)) + 1) * 0.5 * bw)
        pos = max(6, min(pos, bw - 6))
        self.cte_bar.setStyleSheet(
            f'background:qlineargradient(x1:0,x2:1,'
            f'stop:0 {CARD},'
            f'stop:{max(0,(pos-4)/bw):.3f} {CARD},'
            f'stop:{pos/bw:.3f} {cc},'
            f'stop:{min(1,(pos+4)/bw):.3f} {CARD},'
            f'stop:1 {CARD});'
            f'border:1px solid {BORDER};border-radius:3px;')

        # ── LiDAR ─────────────────────────────────────────────────────
        if n.obstacle:
            self.l_obs.setText('¡OBSTÁCULO!')
            self.l_obs.setStyleSheet(
                f'color:{RED};font-size:12px;font-weight:bold;')
        else:
            self.l_obs.setText('LIBRE')
            self.l_obs.setStyleSheet(
                f'color:{GREEN};font-size:12px;font-weight:bold;')
        self.l_stops.setText(str(n.obs_stops))

        # ── Pose ──────────────────────────────────────────────────────
        self.l_x.setText(f'{n.pos_x:.3f} m')
        self.l_y.setText(f'{n.pos_y:.3f} m')
        self.l_hdg.setText(f'{n.heading_deg:.1f}°')


# ── main ──────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = ROSBackend()
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