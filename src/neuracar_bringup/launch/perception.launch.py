"""
perception_launch.py — Neuracar
=================================
Stack de percepción y control de bajo nivel para modo AUTÓNOMO.

Lanzar en este orden:
  1. sensors.launch.py        — hardware (encoder, IMU, LiDAR, cámara)
  2. perception.launch.py     — este archivo
  3. stanley_controller_node  o  pure_pursuit_node  (controlador autónomo)

Para modo TELEOP no es necesario este launch:
  1. sensors.launch.py
  2. remote_control.launch.py  (incluye velocity_pid y obstacle_detector propios)

Nodos que lanza:
  odometry_node        — siempre activo
  obstacle_detector    — activo por default (obstacle_stop:=false para desactivar)
  velocity_pid_node    — siempre activo (convierte cmd_velocity → throttle real)
  lane_detector_node   — desactivado por default (lane_detection:=true para activar)

Tópicos que produce:
  /neuracar/odometry                   (nav_msgs/Odometry)          — pose 2D
  /neuracar/velocity                   (geometry_msgs/TwistStamped) — v y w
  /neuracar/lidar/obstacle_alert       (Bool)                       — obstáculo frente
  /neuracar/lidar/obstacle_alert_rear  (Bool)                       — obstáculo atrás
  /neuracar/user_command               (geometry_msgs/Vector3Stamped)— throttle+steering

Uso:
  ros2 launch neuracar_bringup perception.launch.py
  ros2 launch neuracar_bringup perception.launch.py lane_detection:=true
  ros2 launch neuracar_bringup perception.launch.py obstacle_stop:=false
  ros2 launch neuracar_bringup perception.launch.py distance_threshold:=0.60
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    obstacle_stop  = LaunchConfiguration('obstacle_stop').perform(context).lower() == 'true'
    lane_detection = LaunchConfiguration('lane_detection').perform(context).lower() == 'true'
    dist_thr       = float(LaunchConfiguration('distance_threshold').perform(context))
    reset_yaw      = LaunchConfiguration('reset_yaw_on_start').perform(context).lower() == 'true'
    debug_lidar    = LaunchConfiguration('debug_lidar').perform(context).lower() == 'true'

    nodes = []

    # ── Odometría ────────────────────────────────────────────────────────
    # Fusiona /neuracar/wheel_speed (encoder) + /neuracar/imu (BNO055).
    # Publica pose 2D por dead reckoning y velocidad del vehículo.
    # Broadcast TF: odom → base_link a 50 Hz.
    #
    # reset_yaw_on_start=True: el heading arranca en 0° relativo al arranque,
    # evitando que Stanley/Pure Pursuit vean un error de heading enorme al inicio.
    nodes.append(Node(
        package='neuracar_perception',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        parameters=[{
            'reset_yaw_on_start': reset_yaw,
        }],
    ))

    # ── Obstacle detector ────────────────────────────────────────────────
    # Procesa /scan del RPLidar y publica alertas frontal y trasera.
    # Stanley y Pure Pursuit las consumen para pausar o terminar la prueba.
    #
    # Montaje Neuracar: cable del LiDAR apunta hacia atrás →
    #   lidar_front_offset_rad = π   (frente del vehículo = 180° en frame raw)
    #   lidar_rear_offset_rad  = 0   (trasero = 0°, donde está el cable)
    #
    # Ventana angular:
    #   ±22.5° a v < 1.0 m/s  — cono estrecho, menos falsos positivos
    #   ±30.0° a v ≥ 1.0 m/s  — cono más ancho, frena antes
    if obstacle_stop:
        nodes.append(Node(
            package='neuracar_perception',
            executable='obstacle_detector_node',
            name='obstacle_detector',
            output='screen',
            parameters=[{
                'distance_threshold':     dist_thr,
                'angle_range_low_deg':    22.5,
                'angle_range_high_deg':   30.0,
                'velocity_threshold':     1.0,
                'lidar_front_offset_rad': 3.14159,
                'lidar_rear_offset_rad':  0.0,
                'debug_mode':             debug_lidar,
            }],
        ))

    # ── Velocity PID ─────────────────────────────────────────────────────
    # Convierte cmd_velocity [m/s] en throttle real usando feedforward (LUT)
    # + PID de corrección. Stanley y Pure Pursuit publican en cmd_velocity,
    # este nodo cierra el lazo y manda el comando final al ESP32.
    #
    # Entradas:  /neuracar/cmd_velocity  (del controlador autónomo)
    #            /neuracar/cmd_steering  (del controlador autónomo)
    #            /neuracar/wheel_speed   (feedback encoder, desde sensors)
    # Salida:    /neuracar/user_command  →  esp32_actuadores_node
    nodes.append(Node(
        package='neuracar_perception',
        executable='velocity_pid_node',
        name='velocity_pid_node',
        output='screen',
        parameters=[{
            'gain_scheduling': True,   # ajusta kp/ki/kd según velocidad objetivo
            'max_throttle':    1.0,
            'max_rate':        2.0,    # rampa máxima de throttle [throttle/s]
            'v_deadband':      0.05,   # velocidades < 0.05 m/s → parada completa
            'freq_hz':         50.0,   # frecuencia del loop PID [Hz]
            # Mezcla de LUTs recta/curva según steering
            'steer_lut_start': 0.25,
            'steer_lut_full':  0.90,
            # Calibración salto ESC — recta (lut_recta_20260607)
            'straight_stable_min_v':        0.756,
            'straight_stable_min_throttle': 0.661,
            # Calibración salto ESC — curva (pista5)
            'curve_stable_min_v':           0.590,
            'curve_stable_min_throttle':    0.650,
        }],
    ))

    # ── Lane detector ────────────────────────────────────────────────────
    # Detecta carriles desde la RealSense D415.
    # Requiere sensors.launch.py con camera:=true.
    # Activar cuando se use stanley_lane_follower_node.
    if lane_detection:
        nodes.append(Node(
            package='neuracar_perception',
            executable='lane_detector_node',
            name='lane_detector_node',
            output='screen',
        ))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'obstacle_stop', default_value='true',
            description='Lanzar obstacle_detector_node (true/false)'),
        DeclareLaunchArgument(
            'lane_detection', default_value='false',
            description='Lanzar lane_detector_node — requiere camera:=true en sensors.launch'),
        DeclareLaunchArgument(
            'distance_threshold', default_value='0.50',
            description='Distancia mínima al obstáculo LiDAR [m]'),
        DeclareLaunchArgument(
            'reset_yaw_on_start', default_value='true',
            description='Calibrar yaw IMU en la primera lectura'),
        DeclareLaunchArgument(
            'debug_lidar', default_value='false',
            description='Activar logs de debug del obstacle_detector'),
        OpaqueFunction(function=launch_setup),
    ])