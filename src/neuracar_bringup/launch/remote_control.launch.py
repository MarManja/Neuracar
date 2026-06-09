"""
remote_control_launch.py — Neuracar
=====================================
Lanza el stack completo de control remoto: controlador + velocity PID
+ obstacle detector.

Uso:
  ros2 launch neuracar_bringup remote_control.launch.py                   # teclado
  ros2 launch neuracar_bringup remote_control.launch.py controller:=xbox  # Xbox
  ros2 launch neuracar_bringup remote_control.launch.py controller:=ps    # PlayStation

Opciones:
  controller:=keyboard|xbox|ps    controlador a usar (default: keyboard)
  obstacle_stop:=true|false       habilitar parada automática por LiDAR (default: true)
  distance_threshold:=0.50        umbral de detección LiDAR [m]
  max_speed:=0.5                  velocidad máxima del joystick/teclado [m/s]
  joy_device_id:=0                dispositivo joy (/dev/input/jsX)

Arquitectura (todos los controladores son consistentes — v2.0):

  keyboard / xbox / ps
        │
        ├─ /neuracar/cmd_velocity  [Float32, m/s]
        └─ /neuracar/cmd_steering  [Float32, -1..1]
                    │
           velocity_pid_node       ← feedback: /neuracar/wheel_speed
                    │
        /neuracar/user_command  [Vector3Stamped]
                    │
           esp32_actuadores_node

El obstacle_detector publica alertas LiDAR que cada controlador consume
directamente para bloquear el setpoint de velocidad antes de enviarlo al PID.

Nota: lanzar DESPUÉS de sensors.launch.py únicamente.
      NO requiere perception.launch.py — este launch es autosuficiente para teleop.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    controller    = LaunchConfiguration('controller').perform(context)
    obstacle_stop = LaunchConfiguration('obstacle_stop').perform(context).lower() == 'true'
    dist_thr      = float(LaunchConfiguration('distance_threshold').perform(context))
    max_speed     = float(LaunchConfiguration('max_speed').perform(context))
    joy_device    = int(LaunchConfiguration('joy_device_id').perform(context))

    nodes = []

    # ── Joy driver (solo para xbox y ps) ────────────────────────────────
    if controller in ('xbox', 'ps'):
        nodes.append(Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
            parameters=[{
                'device_id':       joy_device,
                'deadzone':        0.05,
                'autorepeat_rate': 20.0,
            }],
        ))

    # ── Controlador seleccionado ─────────────────────────────────────────
    # Todos publican /neuracar/cmd_velocity [m/s] y /neuracar/cmd_steering.
    # El velocity_pid_node (abajo) convierte esos setpoints en throttle real.
    if controller == 'xbox':
        nodes.append(Node(
            package='neuracar_control',
            executable='remote_control_xbox_node',
            name='xbox_control',
            output='screen',
            parameters=[{'max_speed': max_speed}],
        ))
    elif controller == 'ps':
        nodes.append(Node(
            package='neuracar_control',
            executable='remote_control_ps_node',
            name='ps_control',
            output='screen',
            parameters=[{'max_speed': max_speed}],
        ))
    else:
        # Teclado — v2.0: publica cmd_velocity/cmd_steering igual que xbox/ps
        nodes.append(Node(
            package='neuracar_control',
            executable='keyboard_control_node',
            name='keyboard_control',
            output='screen',
            parameters=[{'max_speed': max_speed}],
        ))

    # ── Velocity PID ─────────────────────────────────────────────────────
    # Convierte cmd_velocity [m/s] en throttle real usando feedforward (LUT)
    # + PID de corrección. Necesario para los tres controladores porque todos
    # publican en cmd_velocity, no en user_command directamente.
    #
    # Entradas:  /neuracar/cmd_velocity, /neuracar/cmd_steering,
    #            /neuracar/wheel_speed  (feedback del encoder, desde sensors)
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
            'steer_lut_start': 0.25,   # steering a partir del que se mezcla LUT curva
            'steer_lut_full':  0.90,   # steering en que se usa 100% LUT curva
            # Calibración salto ESC — recta (lut_recta_20260607)
            'straight_stable_min_v':        0.756,
            'straight_stable_min_throttle': 0.661,
            # Calibración salto ESC — curva (pista5)
            'curve_stable_min_v':           0.590,
            'curve_stable_min_throttle':    0.650,
        }],
    ))

    # ── Obstacle detector ────────────────────────────────────────────────
    # Procesa el LiDAR y publica alertas de obstáculo frontal y trasero.
    # Cada controlador se suscribe a esas alertas y pone su cmd_velocity a 0
    # cuando hay un obstáculo en la dirección de movimiento.
    #
    # lidar_front_offset_rad = π  →  frente del Neuracar (cable atrás = 0 rad)
    # lidar_rear_offset_rad  = 0  →  trasero del Neuracar
    if obstacle_stop:
        nodes.append(Node(
            package='neuracar_perception',
            executable='obstacle_detector_node',
            name='obstacle_detector',
            output='screen',
            parameters=[{
                'distance_threshold':     dist_thr,
                'angle_range_low_deg':    22.5,    # cono a baja velocidad [°]
                'angle_range_high_deg':   30.0,    # cono a alta velocidad [°]
                'velocity_threshold':     1.0,     # m/s para cambiar cono
                'lidar_front_offset_rad': 3.14159, # π = frente (cable atrás)
                'lidar_rear_offset_rad':  0.0,     # 0 = trasero (donde está el cable)
                'debug_mode':             False,
            }],
        ))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'controller', default_value='keyboard',
            description='Controlador: keyboard | xbox | ps'),
        DeclareLaunchArgument(
            'obstacle_stop', default_value='true',
            description='Parada automática con LiDAR (true/false)'),
        DeclareLaunchArgument(
            'distance_threshold', default_value='0.50',
            description='Distancia mínima al obstáculo [m] — +0.15 m de margen extra'),
        DeclareLaunchArgument(
            'max_speed', default_value='0.5',
            description='Velocidad máxima del controlador [m/s]'),
        DeclareLaunchArgument(
            'joy_device_id', default_value='0',
            description='ID del dispositivo joy (/dev/input/jsX)'),
        OpaqueFunction(function=launch_setup),
    ])