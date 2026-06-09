"""
autonomous_launch.py — Neuracar
=================================
Launch único para modo autónomo. Levanta toda la pila necesaria:
sensors → perception (odometry + obstacle + velocity_pid) → controlador.

Uso:
  ros2 launch neuracar_bringup autonomous.launch.py controller:=stanley  run_name:=vuelta_05_5cm
  ros2 launch neuracar_bringup autonomous.launch.py controller:=pure_pursuit
  ros2 launch neuracar_bringup autonomous.launch.py controller:=stanley  loop:=true  max_loops:=3
  ros2 launch neuracar_bringup autonomous.launch.py controller:=lane_follower  lane_detection:=true

El dashboard se lanza SIEMPRE en una terminal separada y se cierra manualmente:
  ros2 run neuracar_control dashboard_node
El dashboard tiene ciclo de vida independiente — sigue corriendo después del
Ctrl+C del controlador para que puedas ver la trayectoria y el análisis final.

Opciones generales:
  controller:=stanley|pure_pursuit|lane_follower   controlador autónomo (default: stanley)
  run_name:=vuelta_05_5cm                          trayectoria CSV a seguir
  obstacle_stop:=true|false                        parada por LiDAR (default: true)
  stop_on_obstacle:=false                          true=termina prueba, false=pausa y reanuda
  distance_threshold:=0.50                         umbral LiDAR [m]
  lane_detection:=false                            activar lane_detector (solo con lane_follower)
  debug_lidar:=false                               logs de debug del obstacle_detector

Opciones del controlador (stanley y pure_pursuit):
  speed:=0.50              velocidad en recta [m/s]
  speed_curve:=0.55        velocidad en curva [m/s]
  loop:=false              repetir trayectoria al llegar a la meta
  max_loops:=1             número de vueltas (0 = infinito)

Opciones solo Stanley:
  k:=0.80                  ganancia Stanley (corrección CTE)
  k_soft:=0.50             suavizado a baja velocidad
  heading_lookahead:=0.30  metros adelante para calcular heading del path [m]

Opciones solo Pure Pursuit:
  lookahead:=0.40          distancia lookahead [m]
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def launch_setup(context, *args, **kwargs):
    controller       = LaunchConfiguration('controller').perform(context)
    run_name         = LaunchConfiguration('run_name').perform(context)
    obstacle_stop    = LaunchConfiguration('obstacle_stop').perform(context)
    stop_on_obstacle = LaunchConfiguration('stop_on_obstacle').perform(context).lower() == 'true'
    dist_thr         = LaunchConfiguration('distance_threshold').perform(context)
    lane_detection   = LaunchConfiguration('lane_detection').perform(context)
    debug_lidar      = LaunchConfiguration('debug_lidar').perform(context)
    speed            = float(LaunchConfiguration('speed').perform(context))
    speed_curve      = float(LaunchConfiguration('speed_curve').perform(context))
    loop             = LaunchConfiguration('loop').perform(context).lower() == 'true'
    max_loops        = int(LaunchConfiguration('max_loops').perform(context))
    k                = float(LaunchConfiguration('k').perform(context))
    k_soft           = float(LaunchConfiguration('k_soft').perform(context))
    heading_lookahead= float(LaunchConfiguration('heading_lookahead').perform(context))
    lookahead        = float(LaunchConfiguration('lookahead').perform(context))

    bringup_dir = get_package_share_directory('neuracar_bringup')

    nodes = []

    # ── sensors.launch ───────────────────────────────────────────────────
    # Levanta todo el hardware: encoder, IMU, LiDAR, cámara (si la necesitas).
    # Si el controlador es lane_follower activa la cámara, si no la deja apagada
    # para no gastar recursos.
    camera_on = 'true' if controller == 'lane_follower' else 'false'
    nodes.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'sensors.launch.py')),
        launch_arguments={
            'camera': camera_on,
            'lidar':  'true',
            'micro':  'true',
        }.items(),
    ))

    # ── perception.launch ────────────────────────────────────────────────
    # Levanta odometry + obstacle_detector + velocity_pid.
    # Si es lane_follower activa también el lane_detector.
    nodes.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'perception.launch.py')),
        launch_arguments={
            'obstacle_stop':    obstacle_stop,
            'distance_threshold': dist_thr,
            'lane_detection':   lane_detection,
            'debug_lidar':      debug_lidar,
        }.items(),
    ))

    # ── Controlador autónomo ─────────────────────────────────────────────
    # Stanley, Pure Pursuit o Lane Follower.
    # Todos publican en /neuracar/cmd_velocity y /neuracar/cmd_steering.
    # El velocity_pid_node (en perception) cierra el lazo hacia el ESP32.

    if controller == 'stanley':
        nodes.append(Node(
            package='neuracar_control',
            executable='stanley_controller_node',
            name='stanley_controller',
            output='screen',
            parameters=[{
                'run_name':          run_name,
                'speed':             speed,
                'speed_curve':       speed_curve,
                'loop':              loop,
                'max_loops':         max_loops,
                'stop_on_obstacle':  stop_on_obstacle,
                # Parámetros Stanley
                'k':                 k,
                'k_soft':            k_soft,
                'heading_lookahead': heading_lookahead,
                # Defaults físicos calibrados
                'wheelbase':         0.256,
                'steering_sign':    -1.0,
                'steering_cmd_gain': 1.0,
                'max_steer':         0.50,
                'min_straight_speed': 0.45,
                'min_curve_speed':    0.55,
                'curve_steer_threshold': 0.35,
                'goal_radius':        0.25,
                'nearest_fwd_steps':  35,
                'monotonic_index':    True,
            }],
        ))

    elif controller == 'pure_pursuit':
        nodes.append(Node(
            package='neuracar_control',
            executable='pure_pursuit_node',
            name='pure_pursuit',
            output='screen',
            parameters=[{
                'run_name':         run_name,
                'speed':            speed,
                'speed_curve':      speed_curve,
                'loop':             loop,
                'max_loops':        max_loops,
                'stop_on_obstacle': stop_on_obstacle,
                # Parámetros Pure Pursuit
                'lookahead':        lookahead,
                # Defaults físicos calibrados
                'wheelbase':        0.256,
                'steering_sign':   -1.0,
                'steering_cmd_gain': 1.0,
                'max_steer':        0.50,
                'min_straight_speed': 0.45,
                'min_curve_speed':    0.55,
                'curve_steer_threshold': 0.35,
                'goal_radius':       0.25,
                'nearest_fwd_steps': 35,
                'monotonic_index':   True,
            }],
        ))

    elif controller == 'lane_follower':
        nodes.append(Node(
            package='neuracar_control',
            executable='stanley_lane_follower_node',
            name='stanley_lane_follower',
            output='screen',
            parameters=[{
                'speed':       speed,
                'speed_curve': speed_curve,
                'k':           k,
                'k_soft':      k_soft,
            }],
        ))

    else:
        raise RuntimeError(
            f'Controlador desconocido: "{controller}". '
            f'Usa stanley | pure_pursuit | lane_follower')

    return nodes


def generate_launch_description():
    return LaunchDescription([
        # ── Controlador ────────────────────────────────────────────────
        DeclareLaunchArgument(
            'controller', default_value='stanley',
            description='Controlador autónomo: stanley | pure_pursuit | lane_follower'),
        DeclareLaunchArgument(
            'run_name', default_value='vuelta_05_5cm',
            description='Nombre del CSV de trayectoria (sin extensión)'),

        # ── LiDAR / seguridad ──────────────────────────────────────────
        DeclareLaunchArgument(
            'obstacle_stop', default_value='true',
            description='Activar obstacle_detector (true/false)'),
        DeclareLaunchArgument(
            'stop_on_obstacle', default_value='false',
            description='true=termina la prueba al detectar obstáculo, false=pausa y reanuda'),
        DeclareLaunchArgument(
            'distance_threshold', default_value='0.50',
            description='Distancia mínima al obstáculo [m]'),
        DeclareLaunchArgument(
            'debug_lidar', default_value='false',
            description='Logs de debug del obstacle_detector'),

        # ── Lane follower ──────────────────────────────────────────────
        DeclareLaunchArgument(
            'lane_detection', default_value='false',
            description='Activar lane_detector — se activa automáticamente con controller:=lane_follower'),

        # ── Velocidad ──────────────────────────────────────────────────
        DeclareLaunchArgument(
            'speed', default_value='0.50',
            description='Velocidad en recta [m/s]'),
        DeclareLaunchArgument(
            'speed_curve', default_value='0.55',
            description='Velocidad en curva [m/s]'),

        # ── Vueltas ────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'loop', default_value='false',
            description='Repetir trayectoria al llegar a la meta'),
        DeclareLaunchArgument(
            'max_loops', default_value='1',
            description='Número de vueltas (0 = infinito)'),

        # ── Stanley ────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'k', default_value='0.80',
            description='Ganancia Stanley — corrección CTE'),
        DeclareLaunchArgument(
            'k_soft', default_value='0.50',
            description='Suavizado Stanley a baja velocidad'),
        DeclareLaunchArgument(
            'heading_lookahead', default_value='0.30',
            description='Metros adelante para calcular heading del path [m]'),

        # ── Pure Pursuit ───────────────────────────────────────────────
        DeclareLaunchArgument(
            'lookahead', default_value='0.40',
            description='Distancia lookahead Pure Pursuit [m]'),

        OpaqueFunction(function=launch_setup),
    ])