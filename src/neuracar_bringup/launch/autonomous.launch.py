"""
autonomous.launch.py — NeuraCar
══════════════════════════════════════════════════════════════════
Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026

Full autonomous stack: includes sensors.launch.py +
perception.launch.py + selected path-tracking controller.
The dashboard must be launched separately and has an independent
lifecycle:
  ros2 run neuracar_control dashboard_node

Nodes launched:
  (via sensors.launch.py)     All hardware drivers
  (via perception.launch.py)  Odometry, obstacle detector, PID
  neuracar_control/stanley_controller_node     if controller:=stanley
  neuracar_control/pure_pursuit_node           if controller:=pure_pursuit
  neuracar_control/stanley_lane_follower_node  if controller:=lane_follower

Parameters:
  controller         (string, stanley):        stanley|pure_pursuit|lane_follower
  run_name           (string, vuelta_05_5cm):  CSV trajectory (no extension)
  obstacle_stop      (bool,   true):           Enable obstacle detector
  stop_on_obstacle   (bool,   false):          true=end run, false=pause/resume
  distance_threshold (float,  0.50):           LiDAR stop distance [m]
  lane_detection     (bool,   false):          Enable lane detector
  debug_lidar        (bool,   false):          Obstacle detector debug logs
  speed              (float,  0.50):           Straight speed [m/s]
  speed_curve        (float,  0.55):           Curve speed [m/s]
  loop               (bool,   false):          Repeat trajectory at goal
  max_loops          (int,    1):              Max laps (0 = infinite)
  — Stanley only —
  k                  (float,  0.80):           CTE gain
  k_soft             (float,  0.50):           Low-speed softening
  heading_lookahead  (float,  0.30):           Heading preview distance [m]
  — Pure Pursuit only —
  lookahead          (float,  0.40):           Lookahead distance Lf [m]
══════════════════════════════════════════════════════════════════
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

    # ── autonomous controller ─────────────────────────────────────────────
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

                'k':                 k,
                'k_soft':            k_soft,
                'heading_lookahead': heading_lookahead,

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

                'lookahead':        lookahead,

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
        DeclareLaunchArgument(
            'controller', default_value='stanley',
            description='Controlador autónomo: stanley | pure_pursuit | lane_follower'),
        DeclareLaunchArgument(
            'run_name', default_value='vuelta_05_5cm',
            description='Nombre del CSV de trayectoria (sin extensión)'),

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

        DeclareLaunchArgument(
            'lane_detection', default_value='false',
            description='Activar lane_detector — se activa automáticamente con controller:=lane_follower'),

        DeclareLaunchArgument(
            'speed', default_value='0.50',
            description='Velocidad en recta [m/s]'),
        DeclareLaunchArgument(
            'speed_curve', default_value='0.55',
            description='Velocidad en curva [m/s]'),

        DeclareLaunchArgument(
            'loop', default_value='false',
            description='Repetir trayectoria al llegar a la meta'),
        DeclareLaunchArgument(
            'max_loops', default_value='1',
            description='Número de vueltas (0 = infinito)'),

        DeclareLaunchArgument(
            'k', default_value='0.80',
            description='Ganancia Stanley — corrección CTE'),
        DeclareLaunchArgument(
            'k_soft', default_value='0.50',
            description='Suavizado Stanley a baja velocidad'),
        DeclareLaunchArgument(
            'heading_lookahead', default_value='0.30',
            description='Metros adelante para calcular heading del path [m]'),

        DeclareLaunchArgument(
            'lookahead', default_value='0.40',
            description='Distancia lookahead Pure Pursuit [m]'),

        OpaqueFunction(function=launch_setup),
    ])