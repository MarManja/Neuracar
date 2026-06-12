"""
perception.launch.py — NeuraCar
══════════════════════════════════════════════════════════════════
Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026

Launches the perception and low-level control stack for autonomous
mode: odometry, obstacle detector, velocity PID, and optionally
the lane detector. Must be launched after sensors.launch.py.
Not required for teleop — remote_control.launch.py is self-contained.

Nodes launched:
  neuracar_perception/odometry_node         Always active
  neuracar_perception/velocity_pid_node     Always active
  neuracar_perception/obstacle_detector_node  If obstacle_stop:=true
  neuracar_perception/lane_detector_node    If lane_detection:=true

Publications:
  /neuracar/odometry                  nav_msgs/Odometry
  /neuracar/velocity                  geometry_msgs/TwistStamped
  /neuracar/lidar/obstacle_alert      std_msgs/Bool
  /neuracar/lidar/obstacle_alert_rear std_msgs/Bool
  /neuracar/user_command              geometry_msgs/Vector3Stamped

Parameters:
  obstacle_stop       (bool,  true):  Launch obstacle detector
  distance_threshold  (float, 0.50):  LiDAR stop distance [m]
  reset_yaw_on_start  (bool,  true):  Calibrate IMU yaw on first read
  lane_detection      (bool,  false): Launch lane detector
                                      (requires camera:=true)
  debug_lidar         (bool,  false): Enable obstacle detector logs
══════════════════════════════════════════════════════════════════
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

    # ── Odometry ────────────────────────────────────────────────────────
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
    nodes.append(Node(
        package='neuracar_perception',
        executable='velocity_pid_node',
        name='velocity_pid_node',
        output='screen',
        parameters=[{
            'gain_scheduling': True,   # adjusts kp/ki/kd according to target speed
            'max_throttle':    1.0,
            'max_rate':        2.0,   
            'v_deadband':      0.05, 
            'freq_hz':         50.0,  

            'steer_lut_start': 0.25,
            'steer_lut_full':  0.90,

            'straight_stable_min_v':        0.756,
            'straight_stable_min_throttle': 0.661,

            'curve_stable_min_v':           0.590,
            'curve_stable_min_throttle':    0.650,
        }],
    ))

    # ── Lane detector ────────────────────────────────────────────────────
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