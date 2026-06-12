"""
remote_control.launch.py — NeuraCar
══════════════════════════════════════════════════════════════════
Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026

Teleop stack: selected controller + velocity PID + obstacle
detector. Self-contained — does not require perception.launch.py.
Must be launched after sensors.launch.py.

Nodes launched:
  joy/joy_node                              If controller:=xbox or ps
  neuracar_control/keyboard_control_node    If controller:=keyboard
  neuracar_control/remote_control_xbox_node If controller:=xbox
  neuracar_control/remote_control_ps_node   If controller:=ps
  neuracar_perception/velocity_pid_node     Always active
  neuracar_perception/obstacle_detector_node  If obstacle_stop:=true

Parameters:
  controller         (string, keyboard): keyboard|xbox|ps
  obstacle_stop      (bool,   true):     Enable LiDAR safety stop
  distance_threshold (float,  1.0):      LiDAR stop distance [m]
  max_speed          (float,  0.50):     Maximum speed [m/s]
  joy_device_id      (int,    0):        Joystick (/dev/input/jsX)
  
══════════════════════════════════════════════════════════════════
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

    # ── Joy driver (only in mood xbox y ps) ────────────────────────────────
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

    # ── Selected controller ─────────────────────────────────────────
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
        nodes.append(Node(
            package='neuracar_control',
            executable='keyboard_control_node',
            name='keyboard_control',
            output='screen',
            parameters=[{'max_speed': max_speed}],
        ))

    # ── Velocity PID ─────────────────────────────────────────────────────
   
    nodes.append(Node(
        package='neuracar_perception',
        executable='velocity_pid_node',
        name='velocity_pid_node',
        output='screen',
        parameters=[{
            'gain_scheduling': True,   
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
            'distance_threshold', default_value='1.0',
            description='Distancia mínima al obstáculo [m] — +0.15 m de margen extra'),
        DeclareLaunchArgument(
            'max_speed', default_value='0.5',
            description='Velocidad máxima del controlador [m/s]'),
        DeclareLaunchArgument(
            'joy_device_id', default_value='0',
            description='ID del dispositivo joy (/dev/input/jsX)'),
        OpaqueFunction(function=launch_setup),
    ])