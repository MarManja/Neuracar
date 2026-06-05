"""
control.launch.py — Neuracar
================================
Uso:
  ros2 launch neuracar_bringup control.launch.py                      # teclado
  ros2 launch neuracar_bringup control.launch.py controller:=xbox     # Xbox
  ros2 launch neuracar_bringup control.launch.py controller:=ps       # PlayStation

Opciones:
  obstacle_stop:=false        deshabilitar parada automática
  distance_threshold:=0.5     umbral LiDAR [m]
  joy_device_id:=0            dispositivo joy (/dev/input/jsX)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    controller     = LaunchConfiguration('controller').perform(context)
    obstacle_stop  = LaunchConfiguration('obstacle_stop').perform(context).lower() == 'true'
    dist_thr       = float(LaunchConfiguration('distance_threshold').perform(context))
    joy_device     = int(LaunchConfiguration('joy_device_id').perform(context))

    nodes = []

    # ── Joy node (Xbox / PS necesitan el driver de joystick) ─────────
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

    # ── Controlador seleccionado ──────────────────────────────────────
    if controller == 'xbox':
        nodes.append(Node(
            package='neuracar_control',
            executable='remote_control_xbox_node',
            name='xbox_control',
            output='screen',
        ))
    elif controller == 'ps':
        nodes.append(Node(
            package='neuracar_control',
            executable='remote_control_ps_node',
            name='ps_control',
            output='screen',
        ))
    else:
        nodes.append(Node(
            package='neuracar_control',
            executable='keyboard_control_node',
            name='keyboard_control',
            output='screen',
        ))

    # ── Obstacle detector (publica /neuracar/lidar/obstacle_alert) ────
    # Los nodos de control ya suscriben a ese tópico internamente.
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
                'lidar_front_offset_rad': 0.0,
                'lidar_rear_offset_rad':  3.14159,   # π = 180° desde el frente
                'debug_mode':             False,
            }],
        ))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'controller', default_value='keyboard',
            description='keyboard | xbox | ps'),
        DeclareLaunchArgument(
            'obstacle_stop', default_value='true',
            description='Parada automática con LiDAR'),
        DeclareLaunchArgument(
            'distance_threshold', default_value='0.35',
            description='Distancia mínima al obstáculo [m]'),
        DeclareLaunchArgument(
            'joy_device_id', default_value='0',
            description='ID dispositivo joy'),
        OpaqueFunction(function=launch_setup),
    ])