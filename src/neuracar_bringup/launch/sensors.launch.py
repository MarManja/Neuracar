from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

#  ros2 launch realsense2_camera rs_launch.py - Opción para lanzar solo la cámara
#  ros2 launch rplidar_ros rplidar_a3_launch.py - Opción para lanzar solo el LiDAR

# Opciones directamente de este launch:
#  ros2 launch neuracar_bringup sensors.launch.py teensy:=false - sin Teensy
#  ros2 launch neuracar_bringup sensors.launch.py camera:=false - sin cámara
#  ros2 launch neuracar_bringup sensors.launch.py lidar:=false - sin LiDAR
#  ros2 launch neuracar_bringup sensors.launch.py auto_shutdown:=false - sin apagado automático (útil en desarrollo)

def generate_launch_description():

    arg_camera = DeclareLaunchArgument(
        'camera', default_value='true',
        description='Lanzar la cámara RealSense D415'
    )
    arg_lidar = DeclareLaunchArgument(
        'lidar', default_value='true',
        description='Lanzar el LiDAR RPLidar A2M12'
    )
    arg_teensy = DeclareLaunchArgument(
        'teensy', default_value='true',
        description='Lanzar el puente serial con la Teensy'
    )
    arg_auto_shutdown = DeclareLaunchArgument(
        'auto_shutdown', default_value='true',
        description='Apagar Jetson automaticamente si bateria critica. False en desarrollo.'
    )

    
    realsense_launch = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )

    camera_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch),
        launch_arguments={
            'camera_name': 'camera',
            'camera_namespace': 'camera',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            # reducción de resolución para ahorrar CPU en la Jetson:
            'color_width':      '640',
            'color_height':     '480',
            'color_fps':        '30',
            'depth_width':      '640',
            'depth_height':     '480',
            'depth_fps':        '30',
        }.items(),
        condition=IfCondition(LaunchConfiguration('camera')),
    )

    lidar_node = Node(
        package='rplidar_ros',
        executable='rplidar_node',
        name='rplidar_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('lidar')),
        parameters=[{
            'channel_type': 'serial',
            'serial_port': '/dev/lidar',
            'serial_baudrate': 256000,
            'frame_id': 'laser',
            'inverted': False,
            'angle_compensate': True,
            'scan_mode': 'Sensitivity'
        }]
    )

    teensy_node = Node(
        package='neuracar_sensors',
        executable='teensy_bridge_node',
        name='teensy_bridge',
        output='screen',
        condition=IfCondition(LaunchConfiguration('teensy')),
        parameters=[{
            'port':         '/dev/teensy',
            'baudrate':     921600,
            'wheel_radius': 0.033, # metros 
            'gear_ratio':   9.5, # ratio encoder-motor → rueda
            'auto_shutdown': LaunchConfiguration('auto_shutdown'),
        }]
    )

    return LaunchDescription([
        arg_camera,
        arg_lidar,
        arg_teensy,
        arg_auto_shutdown,
        camera_node,
        lidar_node,
        teensy_node,
    ])
