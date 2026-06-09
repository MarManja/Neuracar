from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
#from launch_ros.parameter_descriptions import ParameterValue


# Opciones:
# ros2 launch neuracar_bringup sensors.launch.py micro:=false
# ros2 launch neuracar_bringup sensors.launch.py camera:=false
# ros2 launch neuracar_bringup sensors.launch.py lidar:=false
# ros2 launch neuracar_bringup sensors.launch.py port_sensores:=/dev/ttyUSB0
# ros2 launch neuracar_bringup sensors.launch.py port_actuadores:=/dev/ttyUSB0



def generate_launch_description():

    arg_camera = DeclareLaunchArgument(
        'camera',
        default_value='true',
        description='Lanzar la cámara RealSense D415'
    )

    arg_lidar = DeclareLaunchArgument(
        'lidar',
        default_value='true',
        description='Lanzar el LiDAR RPLidar A3M1'
    )

    arg_micro = DeclareLaunchArgument(
        'micro',
        default_value='true',
        description='Lanzar el puente serial con la micro'
    )

    arg_port_sensores = DeclareLaunchArgument(
        'port_sensores',
        default_value='/dev/esp32s',
        description='Puerto serial ESP32 sensores (encoder, IMU, bateria, OLED)'
    )
 
    arg_port_actuadores = DeclareLaunchArgument(
        'port_actuadores',
        default_value='/dev/esp32a',
        description='Puerto serial ESP32 actuadores (ESC, servo)'
    )
 

    camera_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera',
        namespace='camera',
        output='screen',
        condition=IfCondition(LaunchConfiguration('camera')),
        parameters=[{
            'camera_name': 'camera',
            'camera_namespace': 'camera',

            'enable_color': True,
            'enable_depth': False,
            'enable_infra': False,
            'enable_infra1': False,
            'enable_infra2': False,

            # Baja carga para Jetson
            'rgb_camera.color_profile': '640x480x15',
            'depth_module.depth_profile': '640x480x15',

            # Desactivar cosas pesadas por ahora
            'pointcloud.enable': False,
            'align_depth.enable': False,

            # Puede ayudar cuando la cámara queda en estado raro
            'initial_reset': True,

            # Evita esperar indefinidamente si no detecta cámara
            'wait_for_device_timeout': 5.0,
        }]
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

    # Nodo exclusivo de sensores — solo lee serial, sin actuadores
    sensores_node = Node(
        package='neuracar_sensors',
        executable='esp32_sensores_node',
        name='esp32_sensores_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('micro')),
        parameters=[{
            'port':     LaunchConfiguration('port_sensores'),
            'baudrate': 921600,
 
            'wheel_radius': 0.033,
            'gear_ratio':   9.2459,

        }]
    )
 
    # Nodo exclusivo de actuadores — solo escribe serial, latencia mínima
    actuadores_node = Node(
        package='neuracar_sensors',
        executable='esp32_actuadores_node',
        name='esp32_actuadores_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('micro')),
        parameters=[{
            'port':       LaunchConfiguration('port_actuadores'),
            'baudrate':   921600,
            'watchdog_s': 0.5,
        }]
    )

    return LaunchDescription([
        arg_camera,
        arg_lidar,
        arg_micro,
        arg_port_sensores,
        arg_port_actuadores,
        camera_node,
        lidar_node,
        sensores_node,
        actuadores_node,
    ])