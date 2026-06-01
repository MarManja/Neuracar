"""
perception.launch.py — Neuracar
================================
Lanza los nodos de percepción: odometría y (próximamente) lane detection.

Uso:
  ros2 launch neuracar_bringup perception.launch.py

Nota: lanzar DESPUÉS de sensors.launch.py para que los tópicos del bridge
estén disponibles.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    return LaunchDescription([

        # ── Odometría: encoder + IMU → pose + velocity ──────────────
        Node(
            package='neuracar_perception',
            executable='odometry_node',
            name='odometry_node',
            output='screen',
            parameters=[{
                # No tiene parámetros propios — usa los tópicos del bridge.
                # El gear_ratio y wheel_radius ya fueron aplicados por el
                # teensy_bridge_node antes de publicar /neuracar/encoder.
            }],
        ),

        # ── TF estático: imu_link → base_link ───────────────────────
        # BNO055 montado adelante: X=derecha, Y=adelante, Z=arriba
        # ROS REP-103:             X=adelante, Y=izquierda, Z=arriba
        # Rotación: -90° en Z (yaw) para alinear ejes
        # Ajustar x,y,z si el IMU no está en el centro del vehículo.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='imu_tf',
            arguments=[
                '0', '0', '0',          # traslación x y z [m]
                '-1.5708', '0', '0',    # rotación yaw pitch roll [rad]
                'base_link', 'imu_link'
            ],
        ),

        # ── TF estático: laser → base_link ──────────────────────────
        # Ajustar posición del LiDAR según tu montaje
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='laser_tf',
            arguments=[
                '0.1', '0', '0.15',    # LiDAR adelante y arriba del centro
                '0', '0', '0',
                'base_link', 'laser'
            ],
        ),

        # ── Lane detection (próximo) ─────────────────────────────────
        # Node(
        #     package='neuracar_perception',
        #     executable='lane_detection_node',
        #     name='lane_detection_node',
        #     output='screen',
        # ),
    ])