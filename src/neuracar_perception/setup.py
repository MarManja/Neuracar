from setuptools import find_packages, setup

package_name = 'neuracar_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marml',
    maintainer_email='a01735160@tec.mx',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'odometry_node = neuracar_perception.odometry_node:main',
            'obstacle_detector_node = neuracar_perception.obstacle_detector_node:main',
            'lidar_visualizer_node = neuracar_perception.lidar_visualizer_node:main',
            'dashboard_node = neuracar_perception.dashboard_node:main',
            'lane_detector_node = neuracar_perception.lane_detector_node:main',
            'velocity_pid_node = neuracar_perception.velocity_pid_node:main',
            'pid_dashboard_node = neuracar_perception.pid_dashboard_node:main',
        ],
    },
)
