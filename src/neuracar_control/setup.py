from setuptools import find_packages, setup

package_name = 'neuracar_control'

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
    maintainer='devel-ds',
    maintainer_email='devel-ds@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'remote_control_xbox_node = neuracar_control.remote_control_xbox_node:main',
            'remote_control_ps_node = neuracar_control.remote_control_ps_node:main',
            'keyboard_control_node = neuracar_control.keyboard_control_node:main',
            'vehicle_interface_node = neuracar_control.vehicle_interface_node:main',
            'stanley_lane_follower_node = neuracar_control.stanley_lane_follower_node:main',
            'path_recorder_node = neuracar_control.path_recorder_node:main',  
            'stanley_controller_node = neuracar_control.stanley_controller_node:main',
            'pure_pursuit_node = neuracar_control.pure_pursuit_node:main',
        ],
    },
)
