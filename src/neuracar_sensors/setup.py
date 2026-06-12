from setuptools import find_packages, setup

package_name = 'neuracar_sensors'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mariana Manjarrez Lima',
    maintainer_email='a01735160@tec.mx',
    description='Serial bridge nodes for NeuraCar ESP32 microcontrollers '
                '(sensing: encoder + IMU; actuation: ESC + servo).',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'esp32_sensores_node = neuracar_sensors.esp32_sensores_node:main',
            'esp32_actuadores_node = neuracar_sensors.esp32_actuadores_node:main',
        ],
    },
)
