# Neuracar

Este es un proyecto de navegación autónoma basado en ROS 2.

## Estructura
- **neuracar_bringup**: Archivos de lanzamiento y configuración.
- **neuracar_sensors**: Nodos de sensores (IMU, encoder, LiDAR, cámara).
- **neuracar_control**: Nodos de control para tracción, dirección y vehículo.

## Dependencias externas
- **RPLidar A2M12**: LiDAR.
- **Intel RealSense D415**: Cámara.

## Instalación
```bash
git clone https://github.com/tu_usuario/Neuracar.git
cd Neuracar
vcs import src < neuracar.repos
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
