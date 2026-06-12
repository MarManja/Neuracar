"""
lidar_visualizer_node.py — NeuraCar
══════════════════════════════════════════════════════════════════
Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026

Real-time polar plot of the RPLiDAR A3M1 scan. Useful for verifying
sensor mounting, detecting angular offset, and general debug.
Does not publish any topics — visualization only.

On Jetson without display, forward X11 to your laptop:
  ssh -X devel-ds@192.168.0.217
  ros2 run neuracar_perception lidar_visualizer_node
Or run directly on your laptop if ROS_DOMAIN_ID=28 is set.

Subscriptions:
  /scan  sensor_msgs/LaserScan  (RPLiDAR A3M1, BEST_EFFORT QoS)

Publications:
  None — visualization only.

Parameters:
  scan_radius_max (float, 6.0):  Plot radius limit [m]
  update_hz       (float, 10.0): Plot refresh rate [Hz]
══════════════════════════════════════════════════════════════════
"""
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy,
                        QoSProfile, ReliabilityPolicy)
from sensor_msgs.msg import LaserScan


class LidarVisualizerNode(Node):

    def __init__(self):
        super().__init__('lidar_visualizer_node')

        self.declare_parameter('scan_radius_max', 6.0)   
        self.declare_parameter('update_hz',       10.0)  
        self._r_max = self.get_parameter('scan_radius_max').value
        hz          = self.get_parameter('update_hz').value

        qos_lidar = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            LaserScan, '/scan', self._lidar_callback, qos_lidar)

        self.create_timer(1.0 / hz, self._timer_callback)
        self._latest_scan = None

        plt.ion()
        self._fig, self._ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self._ax.set_title('Neuracar — RPLidar A3M1', va='bottom')
        self._ax.set_rmax(self._r_max)
        self._ax.set_theta_zero_location('N')   
        self._ax.set_theta_direction(-1)         

        self.get_logger().info('LiDAR visualizer iniciado — escuchando /scan')
        self.get_logger().info(
            'Si no aparece la ventana, asegúrate de tener DISPLAY configurado.')

    def _lidar_callback(self, msg: LaserScan):
        self._latest_scan = msg

    def _timer_callback(self):
        if self._latest_scan is None:
            return
        self._plot_scan(self._latest_scan)

    def _plot_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=float)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        ranges = np.clip(ranges, msg.range_min, msg.range_max)
        valid  = np.isfinite(ranges) & (ranges > msg.range_min)

        if not np.any(valid):
            self.get_logger().warn('Sin puntos válidos en el escaneo')
            return

        self._ax.clear()
        self._ax.scatter(angles[valid], ranges[valid], s=4, c='#EF9F27')
        self._ax.set_theta_zero_location('N')
        self._ax.set_theta_direction(-1)
        self._ax.set_rmax(self._r_max)
        self._ax.set_title('Neuracar — RPLidar A3M1', va='bottom')

        theta_front = np.linspace(-np.radians(30), np.radians(30), 50)
        self._ax.fill_between(
            theta_front, 0, 0.35,
            alpha=0.15, color='red', label='zona alerta 35cm'
        )

        plt.pause(0.001)

        d_min  = np.min(ranges[valid])
        d_mean = np.mean(ranges[valid])
        self.get_logger().info(
            f'Min: {d_min:.2f} m  |  Media: {d_mean:.2f} m  |  '
            f'Puntos válidos: {np.sum(valid)}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = LidarVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()