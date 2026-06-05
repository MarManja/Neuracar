#!/usr/bin/env python3
"""
=======================================================================
 Path Recorder Node — Neuracar  v2.0
 Proyecto: Neuracar / Smart Mobility
-----------------------------------------------------------------------
 Graba la trayectoria del robot desde /neuracar/odometry.

 Cambios v2.0:
   - Graba TODOS los puntos que llegan (sin filtro de distancia mínima),
     igual que el grabador del QCar — el carro puede estar quieto al
     inicio sin perder el punto de arranque.
   - Parámetro downsample_dist: distancia mínima al GUARDAR en CSV
     (post-procesamiento, no filtra en tiempo real). Default 0.0 = sin filtro.
   - Imprime resumen de densidad al guardar para ayudar a elegir el
     lookahead correcto en el controlador.

 Suscribe:
   /neuracar/odometry  (nav_msgs/Odometry)

 Salida CSV:
   ~/Workspaces/Neuracar/src/neuracar_control/data/trajectories/<run_name>.csv
   Columnas: x, y, theta   [m, m, rad]

 Parámetros ROS2:
   run_name        (str)   — nombre del archivo sin extensión
                              default: 'track_YYYYMMDD_HHMMSS'
   downsample_dist (float) — distancia mínima entre puntos al guardar [m]
                              0.0 = sin downsample (guarda todo)
                              0.05 = cada 5 cm (recomendado para Pure Pursuit)
                              0.02 = cada 2 cm  (recomendado para Stanley)

 Uso:
   ros2 run neuracar_control path_recorder --ros-args -p run_name:=vuelta_04
   ros2 run neuracar_control path_recorder \
     --ros-args -p run_name:=vuelta_04 -p downsample_dist:=0.05
=======================================================================
"""

import csv
import math
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def resolve_data_dir() -> str:
    """
    Carpeta de trayectorias de referencia:
      ~/Workspaces/Neuracar/src/neuracar_control/data/trajectories/
    Si no existe, la crea (equivalente a mkdir -p).
    """
    pkg_src = os.path.expanduser(
        "~/Workspaces/Neuracar/src/neuracar_control"
    )
    if os.path.isdir(pkg_src):
        base = os.path.join(pkg_src, "data", "trajectories")
        os.makedirs(base, exist_ok=True)  # equivale a mkdir -p
        return base

    try:
        from ament_index_python.packages import get_package_share_directory
        get_package_share_directory("neuracar_control")
        base = os.path.join(
            os.path.expanduser("~/Workspaces/Neuracar/src"),
            "neuracar_control", "data", "trajectories"
        )
        os.makedirs(base, exist_ok=True)  # equivale a mkdir -p
        return base
    except Exception:
        pass

    fallback = os.path.expanduser(
        "~/Workspaces/Neuracar/src/neuracar_control/data/trajectories"
    )
    os.makedirs(fallback, exist_ok=True)  # equivale a mkdir -p
    return fallback


def yaw_from_quaternion(qx, qy, qz, qw) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
    return math.atan2(siny, cosy)


def dist2d(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def downsample(points, min_dist: float):
    """Filtra puntos manteniendo solo los que están al menos min_dist del anterior."""
    if min_dist <= 0.0 or not points:
        return points
    out = [points[0]]
    for p in points[1:]:
        if dist2d((p[0], p[1]), (out[-1][0], out[-1][1])) >= min_dist:
            out.append(p)
    return out


class PathRecorder(Node):

    def __init__(self):
        super().__init__('path_recorder')

        # ── Parámetros ─────────────────────────────────────────────
        self.declare_parameter('run_name',        '')
        self.declare_parameter('downsample_dist', 0.0)  # sin filtro por defecto

        run_name = self.get_parameter('run_name').value
        if not run_name:
            run_name = 'track_' + datetime.now().strftime('%Y%m%d_%H%M%S')

        self._ds_dist = self.get_parameter('downsample_dist').value

        # ── Ruta de salida ─────────────────────────────────────────
        self._data_dir = resolve_data_dir()
        self._outfile  = os.path.join(self._data_dir, f'{run_name}.csv')

        if os.path.exists(self._outfile):
            suffix = datetime.now().strftime('_%H%M%S')
            self._outfile = os.path.join(
                self._data_dir, f'{run_name}{suffix}.csv')

        # ── Estado — guarda TODOS los puntos sin filtro ────────────
        self._points = []   # lista de (x, y, theta)

        # ── Subscriber ─────────────────────────────────────────────
        self.create_subscription(
            Odometry, '/neuracar/odometry', self._odom_cb, 10)

        ds_str = (f'{self._ds_dist*100:.0f} cm'
                  if self._ds_dist > 0 else 'sin filtro (guarda todo)')
        self.get_logger().info('=' * 52)
        self.get_logger().info(' PATH RECORDER v2.0 — Neuracar')
        self.get_logger().info('=' * 52)
        self.get_logger().info(f'  Archivo       : {self._outfile}')
        self.get_logger().info(f'  Downsample    : {ds_str}')
        self.get_logger().info('  Graba desde el primer mensaje de odometría.')
        self.get_logger().info('  Puedes arrancar quieto — no se perderá el punto inicial.')
        self.get_logger().info('  Ctrl+C para guardar.')

    # ──────────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        """Guarda TODOS los puntos — sin filtro de distancia."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._points.append((x, y, theta))

        if len(self._points) % 500 == 0:
            self.get_logger().info(
                f'  {len(self._points):5d} pts | '
                f'pos=({x:.2f}, {y:.2f}) θ={math.degrees(theta):.1f}°'
            )

    # ──────────────────────────────────────────────────────────────
    def save(self):
        if not self._points:
            self.get_logger().warn('No se grabaron puntos — CSV no creado.')
            return

        raw_n = len(self._points)

        # Aplicar downsample al guardar (no en tiempo real)
        pts_to_save = downsample(self._points, self._ds_dist)
        saved_n = len(pts_to_save)

        # Calcular densidad media
        if saved_n > 1:
            total_len = sum(
                dist2d((pts_to_save[i][0], pts_to_save[i][1]),
                       (pts_to_save[i+1][0], pts_to_save[i+1][1]))
                for i in range(saved_n - 1)
            )
            mean_dist_cm = total_len / (saved_n - 1) * 100
        else:
            total_len = 0.0
            mean_dist_cm = 0.0

        try:
            with open(self._outfile, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'theta'])
                writer.writerows(pts_to_save)

            self.get_logger().info('=' * 52)
            self.get_logger().info(f'✓ CSV guardado: {self._outfile}')
            self.get_logger().info(f'  Puntos grabados (raw): {raw_n}')
            self.get_logger().info(f'  Puntos en CSV:         {saved_n}')
            self.get_logger().info(
                f'  Longitud total:        {total_len:.2f} m')
            self.get_logger().info(
                f'  Distancia media/wp:    {mean_dist_cm:.1f} cm')
            # Recomendación de lookahead para el controlador
            if mean_dist_cm > 0:
                wp_per_sec_55 = 55.0 / mean_dist_cm  # a 0.55 m/s
                self.get_logger().info(
                    f'  wp/s a speed=0.55:     {wp_per_sec_55:.0f}  '
                    f'→ lookahead recomendado ≥ {int(wp_per_sec_55*3)}')
            self.get_logger().info('=' * 52)

        except Exception as e:
            self.get_logger().error(f'Error guardando CSV: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PathRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()