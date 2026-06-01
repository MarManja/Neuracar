#!/usr/bin/env python3
"""
=======================================================================
 Lane Detector — Neuracar  (RealSense D415)
 Proyecto: Neuracar / Smart Mobility
-----------------------------------------------------------------------
 Detecta la línea amarilla de la pista Quanser mediante segmentación
 HSV y publica el error lateral normalizado para el controlador
 Stanley Lane Follower.

 Tópicos RealSense D415:
   /camera/color/image_raw         (sensor_msgs/Image)        30 Hz
   /camera/color/image_raw/compressed  (sensor_msgs/CompressedImage)

 Suscribe (configurable por parámetro):
   /camera/color/image_raw  OR  /camera/color/image_raw/compressed

 Publica:
   /neuracar/lane_error  (geometry_msgs/Vector3Stamped)
       vector.x = cross-track error normalizado [-1, 1]
                  negativo = línea a la izquierda del centro
                  positivo = línea a la derecha del centro
       vector.y = confianza de detección [0, 1]
       vector.z = cx en píxeles (debug)

   /neuracar/lane_image  (sensor_msgs/CompressedImage)   [debug]
       Imagen anotada con ROI y centroide detectado

 Parámetros ROS2:
   use_compressed  (bool)  — usar imagen comprimida  [default: False]
   roi_top         (float) — fracción superior de ROI [default: 0.55]
   target_x_ratio  (float) — x objetivo normalizado   [default: 0.5]
   min_area        (int)   — área mínima de blob px²   [default: 3000]
   max_cx_jump     (int)   — salto máximo de cx px     [default: 150]
   publish_debug   (bool)  — publicar imagen debug     [default: True]

 Rangos HSV para línea amarilla (pista Quanser):
   H: 15–35  S: 80–255  V: 80–255
   (ajusta con parámetros hsv_low_h/s/v, hsv_high_h/s/v si es necesario)
=======================================================================
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Image, CompressedImage

# RealSense D415: imagen a color — NO requiere pipeline GStreamer
# Solo se usa cv_bridge o decodificación manual con numpy


class LaneDetector(Node):

    def __init__(self):
        super().__init__('lane_detector')

        # ── Parámetros ─────────────────────────────────────────────
        self.declare_parameter('use_compressed',  False)
        self.declare_parameter('roi_top',         0.55)
        self.declare_parameter('target_x_ratio',  0.50)
        self.declare_parameter('min_area',        3000)
        self.declare_parameter('max_cx_jump',     150)
        self.declare_parameter('publish_debug',   True)
        # Rango HSV — línea amarilla pista Quanser
        self.declare_parameter('hsv_low_h',   15)
        self.declare_parameter('hsv_low_s',   80)
        self.declare_parameter('hsv_low_v',   80)
        self.declare_parameter('hsv_high_h',  35)
        self.declare_parameter('hsv_high_s',  255)
        self.declare_parameter('hsv_high_v',  255)

        self._use_compressed  = self.get_parameter('use_compressed').value
        self._roi_top         = self.get_parameter('roi_top').value
        self._target_x_ratio  = self.get_parameter('target_x_ratio').value
        self._min_area        = self.get_parameter('min_area').value
        self._max_cx_jump     = self.get_parameter('max_cx_jump').value
        self._publish_debug   = self.get_parameter('publish_debug').value

        self._hsv_low  = np.array([
            self.get_parameter('hsv_low_h').value,
            self.get_parameter('hsv_low_s').value,
            self.get_parameter('hsv_low_v').value,
        ])
        self._hsv_high = np.array([
            self.get_parameter('hsv_high_h').value,
            self.get_parameter('hsv_high_s').value,
            self.get_parameter('hsv_high_v').value,
        ])

        # ── Estado ─────────────────────────────────────────────────
        self._last_cx    = None
        self._lost_count = 0
        self._MAX_LOST   = 30

        # ── Publishers ─────────────────────────────────────────────
        self._pub_error = self.create_publisher(
            Vector3Stamped, '/neuracar/lane_error', 10)

        if self._publish_debug:
            self._pub_img = self.create_publisher(
                CompressedImage, '/neuracar/lane_image', qos_profile_sensor_data)

        # ── Subscriber (imagen RealSense D415) ─────────────────────
        if self._use_compressed:
            self.create_subscription(
                CompressedImage,
                '/camera/color/image_raw/compressed',
                self._compressed_cb,
                qos_profile_sensor_data,
            )
            self.get_logger().info('Suscrito a imagen comprimida D415')
        else:
            self.create_subscription(
                Image,
                '/camera/color/image_raw',
                self._image_cb,
                qos_profile_sensor_data,
            )
            self.get_logger().info('Suscrito a imagen raw D415')

        self.get_logger().info('=== Lane Detector iniciado ===')
        self.get_logger().info(
            f'  HSV bajo:  {self._hsv_low.tolist()}')
        self.get_logger().info(
            f'  HSV alto: {self._hsv_high.tolist()}')
        self.get_logger().info(
            f'  ROI top: {self._roi_top*100:.0f}%  |  target_x: {self._target_x_ratio*100:.0f}%')

    # ── Callbacks de imagen ────────────────────────────────────────
    def _image_cb(self, msg: Image):
        """Convierte sensor_msgs/Image a numpy BGR."""
        # Decodificación manual sin cv_bridge
        dtype = np.uint8
        frame = np.frombuffer(msg.data, dtype=dtype).reshape(
            msg.height, msg.width, -1)
        # D415 publica en bgr8 o rgb8
        if msg.encoding == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._process(frame, msg.header.stamp)

    def _compressed_cb(self, msg: CompressedImage):
        """Decodifica imagen comprimida JPEG."""
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame   = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        self._process(frame, msg.header.stamp)

    # ── Procesamiento de carril ────────────────────────────────────
    def _process(self, frame: np.ndarray, stamp):
        h, w = frame.shape[:2]

        # ── ROI: solo la mitad inferior de la imagen ──────────────
        roi_y = int(h * self._roi_top)
        roi   = frame[roi_y:h, :]

        # ── Segmentación HSV ──────────────────────────────────────
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_low, self._hsv_high)

        # ── Morfología para limpiar ruido ─────────────────────────
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.erode(mask,  kernel, iterations=1)
        mask   = cv2.dilate(mask, kernel, iterations=2)

        # ── Centroide ─────────────────────────────────────────────
        M    = cv2.moments(mask)
        area = M['m00']

        valid = False
        cx    = None
        confidence = 0.0

        if area > self._min_area:
            cx_candidate = int(M['m10'] / area)

            if (self._last_cx is None or
                    abs(cx_candidate - self._last_cx) < self._max_cx_jump):
                cx         = cx_candidate
                valid      = True
                confidence = min(1.0, area / (self._min_area * 10))
            else:
                self.get_logger().warn(
                    f'Salto brusco ignorado: cx={cx_candidate} last={self._last_cx}',
                    throttle_duration_sec=0.5)

        # ── Publicar error lateral ────────────────────────────────
        err_msg = Vector3Stamped()
        err_msg.header.stamp    = stamp
        err_msg.header.frame_id = 'camera_color_optical_frame'

        if valid:
            self._lost_count = 0
            self._last_cx    = cx
            target_x = int(w * self._target_x_ratio)
            # Error normalizado: [-1, 1]  positivo = línea a la derecha
            error = (cx - target_x) / float(w / 2)
            err_msg.vector.x = float(error)
            err_msg.vector.y = confidence
            err_msg.vector.z = float(cx)

            self.get_logger().info(
                f'cx={cx} err={error:+.3f} area={int(area)} conf={confidence:.2f}',
                throttle_duration_sec=0.3)
        else:
            self._lost_count += 1
            if self._lost_count > self._MAX_LOST:
                self._last_cx = None
            # Mantiene el último error conocido con confianza 0
            err_msg.vector.x = 0.0
            err_msg.vector.y = 0.0
            err_msg.vector.z = -1.0  # indicador de "sin detección"

            self.get_logger().warn(
                f'Línea no detectada ({self._lost_count})',
                throttle_duration_sec=0.5)

        self._pub_error.publish(err_msg)

        # ── Imagen de debug ───────────────────────────────────────
        if self._publish_debug:
            debug = frame.copy()
            # Dibujar ROI
            cv2.rectangle(debug, (0, roi_y), (w, h), (0, 255, 0), 2)
            # Dibujar máscara sobre ROI (canal verde)
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_color[:, :, 0] = 0  # quita canal azul
            mask_color[:, :, 2] = 0  # quita canal rojo
            debug[roi_y:h, :] = cv2.addWeighted(
                debug[roi_y:h, :], 0.6, mask_color, 0.4, 0)
            # Dibujar centroide
            if valid:
                cy_roi = int(M['m01'] / area)
                cv2.circle(debug, (cx, roi_y + cy_roi), 10, (0, 0, 255), -1)
                cv2.line(debug, (int(w * self._target_x_ratio), roi_y),
                         (int(w * self._target_x_ratio), h), (255, 0, 0), 2)

            _, buf = cv2.imencode('.jpg', debug, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_msg = CompressedImage()
            img_msg.header.stamp    = stamp
            img_msg.header.frame_id = 'camera_color_optical_frame'
            img_msg.format          = 'jpeg'
            img_msg.data            = buf.tobytes()
            self._pub_img.publish(img_msg)


# ────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print('Lane Detector detenido.')


if __name__ == '__main__':
    main()