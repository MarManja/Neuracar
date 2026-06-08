#!/usr/bin/env python3
"""
=======================================================================
 Lane Detector — Neuracar  (RealSense D415)  v2.2
 Proyecto: Neuracar
-----------------------------------------------------------------------
 Detecta la línea amarilla de la pista Quanser mediante segmentación
 HSV y publica el error lateral normalizado para el controlador
 Stanley Lane Follower.

 Geometría de cámara (Neuracar):
   Altura:       131.5 mm del piso a la base de la cámara
   Inclinación:  11° hacia abajo
   Visión:       ~170 mm al frente del carro (borde externo)
                 La línea aparece en la franja 50–90% vertical de imagen

 Topic real RealSense D415 en este setup:
   /camera/camera/color/image_raw   ← usar con remapping

 Lanzar con:
   ros2 run neuracar_perception lane_detector_node --ros-args \
     -r /camera/color/image_raw:=/camera/camera/color/image_raw

 Publica:
   /neuracar/lane_error  (geometry_msgs/Vector3Stamped)
       vector.x = cross-track error normalizado [-1, 1]
       vector.y = confianza de detección [0, 1]
       vector.z = cx en píxeles (debug)

   /neuracar/lane_image  (sensor_msgs/CompressedImage)   [debug]

 Parámetros ROS2:
   use_compressed   (bool)  — usar imagen comprimida     [default: False]
   roi_top          (float) — fracción superior de ROI   [default: 0.50]
   roi_bottom       (float) — fracción inferior de ROI   [default: 0.95]
   target_x_ratio   (float) — x objetivo normalizado     [default: 0.50]
   min_area         (int)   — área mínima de blob px²    [default: 2000]
   max_area         (int)   — área máxima de blob px²    [default: 60000]
   max_cx_jump      (int)   — salto máximo de cx px      [default: 200]
   publish_debug    (bool)  — publicar imagen debug      [default: True]

 Rangos HSV para línea amarilla (pista Quanser, luz interior):
   H: 20–30  S: 120–255  V: 120–255
=======================================================================
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Image, CompressedImage


class LaneDetector(Node):

    def __init__(self):
        super().__init__('lane_detector')

        # ── Parámetros ─────────────────────────────────────────────
        self.declare_parameter('use_compressed',  False)
        self.declare_parameter('roi_top',         0.50)   # línea aparece desde el 50%
        self.declare_parameter('roi_bottom',      0.95)   # deja margen inferior
        self.declare_parameter('target_x_ratio',  0.50)
        self.declare_parameter('min_area',        2000)
        self.declare_parameter('max_area',        60000)
        self.declare_parameter('max_cx_jump',     200)
        self.declare_parameter('publish_debug',   True)
        # Rango HSV — línea amarilla pista Quanser
        self.declare_parameter('hsv_low_h',   20)
        self.declare_parameter('hsv_low_s',   120)
        self.declare_parameter('hsv_low_v',   120)
        self.declare_parameter('hsv_high_h',  30)
        self.declare_parameter('hsv_high_s',  255)
        self.declare_parameter('hsv_high_v',  255)

        self._use_compressed  = self.get_parameter('use_compressed').value
        self._roi_top         = self.get_parameter('roi_top').value
        self._roi_bottom      = self.get_parameter('roi_bottom').value
        self._target_x_ratio  = self.get_parameter('target_x_ratio').value
        self._min_area        = self.get_parameter('min_area').value
        self._max_area        = self.get_parameter('max_area').value
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

        # ── Subscriber ─────────────────────────────────────────────
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

        self.get_logger().info('=== Lane Detector v2.2 iniciado ===')
        self.get_logger().info(f'  HSV bajo:  {self._hsv_low.tolist()}')
        self.get_logger().info(f'  HSV alto:  {self._hsv_high.tolist()}')
        self.get_logger().info(
            f'  ROI: {self._roi_top*100:.0f}% – {self._roi_bottom*100:.0f}%  '
            f'target_x: {self._target_x_ratio*100:.0f}%')
        self.get_logger().info(
            f'  Área válida: [{self._min_area}, {self._max_area}] px²')

    # ── Callbacks de imagen ────────────────────────────────────────
    def _image_cb(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)
        if msg.encoding == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._process(frame, msg.header.stamp)

    def _compressed_cb(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame   = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        self._process(frame, msg.header.stamp)

    # ── Procesamiento de carril ────────────────────────────────────
    def _process(self, frame: np.ndarray, stamp):
        h, w = frame.shape[:2]

        # ── ROI configurable ──────────────────────────────────────
        roi_y_top = int(h * self._roi_top)
        roi_y_bot = int(h * self._roi_bottom)
        roi = frame[roi_y_top:roi_y_bot, :]

        # ── Segmentación HSV ──────────────────────────────────────
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_low, self._hsv_high)

        # ── Morfología ────────────────────────────────────────────
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.erode(mask,  kernel, iterations=1)
        mask   = cv2.dilate(mask, kernel, iterations=2)

        # ── Contorno más grande dentro del rango de área ──────────
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid      = False
        cx         = None
        cy_roi     = None
        best_area  = 0
        confidence = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self._min_area < area < self._max_area and area > best_area:
                M_cnt = cv2.moments(cnt)
                if M_cnt['m00'] > 0:
                    cx_candidate = int(M_cnt['m10'] / M_cnt['m00'])
                    if (self._last_cx is None or
                            abs(cx_candidate - self._last_cx) < self._max_cx_jump):
                        best_area  = area
                        cx         = cx_candidate
                        cy_roi     = int(M_cnt['m01'] / M_cnt['m00'])
                        valid      = True
                        confidence = min(1.0, area / (self._min_area * 10))
                    else:
                        self.get_logger().warn(
                            f'Salto brusco ignorado: cx={cx_candidate} '
                            f'last={self._last_cx}  area={int(area)}',
                            throttle_duration_sec=0.5)

        # ── Publicar error lateral ────────────────────────────────
        err_msg = Vector3Stamped()
        err_msg.header.stamp    = stamp
        err_msg.header.frame_id = 'camera_color_optical_frame'

        if valid:
            self._lost_count = 0
            self._last_cx    = cx
            target_x = int(w * self._target_x_ratio)
            error = (cx - target_x) / float(w / 2)
            err_msg.vector.x = float(error)
            err_msg.vector.y = confidence
            err_msg.vector.z = float(cx)

            self.get_logger().info(
                f'cx={cx}  err={error:+.3f}  area={int(best_area)}  conf={confidence:.2f}',
                throttle_duration_sec=0.3)
        else:
            self._lost_count += 1
            if self._lost_count > self._MAX_LOST:
                self._last_cx = None
            err_msg.vector.x = 0.0
            err_msg.vector.y = 0.0
            err_msg.vector.z = -1.0

            self.get_logger().warn(
                f'Línea no detectada ({self._lost_count})',
                throttle_duration_sec=0.5)

        self._pub_error.publish(err_msg)

        # ── Imagen de debug ───────────────────────────────────────
        if self._publish_debug:
            debug = frame.copy()

            # ROI activo (verde)
            cv2.rectangle(debug, (0, roi_y_top), (w, roi_y_bot), (0, 255, 0), 2)

            # Máscara amarilla superpuesta en ROI
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_color[:, :, 0] = 0
            mask_color[:, :, 2] = 0
            debug[roi_y_top:roi_y_bot, :] = cv2.addWeighted(
                debug[roi_y_top:roi_y_bot, :], 0.6, mask_color, 0.4, 0)

            # Centroide y línea objetivo
            if valid:
                cv2.circle(debug, (cx, roi_y_top + cy_roi), 10, (0, 0, 255), -1)
                cv2.line(debug,
                         (int(w * self._target_x_ratio), roi_y_top),
                         (int(w * self._target_x_ratio), roi_y_bot),
                         (255, 0, 0), 2)

            # Info en imagen
            status = f'area={int(best_area)} cx={cx}' if valid else 'NO DETECT'
            cv2.putText(debug, status, (10, roi_y_top + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

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