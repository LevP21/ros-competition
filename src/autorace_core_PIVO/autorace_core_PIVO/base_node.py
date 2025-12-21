#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2, Image, CameraInfo, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge, CvBridgeError
import time

from sensor_msgs_py.point_cloud2 import read_points
# import ros_numpy
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup



class PID:
    def __init__(self, kp=0.8, ki=0.0, kd=0.05, windup_limit=1.0, output_limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.windup_limit = windup_limit
        self.min_output, self.max_output = output_limits
        self._last_error = 0.0
        self._int = 0.0
        self._last_time = 0.0
        self._now = 0.0


    def reset(self):
        self._last_error = 0.0
        self._int = 0.0
        self._last_time = 0.0

    def update_time(self, time_msg):
        now = time_msg.clock.sec + time_msg.clock.nanosec * 1e-9
        self._last_time = self._now
        self._now = now

    def compute(self, error):
        dt = self._now - self._last_time
        if dt <= 0.0:
            dt = 1e-3

        # Proportional
        p = self.kp * error

        # Integral with anti-windup
        self._int += error * dt

        # clamp integrator
        if self.windup_limit is not None:
            self._int = max(min(self._int, self.windup_limit), -self.windup_limit)
        i = self.ki * self._int

        # Derivative
        d_error = (error - self._last_error) / dt
        d = self.kd * d_error

        out = p + i + d

        # Save state
        self._last_error = error

        # Output limits
        out = max(self.min_output, min(out, self.max_output))

        return out


class BaseNode(Node):
    def __init__(self):
        super().__init__('base_node')

        # QoS for image topics (best-effort or reliable depending on setup)
        qos_img = QoSProfile(depth=2)
        qos_img.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_img.durability = QoSDurabilityPolicy.VOLATILE

        self.window_size_for_depth_camera = 10

        self.depth_executor = SingleThreadedExecutor()
        self.depth_callback_group = ReentrantCallbackGroup()
        
        self.depth_point = self.create_subscription(
            PointCloud2,
            '/depth/points',
            self.depth_point_callback,
            10,
            callback_group=self.depth_callback_group
        )
        
        # Запускаем executor в отдельном потоке
        import threading
        self.depth_executor = SingleThreadedExecutor()
        self.depth_executor.add_node(self)
        
        # Запускаем в отдельном потоке
        self.depth_thread = threading.Thread(
            target=self._run_depth_executor,
            daemon=True
        )
        self.depth_thread.start()
        
        # Флаг для контроля обработки
        self.min_distance = float('inf')
        self.obstacle_x_norm = float('inf')
        self.DEPTH_THRESHOLD = 1.5
        self.depth_processing_enabled = False
        self.last_depth_time = time.time()

        self.clock = self.create_subscription(
            Clock,
            '/clock',
            self.clock_callback,
            10
        )

        self.scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.depth_image = self.create_subscription(
            Image,
            '/depth/image',
            self.depth_image_callback,
            10
        )

        self.color_image = self.create_subscription(
            Image,
            '/color/image',
            self.color_image_callback,
            10
        )

        self.depth_info = self.create_subscription(
            CameraInfo,
            '/depth/camera_info',
            self.depth_info_callback,
            10
        )

        self.color_info = self.create_subscription(
            CameraInfo,
            '/color/camera_info',
            self.color_info_callback,
            10
        )

        # self.depth_point = self.create_subscription(
        #     PointCloud2,
        #     '/depth/points',
        #     self.depth_point_callback,
        #     10
        # )

        self.imu = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.aruco = self.create_publisher(
            Float32,
            '/mission_aruco',
            10
        )

        self.finish = self.create_publisher(
            String,
            '/robot_finish',
            10
        )

        self.yellow_lower = np.array([27, 120, 120])
        self.yellow_upper = np.array([35, 255, 255])
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 40, 255])

        # PID & speed params
        self.kp = 4.0
        self.ki = 0.0
        self.kd = 1.5
        self.max_angular = 1.2

        self.max_speed = 0.2
        self.min_speed = 0.1
        self.speed_reduction_factor = 0.8

        self.last_x_target = 0.05
        self.beta = 0.15

        self.pid = PID(kp=self.kp, ki=self.ki, kd=self.kd, windup_limit=1.0, output_limits=(-self.max_angular, self.max_angular))

        self.bridge = CvBridge()

        self.started = False
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        #! Путь для записи видео
        self.output = cv2.VideoWriter("/home/babrakadabra/ros-competition/output.mp4", fourcc, 30, (848, 480))
        self.depth_output = cv2.VideoWriter("/home/babrakadabra/ros-competition/depth_output.mp4", fourcc, 10, (848, 480))

        self.green_lower = np.array([40, 80, 80])
        self.green_upper = np.array([80, 255, 255])

        self.x_target = 0
        self.flag_sign = 0
        self.sign = 0

    def _run_depth_executor(self):
        try:
            self.depth_executor.spin()
        except Exception as e:
            self.get_logger().error(f"Depth executor error: {e}")
    
    def clock_callback(self, msg: Clock):
        self.pid.update_time(msg)


    def scan_callback(self, msg: LaserScan):
        pass


    def depth_image_callback(self, msg: Image):
        self._save_depth_output(msg)


    def depth_info_callback(self, msg: CameraInfo):
        pass


    def color_info_callback(self, msg: CameraInfo):
        pass


    def _min_distance_in_window(self, pc: PointCloud2, window_height_ratio=0.5, width_margin_ratio=0.1):
        points = read_points(pc, field_names=("x","y","z"), skip_nans=True)
    
        if len(points) == 0:
            return float('inf'), None
        
        width = pc.width
        height = pc.height
        
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        # 1. Создаем маску для окна
        total_points = len(points)
        indices = np.arange(total_points)
        
        # Вычисляем строки (v) и столбцы (u) для каждой точки
        rows = indices // width  # целочисленное деление = строка
        cols = indices % width   # остаток от деления = столбец
        
        # 2. Маска для высоты (верхние window_height_ratio%)
        height_mask = rows < (height * window_height_ratio)
        
        # 3. Маска для ширины (отступы width_margin_ratio с каждой стороны)
        width_mask = (cols >= (width * width_margin_ratio)) & (cols < (width * (1 - width_margin_ratio)))
        
        # 4. Общая маска окна
        window_mask = height_mask & width_mask
        
        # 5. Берем только точки в окне (и их координаты)
        window_points = points[window_mask]
        window_cols = cols[window_mask]  # сохраняем u-координаты
        
        if len(window_points) == 0:
            return float('inf'), None
        
        # 6. Берем x-координаты (глубину)
        x_coords = window_points[:, 0]
        
        # 7. Фильтруем по расстоянию
        valid_mask = (x_coords > 0) & (x_coords < self.DEPTH_THRESHOLD)
        valid_x = x_coords[valid_mask]
        valid_cols = window_cols[valid_mask]  # соответствующие u-координаты
        
        if len(valid_x) == 0:
            return float('inf'), None
        
        # 8. Находим индекс минимального расстояния
        min_idx = np.argmin(valid_x)
        min_dist = valid_x[min_idx]
        obstacle_u = valid_cols[min_idx]  # u-координата точки с минимальным расстоянием
        
        # 9. Нормализуем координату (-1..1) относительно центра кадра
        center_u = width // 2
        obstacle_u_norm = (obstacle_u - center_u) / (width // 2)
        
        self.get_logger().debug(
            f"Obstacle: dist={min_dist:.3f}m, "
            f"u={obstacle_u}, "
            f"norm={obstacle_u_norm:.3f}"
        )
        
        return min_dist, obstacle_u_norm

    def depth_point_callback(self, msg):
        """Callback выполняется в отдельном потоке"""
        if not self.depth_processing_enabled:
            return
        
        try:
            # self.min_distance = self._min_distance_in_window(msg)
            
            if time.time() - self.last_depth_time > 0.1:  # 10 Гц
                self.min_distance, self.obstacle_x_norm = self._min_distance_in_window(msg)
                self.last_depth_time = time.time()
            
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")

    
    def _get_lane_boundaries(self, yellow_mask, white_mask, width, height, bottom_ratio=0.3, margin_px=0):
        # Область анализа (нижние 30%)
        bottom_start = int(height * (1 - bottom_ratio))
        
        # 1. Желтая линия - левая граница
        yellow_bottom = yellow_mask[bottom_start:, :]
        yellow_pixels = np.column_stack(np.where(yellow_bottom > 0))
        
        if len(yellow_pixels) > 0:
            # Берем 90-й перцентиль правых желтых пикселей (устойчивее к выбросам)
            right_yellows = np.percentile(yellow_pixels[:, 1], 90)
            left_boundary = int(right_yellows) + margin_px  # отступ от желтой линии
        else:
            left_boundary = margin_px  # минимальный отступ от левого края
        
        # 2. Белая линия - правая граница
        white_bottom = white_mask[bottom_start:, :]
        white_pixels = np.column_stack(np.where(white_bottom > 0))
        
        if len(white_pixels) > 0:
            # Берем 10-й перцентиль левых белых пикселей
            left_whites = np.percentile(white_pixels[:, 1], 10)
            right_boundary = int(left_whites) - margin_px  # отступ от белой линии
        else:
            right_boundary = width - margin_px  # минимальный отступ от правого края
        
        # # 3. ЖЕСТКИЕ ОГРАНИЧЕНИЯ:
        
        # # а) Минимальная ширина полосы
        # MIN_LANE_WIDTH = width * 0.3  # не менее 30% ширины
        # if right_boundary - left_boundary < MIN_LANE_WIDTH:
        #     # Расширяем от центра
        #     center_x = (left_boundary + right_boundary) // 2
        #     left_boundary = max(margin_px, int(center_x - MIN_LANE_WIDTH / 2))
        #     right_boundary = min(width - margin_px, int(center_x + MIN_LANE_WIDTH / 2))
        
        # # б) Гарантированный отступ от краев кадра
        # left_boundary = max(margin_px, left_boundary)
        # right_boundary = min(width - margin_px, right_boundary)
        
        # # в) Белая должна быть справа от желтой с запасом
        # if right_boundary <= left_boundary + margin_px * 2:
        #     # Пересечение границ - используем безопасные значения
        #     left_boundary = margin_px
        #     right_boundary = width - margin_px
        
        # self.get_logger().debug(f"Strict boundaries: [{left_boundary}, {right_boundary}]")
        
        return left_boundary, right_boundary
    

    def compute_avoidance_x(self, lane_target_x, left_boundary, right_boundary):
        """
        Формула отталкивания: res = x - alpha * (y - x) / (abs(y - x) + eps) * x * (1 - x)
        где x - нормализованная цель (0..1 в пределах полосы)
        y - нормализованная координата препятствия (0..1 в пределах полосы)
        """
        if self.min_distance > self.DEPTH_THRESHOLD or self.obstacle_x_norm is None:
            return lane_target_x
        
        # 1. Нормализуем координату цели (x) в диапазон 0..1 относительно полосы
        x_norm = (lane_target_x - left_boundary) / (right_boundary - left_boundary)
        
        # 2. Преобразуем obstacle_x_norm (-1..1 от центра кадра) в 0..1 относительно полосы
        # Предполагаем, что полоса в центре кадра
        image_center = (right_boundary + left_boundary) / 2
        lane_half_width = (right_boundary - left_boundary) / 2
        
        # obstacle_x_norm = -1..1 относительно центра кадра -> -1..1 относительно центра полосы
        # Просто считаем, что полоса и кадр совпадают по центру
        y_norm_relative = (self.obstacle_x_norm + 1) / 2  # -1..1 -> 0..1
        
        # Ограничиваем в пределах полосы (0..1)
        y_norm = np.clip(y_norm_relative, 0.0, 1.0)
        
        # 3. Параметры формулы
        alpha = 1.5  # степень отталкивания (0.5-2.0)
        eps = 1e-6   # для избежания деления на 0
        
        # 4. Масштабируем alpha в зависимости от расстояния
        # Чем ближе препятствие - тем сильнее отталкивание
        distance_factor = 1.0 - min(self.min_distance / 0.5, 1.0)  # 0.5м - максимальная сила
        alpha_scaled = alpha * distance_factor
        
        # 5. Применяем вашу формулу
        diff = y_norm - x_norm
        direction = diff / (abs(diff) + eps)  # -1 или +1
        
        # Основная формула
        res_norm = x_norm - alpha_scaled * direction * x_norm * (1 - x_norm)
        
        # 6. Ограничиваем результат 0..1
        res_norm = np.clip(res_norm, 0.0, 1.0)
        
        # 7. Преобразуем обратно в пиксели
        avoidance_x = left_boundary + res_norm * (right_boundary - left_boundary)
        
        # 8. Дополнительная логика для экстремальных случаев
        if self.min_distance < 0.3:  # Очень близко (30см)
            # Усиливаем отталкивание
            if y_norm < 0.5:  # Препятствие в левой половине
                # Сильно смещаемся вправо
                avoidance_x = right_boundary - (right_boundary - left_boundary) * 0.2
            else:  # Препятствие в правой половине
                # Сильно смещаемся влево
                avoidance_x = left_boundary + (right_boundary - left_boundary) * 0.2
        
        # 9. Логирование
        self.get_logger().info(
            f"Avoidance formula: "
            f"x_norm={x_norm:.2f}, "
            f"y_norm={y_norm:.2f} (raw={self.obstacle_x_norm:.2f}), "
            f"dist={self.min_distance:.2f}m, "
            f"alpha={alpha_scaled:.2f}, "
            f"res_norm={res_norm:.2f}, "
            f"avoidance={avoidance_x:.0f}px"
        )
        
        return avoidance_x


    def imu_callback(self, msg: Imu):
        pass


    def color_image_callback(self, msg: Image):
        self._cv_steer(msg)


    def detect_turn_sign(self, image, min_blue_area=500):
        """
        Returns:
            0  - no sign
            1  - turn right
            -1  - turn left
        """

        h, w = image.shape[:2]
        mid_image = image #[h//4 : 3*h//4, :]

        hsv = cv2.cvtColor(mid_image, cv2.COLOR_BGR2HSV)
        blue_lower = np.array([95, 120, 70])
        blue_upper = np.array([130, 255, 255])

        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_blue_area:
                continue

            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            cx, cy, r = int(cx), int(cy), int(r)

            if r < 15:
                continue

            roi = blue_mask[cy - r:cy + r, cx - r:cx + r]
            if roi.size == 0:
                continue

            h, w = roi.shape
            left_pixels = np.count_nonzero(roi[h // 2:, :w // 2])
            right_pixels = np.count_nonzero(roi[h // 2:, w // 2:])

            flag = 1 if right_pixels > left_pixels else -1

            return flag, right_pixels, left_pixels

        return 0, 0, 0


    def _detect_green_light(self, image, min_area=1500):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        green_area = np.count_nonzero(green_mask)

        return green_area > min_area


    def _cv_steer(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return

        h, w = cv_image.shape[:2]

        # Crop bottom portion where lanes are nearer
        crop_h = h // 2
        roi = cv_image[h - crop_h:h, 0:w]
        roi_h, roi_w = roi.shape[:2]

        # Preprocess
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Masks
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        # Morphology to clean masks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        self.get_logger().info(f"❕ self.started {self.started}")

        if self._detect_green_light(cv_image):
            self.started = True

        self.started = True
        if self.started:
            left_boundary, right_boundary = self._get_lane_boundaries(
                yellow_mask, white_mask, roi_w, roi_h
            )

            tmp_sign, right_pixels, left_pixels = self.detect_turn_sign(cv_image, 10000)
                
            if tmp_sign != 0 and self.flag_sign == 0:
                self.sign = tmp_sign
                self.flag_sign = 1
            elif tmp_sign == 0 and self.flag_sign == 1:
                self.sign = tmp_sign
                self.flag_sign = 0
            
            # left_boundary, right_boundary = self._get_lane_boundaries(
            #     yellow_mask, white_mask, roi_h, roi_w
            # )

            # Compute target and error
            self.x_target = self._compute_lane_target(roi, min_area=2000)

            # объезд препятствий
            if self.min_distance is not None and self.min_distance < float('inf'):
                self.x_target = self.compute_avoidance_x(
                    self.x_target, left_boundary, right_boundary
                )
            image_center_x = w / 2.0 

            # self.x_target = (self.x_target + (image_center_x + self.sign * image_center_x)) / 2
            self.x_target = self.x_target if self.sign == 0 else (image_center_x + self.sign * image_center_x)

            self.get_logger().info(f"{"❕" if self.sign == 0 else "❗"} sign {self.sign}| target {self.x_target}")

            self.x_target = self.x_target - self.beta * (self.x_target - self.last_x_target)
            self.last_x_target = self.x_target

            error_px = self.x_target - image_center_x
            norm_error = error_px / (w / 2.0)

            # square_error = error_px * abs(error_px) / (w / 2.0)

            ang = self.pid.compute(norm_error)

            # Compute forward speed with penalties
            steering_penalty = min(abs(ang) * self.speed_reduction_factor, 1.0)
            edge_penalty = max(0.0, abs(norm_error) - 0.7)
            edge_penalty = min(edge_penalty, 1.0)

            speed = self.max_speed * (1.0 - steering_penalty - edge_penalty)
            speed = max(min(speed, self.max_speed), self.min_speed)
        
            self.get_logger().info("🟢 START")
            twist = Twist()
            twist.linear.x = float(speed)
            twist.angular.z = float(-ang)
            self.cmd_pub.publish(twist)
        else:
            self.get_logger().info("❌ STOP")


        # Debug image overlay
        debug = cv_image.copy()

        mask_vis = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
        mask_vis2 = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        combined_mask = cv2.bitwise_or(mask_vis, mask_vis2)
        h_small = h // 3
        debug[0:h_small, 0:(w // 3)] = cv2.resize(combined_mask, (w // 3, h_small))

        if self.started:
            left_int = int(left_boundary)
            right_int = int(right_boundary)
            target_int = int(self.x_target)
            
            cv2.line(debug, (left_int, h), (left_int, h - crop_h), 
                    (0, 255, 255), 2)  # желтая
            cv2.line(debug, (right_int, h), (right_int, h - crop_h), 
                    (255, 255, 255), 2)  # белая
            cv2.line(debug, (target_int, h), (target_int, h - crop_h), 
                    (0, 255, 0), 3)  # зеленая

        self.output.write(debug)
    

    def _compute_lane_target(self, roi, min_area=100):
        h, w = roi.shape[:2]

        # Preprocess
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Masks
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        # Morphology to clean masks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        # Find center of bright region(s)

        cx = int(w * 0.3)

        yellow_M = cv2.moments(yellow_mask)
        white_M = cv2.moments(white_mask)
        yellow_area = np.count_nonzero(yellow_mask)
        white_area = np.count_nonzero(white_mask)
        
        if yellow_area > min_area:
            if white_area > min_area:
                yellow_x = int(yellow_M['m10']/yellow_M['m00'])
                white_x = int(white_M['m10']/white_M['m00'])

                if white_x - yellow_x > 0:                   
                    cx = yellow_x + (white_x - yellow_x) // 2
            else:
                cx = int(w * 0.7)
        else:
            if white_area <= min_area:
                cx = w // 2

        return cx
    

    def _save_depth_output(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return
        
        # Depth image processing
        img_clean = np.nan_to_num(cv_image, nan=0.0, posinf=0.0, neginf=0.0)
        vis = cv2.normalize(img_clean, None, 0, 255, cv2.NORM_MINMAX)
        vis = vis.astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        self.depth_output.write(vis)


def main(args=None):
    rclpy.init(args=args)
    node = BaseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
