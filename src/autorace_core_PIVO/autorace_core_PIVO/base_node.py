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
from nav_msgs.msg import Odometry


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

        # Флаг для контроля обработки
        self.min_distance = float('inf')
        self.obstacle_x_norm = float('inf')
        self.DEPTH_THRESHOLD = 0.5
        # self.depth_processing_enabled = False
        self.last_depth_time = 0
        self.current_ros_time = None
        self.first_time = None

        # self.depth_callback_group = ReentrantCallbackGroup()
        
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
            10,
        )

        self.depth_image = self.create_subscription(
            Image,
            '/depth/image',
            self.depth_image_callback,
            10,
        )

        self.color_image = self.create_subscription(
            Image,
            '/color/image',
            self.color_image_callback,
            10,
        )

        self.depth_info = self.create_subscription(
            CameraInfo,
            '/depth/camera_info',
            self.depth_info_callback,
            10,
        )

        self.color_info = self.create_subscription(
            CameraInfo,
            '/color/camera_info',
            self.color_info_callback,
            10,
        )

        self.imu = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10,
        )

        self.depth_point = self.create_subscription(
            PointCloud2,
            '/depth/points',
            self.depth_point_callback,
            10,
            # callback_group=self.depth_callback_group
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

         # Подписка на одометрию
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # Позиция робота
        self.robot_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        # self.robot_orientation = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.robot_pose_initialized = False
        self.current_checkpoint_index = 0
        self.special_avoidance_mode = False
        self.checkpoints = [
            {'x': 0.9540668182366147, 'y': 3.4298825170464355, 'z': 0.0, 'reached': False, 'distance_threshold': 0.3},
            {'x': 0.6, 'y': 3.962901778906285, 'z': 0.0, 'reached': False, 'distance_threshold': 0.4}
        ]



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
    
    def odom_callback(self, msg: Odometry):
        if self.current_checkpoint_index != -1:
            self.robot_position['x'] = msg.pose.pose.position.x
            self.robot_position['y'] = msg.pose.pose.position.y
            self.robot_position['z'] = msg.pose.pose.position.z
            self.get_logger().info(
                f"robot_pose: {self.robot_position}"
            )
            self.robot_pose_initialized = True
        
            self._check_checkpoints_reached()
        else:
            self.get_logger().info(
                f"низя"
            )
    
    def _check_checkpoints_reached(self):
        if self.robot_position is None:
            return
        
        if self.current_checkpoint_index >= len(self.checkpoints):
            self.special_avoidance_mode = False
            self.current_checkpoint_index = -1
            return
        
        current_checkpoint = self.checkpoints[self.current_checkpoint_index]
        
        dx = self.robot_position['x'] - current_checkpoint['x']
        dy = self.robot_position['y'] - current_checkpoint['y']
        dz = self.robot_position['z'] - current_checkpoint['z']
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if distance < current_checkpoint['distance_threshold'] and not current_checkpoint['reached']:
            current_checkpoint['reached'] = True
            self.current_checkpoint_index += 1
            
            self.get_logger().info(
                f"✅ Checkpoint {self.current_checkpoint_index} reached! "
                f"Distance: {distance:.3f}m"
            )
            
            if self.current_checkpoint_index == 1:
                self.special_avoidance_mode = True
                self.get_logger().info("🔶 Special avoidance mode ACTIVATED")
            elif self.current_checkpoint_index == 2:
                self.special_avoidance_mode = False
                self.get_logger().info("✅ Special avoidance mode DEACTIVATED")

    def clock_callback(self, msg: Clock):
        self.pid.update_time(msg)
        self.current_ros_time = msg.clock.sec + msg.clock.nanosec * 1e-9
        if self.first_time == None:
            self.first_time = self.current_ros_time


    def scan_callback(self, msg: LaserScan):
        pass


    def depth_image_callback(self, msg: Image):
        self._save_depth_output(msg)


    def depth_info_callback(self, msg: CameraInfo):
        pass


    def color_info_callback(self, msg: CameraInfo):
        pass


    def _min_distance_in_window(self, pc: PointCloud2, window_height_ratio=0.5, width_margin_ratio=0.0):
        points = read_points(pc, field_names=("x", "y", "z"), skip_nans=True)
        
        if len(points) == 0:
            return float('inf'), None
        
        width = pc.width
        height = pc.height
        
        if points.dtype.names:  # Проверяем, структурированный ли это массив
            x_coords_full = points['x']
            y_coords_full = points['y']
            z_coords_full = points['z']
            
            # Создаем обычный массив [x, y, z]
            points_array = np.column_stack([x_coords_full, y_coords_full, z_coords_full])
        else:
            points_array = np.array(points)
        
        if points_array.ndim == 1:
            points_array = points_array.reshape(1, -1)
        
        total_points = len(points_array)
        indices = np.arange(total_points)
        
        # Вычисляем строки (v) и столбцы (u) для каждой точки
        rows = indices // width
        cols = indices % width
        
        # Маска для высоты (верхние window_height_ratio%)
        height_mask1 = rows < (height * window_height_ratio)
        height_mask2 = rows > (height * 0.15)
        
        # Маска для ширины (отступы width_margin_ratio с каждой стороны)
        width_mask = (cols >= (width * width_margin_ratio)) & (cols < (width * (1 - width_margin_ratio)))
        
        # Общая маска окна
        window_mask = height_mask1 & width_mask & height_mask2
        
        # Берем только точки в окне (и их координаты)
        window_points = points_array[window_mask]
        window_cols = cols[window_mask]
        
        if len(window_points) == 0:
            return float('inf'), None
        
        # Берем x-координаты (глубину)
        x_coords = window_points[:, 0]
        
        # Фильтруем по расстоянию - теперь x_coords это обычный числовой массив
        valid_mask = (x_coords > 0) & (x_coords < self.DEPTH_THRESHOLD)
        valid_x = x_coords[valid_mask]
        valid_cols = window_cols[valid_mask]  # соответствующие u-координаты
        
        if len(valid_x) == 0:
            return float('inf'), None
        
        # Находим индекс минимального расстояния
        min_idx = np.argmin(valid_x)
        min_dist = valid_x[min_idx]
        obstacle_u = valid_cols[min_idx]  # u-координата точки с минимальным расстоянием
        
        # Нормализуем координату (-1..1) относительно центра кадра
        center_u = width // 2
        obstacle_u_norm = (obstacle_u - center_u) / (width // 2)
        
        self.get_logger().debug(
            f"Obstacle: dist={min_dist:.3f}m, "
            f"u={obstacle_u}, "
            f"norm={obstacle_u_norm:.3f}"
        )
        
        return min_dist, obstacle_u_norm

    def depth_point_callback(self, msg):
        
        try:
            if (self.current_ros_time is not None and 
                self.last_depth_time is not None and
                self.current_ros_time - self.last_depth_time > 0.1):

                self.min_distance, self.obstacle_x_norm = self._min_distance_in_window(msg)
                self.last_depth_time = self.current_ros_time
            
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")

    
    def _get_lane_boundaries(self, yellow_mask, white_mask, width, height, bottom_ratio=0.3):
        bottom_start = int(height * (1 - bottom_ratio))
        margin_px = width*0.1
        
        # Желтая линия - левая граница
        yellow_bottom = yellow_mask[bottom_start:, :]
        yellow_pixels = np.column_stack(np.where(yellow_bottom > 0))
        
        if len(yellow_pixels) > 0:
            # Берем 90-й перцентиль правых желтых пикселей 
            right_yellows = np.percentile(yellow_pixels[:, 1], 90)
            left_boundary = int(right_yellows)
        else:
            left_boundary = margin_px
        
        # Белая линия - правая граница
        white_bottom = white_mask[bottom_start:, :]
        white_pixels = np.column_stack(np.where(white_bottom > 0))
        
        if len(white_pixels) > 0:
            # Берем 10-й перцентиль левых белых пикселей
            left_whites = np.percentile(white_pixels[:, 1], 10)
            right_boundary = int(left_whites)
        else:
            right_boundary = width - margin_px
        
        if left_boundary >= 0.6 * width:
            left_boundary = margin_px
        if right_boundary <= 0.4 * width:
            right_boundary = width - margin_px
        
        return left_boundary, right_boundary
    

    def compute_avoidance_x(self, lane_target_x, left_boundary, right_boundary):
        """
        Формула отталкивания: res = x - alpha * (y - x) / (abs(y - x) + eps) * x * (1 - x)
        где x - нормализованная цель (0..1 в пределах полосы)
        y - нормализованная координата препятствия (0..1 в пределах полосы)
        """
        if self.min_distance > self.DEPTH_THRESHOLD or self.obstacle_x_norm is None:
            return lane_target_x
        
        # Нормализуем координату цели (x) в диапазон 0..1 относительно полосы
        x_norm = (lane_target_x - left_boundary) / (right_boundary - left_boundary)
        
        # obstacle_x_norm = -1..1 относительно центра кадра -> -1..1 относительно центра полосы
        # Просто считаем, что полоса и кадр совпадают по центру
        y_norm_relative = (self.obstacle_x_norm + 1) / 2  # -1..1 -> 0..1
        
        # Ограничиваем в пределах полосы (0..1)
        y_norm = np.clip(y_norm_relative, 0.0, 1.0)
        
        # Параметры формулы
        alpha = 1.5  # степень отталкивания (0.5-2.0)
        eps = 1e-6   # для избежания деления на 0
        
        # Масштабируем alpha в зависимости от расстояния
        # Чем ближе препятствие - тем сильнее отталкивание
        distance_factor = 1.0 - min(self.min_distance / 0.5, 1.0)
        alpha_scaled = alpha * distance_factor
        
        diff = y_norm - x_norm
        direction = diff / (abs(diff) + eps)  # -1 или +1
        
        if self.special_avoidance_mode and self.current_checkpoint_index == 1:

            target_point = self.checkpoints[1]  # Вторая точка
            # Вычисляем направление от текущей позиции к целевой точке
            # В мировых координатах
            dx = target_point['x'] - self.robot_position['x']
            dy = target_point['y'] - self.robot_position['y']
            
            # Вычисляем угол к цели
            target_angle = np.arctan2(dy, dx)  # угол в радианах
            
            fov_rad = 1.047  # 60 градусов
            
            # Ограничиваем угол в пределах FOV камеры
            target_angle = np.clip(target_angle, -fov_rad/2, fov_rad/2)
            
            # Преобразуем угол в нормализованную координату (-1..1)
            target_norm = target_angle / (fov_rad / 2)
            
            # Преобразуем -1..1 в 0..1 относительно полосы
            target_norm_lane = (target_norm + 1) / 2
            
            # Теперь используем target_norm_lane как целевую точку
            # вместо вычисления через границы
            
            # Вычисляем разницу между текущей целью и целевой точкой
            diff_to_target = target_norm_lane - x_norm
            
            # Направление к целевой точке
            direction = diff_to_target / (abs(diff_to_target) + eps)
        
        # Основная формула
        res_norm = x_norm - alpha_scaled * direction * x_norm * (1 - x_norm)
        
        res_norm = np.clip(res_norm, 0.0, 1.0)
        
        # Преобразуем обратно в пиксели
        avoidance_x = left_boundary + res_norm * (right_boundary - left_boundary)
        
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
            # left_boundary, right_boundary = self._get_lane_boundaries(
            #     yellow_mask, white_mask, roi_w, roi_h
            # )

            tmp_sign, right_pixels, left_pixels = self.detect_turn_sign(cv_image, 10000)
                
            if tmp_sign != 0 and self.flag_sign == 0:
                self.sign = tmp_sign
                self.flag_sign = 1
            elif tmp_sign == 0 and self.flag_sign == 1:
                self.sign = tmp_sign
                self.flag_sign = 0
            
            # Compute target and error
            self.x_target, left_boundary, right_boundary = self._compute_lane_target(roi, min_area=2000)

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

            obstacle_penalty = 0.0
    
            if self.min_distance is not None and self.min_distance < float('inf'):
                obstacle_penalty = 1.0 - (self.min_distance / 0.7)
                
                # Дополнительный штраф, если препятствие прямо по курсу
                if abs(self.obstacle_x_norm) < 0.2:
                    obstacle_penalty = 1
            
            total_penalty = steering_penalty + edge_penalty + obstacle_penalty
            total_penalty = min(total_penalty, 1.0)
            
            speed = self.max_speed * (1.0 - total_penalty)
            speed = max(min(speed, self.max_speed), self.min_speed)
            
            if self.min_distance is not None and self.min_distance < 0.05:
                speed = 0.0
                self.get_logger().warn("⚠️ VERY CLOSE OBSTACLE - STOPPING!")
                # можно добавить логику движения назад
        
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
    

    def _compute_lane_target(self, roi, min_area=100, bottom_ratio=0.3):
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

        bottom_start = int(h * (1 - bottom_ratio))
        margin_px = int(w * 0.1)

        # желтая линия
        yellow_bottom = yellow_mask[bottom_start:, :]
        yellow_pixels = np.column_stack(np.where(yellow_bottom > 0))
        
        if len(yellow_pixels) > min_area:
            # 90-й перцентиль правых желтых пикселей
            yellow_x = int(np.percentile(yellow_pixels[:, 1], 90))
            if yellow_x >= 0.6 * w:
                yellow_x = int(margin_px)
        else:
            yellow_x = int(margin_px)

        # белая линия
        white_bottom = white_mask[bottom_start:, :]
        white_pixels = np.column_stack(np.where(white_bottom > 0))
        
        if len(white_pixels) > min_area:
            # 10-й перцентиль левых белых пикселей
            white_x = int(np.percentile(white_pixels[:, 1], 10))
            if white_x <= 0.4 * w:
                white_x = int(w - margin_px)
        else:
            white_x = int(w - margin_px)

        # Вычисляем центр между границами
        if white_x > yellow_x and (white_x - yellow_x) > 10:
            cx = yellow_x + (white_x - yellow_x) // 2
        else:
            cx = w // 2
            # Если есть только желтая линия - держимся от нее справа
            if yellow_x > margin_px and white_x == w - margin_px:
                cx = w * 0.7
            # Если есть только белая линия - держимся от нее слева
            elif white_x < w - margin_px and yellow_x == margin_px:
                cx = w * 0.3

        lane_width = white_x - yellow_x

        self.get_logger().info(
            f"Lane detection: yellow_x={yellow_x}, white_x={white_x}, "
            f"center={cx}, lane_width={lane_width}"
        )

        return cx, yellow_x, white_x
    

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
