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
        self.started = False


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

        self.depth_point = self.create_subscription(
            PointCloud2,
            '/depth/points',
            self.depth_point_callback,
            10
        )

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

        self.max_speed = 1.0
        self.min_speed = 0.2
        self.speed_reduction_factor = 0.8

        self.last_x_target = 0.0
        self.beta = 0.15

        self.pid = PID(kp=self.kp, ki=self.ki, kd=self.kd, windup_limit=1.0, output_limits=(-self.max_angular, self.max_angular))

        self.bridge = CvBridge()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        #! Путь для записи видео
        self.out = cv2.VideoWriter("/home/ilya/Documents/ros-competition/video/output.mp4", fourcc, 30, (848, 480))

        self.green_lower = np.array([40, 80, 80])
        self.green_upper = np.array([80, 255, 255])


    
    def clock_callback(self, msg: Clock):
        self.pid.update_time(msg)


    def scan_callback(self, msg: LaserScan):
        pass


    def depth_image_callback(self, msg: Image):
        pass


    def depth_info_callback(self, msg: CameraInfo):
        pass


    def color_info_callback(self, msg: CameraInfo):
        pass


    def depth_point_callback(self, msg: PointCloud2):
        pass


    def imu_callback(self, msg: Imu):
        pass


    def color_image_callback(self, msg: Image):
        self._cv_steer(msg)


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

        if self._detect_green_light(cv_image):

            # Compute target and error
            x_target = self._compute_lane_target(roi, min_area=2000)

            x_target = x_target - self.beta * (x_target - self.last_x_target)
            self.last_x_target = x_target

            image_center_x = w / 2.0 
            error_px = x_target - image_center_x
            norm_error = error_px / (w / 2.0)

            # square_error = error_px * abs(error_px) / (w / 2.0)

            ang = self.pid.compute(norm_error)

            # Compute forward speed with penalties
            steering_penalty = min(abs(ang) * self.speed_reduction_factor, 1.0)
            edge_penalty = max(0.0, abs(norm_error) - 0.7)
            edge_penalty = min(edge_penalty, 1.0)

            speed = self.max_speed * (1.0 - steering_penalty - edge_penalty)
            speed = max(min(speed, self.max_speed), self.min_speed)

        #! Publish Twist

        # self.get_logger().info("🟢 START")
        # twist = Twist()
        # twist.linear.x = float(speed)
        # twist.angular.z = float(-ang)
        # self.cmd_pub.publish(twist)


        
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

        self.out.write(debug)
    

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

        cx = int(w * 0.2)

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
                cx = int(w * 0.8)
        else:
            if white_area <= min_area:
                cx = w // 2

        return cx


def main(args=None):
    rclpy.init(args=args)
    node = BaseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
