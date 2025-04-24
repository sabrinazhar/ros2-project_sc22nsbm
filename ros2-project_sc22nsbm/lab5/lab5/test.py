import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
import signal

class ColourIdentifier(Node):
    def __init__(self):
        super().__init__('colour_identifier')

        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()

        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # Camera topic
            self.camera_callback,
            10
        )

        # Subscribe to LIDAR scan topic for stopping at 1m distance
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',  # LIDAR sensor topic
            self.lidar_callback,
            10
        )

        # Initialize publisher for movement
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Define movement parameters
        self.move_cmd = Twist()
        self.moving_forward = True  
        self.obstacle_detected = False  
        self.blue_detected = False  

        # Define distance to stop in front of the blue box
        self.target_distance = 1.0  # Stop at 1 meter

        # Movement speed settings
        self.forward_speed = 1.0  # Slightly reduced for better control
        self.turn_speed = 0.4  # Moderate turning speed

        # Create a timer for motion planning
        self.timer = self.create_timer(0.3, self.motion_callback)

    def lidar_callback(self, msg):
        """
        Processes LIDAR data to stop at 1 meter from a detected blue box.
        """
        front_distance = msg.ranges[0]  

        if self.blue_detected and front_distance < self.target_distance and front_distance > 0.1:
            self.get_logger().info(f"Near blue box!")
            self.moving_forward = False  # Stop moving when close enough
        elif front_distance < 2.0 and front_distance > 0.1:
            self.obstacle_detected = True  # Standard obstacle avoidance
        else:
            self.obstacle_detected = False

    def motion_callback(self):
        """
        Controls the robot's movement based on obstacle detection and blue box tracking.
        """
        if self.obstacle_detected:
            self.move_cmd.linear.x = 0.0
            self.move_cmd.angular.z = self.turn_speed  
        elif self.blue_detected:
            self.publisher.publish(self.move_cmd)  # Move based on blue box tracking
        else:
            self.move_cmd.linear.x = self.forward_speed  
            self.move_cmd.angular.z = 0.0  

        self.publisher.publish(self.move_cmd)

    def camera_callback(self, data):
        """
        Detects and tracks the blue box.
        Adjusts movement based on its position in the frame.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Define blue color range
            blue_lower = np.array([110 - 10, 100, 100])
            blue_upper = np.array([110 + 10, 255, 255])

            # Create mask for blue
            blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

            # Find contours
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                self.moving_forward = False  # Stop moving forward and prepare to navigate towards the box
                self.blue_detected = True  
                self.move_cmd.linear.x = 0.0
                self.move_cmd.angular.z = 0.0  

                # Get the largest detected contour (assumes it's the blue box)
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)

                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])  # X position of the blue box
                    frame_center = cv_image.shape[1] // 2  

                    if abs(cx - frame_center) < 30:  # Blue box is centered
                        self.move_cmd.linear.x = self.forward_speed  
                        self.move_cmd.angular.z = 0.0  
                        self.get_logger().info("Moving forward to the blue box!")
                    elif cx < frame_center:  # Blue box is on the left
                        self.move_cmd.linear.x = 0.0  
                        self.move_cmd.angular.z = 0.5  
                        self.get_logger().info("Turning left to align with blue box!")
                    else:  # Blue box is on the right
                        self.move_cmd.linear.x = 0.0  
                        self.move_cmd.angular.z = -0.5  
                        self.get_logger().info("Turning right to align with blue box!")

            else:
                self.blue_detected = False  

            cv2.imshow("Camera Feed", cv_image)
            cv2.waitKey(3)

        except CvBridgeError as e:
            self.get_logger().info(f"CvBridge Error: {e}")

def main():
    def signal_handler(sig, frame):
        rclpy.shutdown()
        cv2.destroyAllWindows()

    rclpy.init()
    node = ColourIdentifier()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            continue
    except rclpy.exceptions.ROSInterruptException:
        pass

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
