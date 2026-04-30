import numpy as np
import math
from typing import Tuple
from threading import Lock

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud, LaserScan, Image
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import UInt8MultiArray

from turtlebot_landmark_slam.ekf_orchestrator import EkfOrchestrator
from turtlebot_landmark_slam.types import ControlMeasurement
from turtlebot_landmark_slam.utils import yaw_from_quaternion

class EkfInterface(object):
    def __init__(self, node: Node, orchestrator: EkfOrchestrator) -> None:
        self._node = node
        self._orchestrator = orchestrator
        self._lock = Lock()

        self._last_motion_msg_time = None
        self._last_small_motion_log_time = None

        """
        tue_test_0 -> lin : 0.2, ang : 60
        """
        self.std_dev_linear_vel = float(self._node.declare_parameter("std_dev_linear_vel", 0.2).value) 
        self.std_dev_angular_vel = float(self._node.declare_parameter("std_dev_angular_vel", (60 * np.pi) / 180).value)

        self._node.get_logger().info(
            f"[DataProvider] std_dev_linear_vel: {self.std_dev_linear_vel}"
        )
        self._node.get_logger().info(
            f"[DataProvider] std_dev_angular_vel: {self.std_dev_angular_vel}"
        )

        
        self.is_real = bool(self._node.get_parameter("is_real").value)

        if self.is_real:
            self._node.get_logger().info(
                "is_real=true: Taking Lidar data from /pointcloud2d."
                " No noise will be added to motion inputs."
                " Control noise std deviation can be set in ekf_pipeline.launch."
            )

            self._orchestrator.real_env = True
            self._lidar_message_type = PointCloud
            self._lidar_topic = 'pointcloud2d'
        else:
            self._node.get_logger().info(
                "is_real=false: Taking Lidar data from /scan."
                " No noise will be added to motion inputs."
                " Control noise std deviation can be set in ekf_pipeline.launch."
            )
            
            self._orchestrator.real_env = False
            self._lidar_message_type = LaserScan
            self._lidar_topic = 'scan'

        ## Subscribers ## 
        self._lidar_subscription = self._node.create_subscription(
            self._lidar_message_type, 
            self._lidar_topic, 
            self._lidar_callback, 
            qos_profile=10
        )

        self._motion_subscription = self._node.create_subscription(
            Odometry,
            'odom',
            self._motion_callback,
            qos_profile=10
        )

        self._image_subscription = self._node.create_subscription(
            Image,
            '/camera/image_raw',
            self._image_callback,
            qos_profile=10
        )

        ## Publishers ## 
        self.odom_publisher = self._node.create_publisher(Odometry, "~/odom", 1)
        self.map_publisher = self._node.create_publisher(MarkerArray, "~/map", 5)
        self.telemetry_publisher = self._node.create_publisher(UInt8MultiArray, "~/telemetry", 1)
        self.visible_landmark_publisher = self._node.create_publisher(Image, '~/visible_landmarks', 1)

        self._orchestrator.handover_telem_publisher(self.telemetry_publisher)
        self._orchestrator.handover_visible_landmark_publisher(self.visible_landmark_publisher)

    def _motion_callback(self, msg):
        now = self._node.get_clock().now()

        if self._last_motion_msg_time is None:
            self._last_motion_msg_time = now
            return
        
        # ### FOR SIMULATED LANDMARKS ONLY ###
        # x = msg.pose.pose.position.x
        # y = msg.pose.pose.position.y
        # theta = yaw_from_quaternion(msg.pose.pose.orientation)
        # self._orchestrator.lidar_observer.update_pose([x,y,theta])

        dt = (now - self._last_motion_msg_time).nanoseconds / 1e9
        self._last_motion_msg_time = now

        linear_vel = msg.twist.twist.linear.x
        angular_vel = msg.twist.twist.angular.z
        
        # ignore small motions and don't spam log with status messages
        if abs(linear_vel) < 0.009 and abs(angular_vel) < 0.09:
            should_log = (
                self._last_small_motion_log_time is None 
                or (now - self._last_small_motion_log_time).nanoseconds / 1e9 >= 10.0
            )
        
            if should_log:
                self._node.get_logger().info(
                    "Small linear or angular motion. Skipping predict step"
                )
                self._last_small_motion_log_time = now
            return
        

        motion_command, motion_covariance = self._constructMotionWithCovariance(
            linear_vel, angular_vel, self.std_dev_linear_vel, self.std_dev_angular_vel, dt
        )

        assert(motion_command.shape == (3,1))
        assert(motion_covariance.shape == (3,3))

        dx = motion_command[0][0]
        dy = motion_command[1][0]
        dtheta = motion_command[2][0]

        motion_measurement = ControlMeasurement(dx, dy, dtheta, motion_covariance)
        self._orchestrator.motion_handler(motion_measurement)

        self.publishOdometry(
            self._orchestrator._ekf.pose,
            self._orchestrator._ekf.pose_covariance)

        
    def _lidar_callback(self, msg):
        print("lidar callback fired")
        
        if self._last_motion_msg_time is None:
            print("No previous motion, skipping...")
            return
    
        if self.is_real:
            points = np.array([[p.x, p.y] for p in msg.points], dtype=float)
        else:
            points = self._laserscan_to_rel_point(msg)

        self._orchestrator.lidar_handler(points)

        self.publishOdometry(
            self._orchestrator._ekf.pose,
            self._orchestrator._ekf.pose_covariance)

        self._publishLandmarkMap()

    def _image_callback(self, msg):
        self._orchestrator.image_handler(msg)


    def publishOdometry(self, pose, pose_covariance):
        msg = Odometry()
        msg.header.stamp = self._last_motion_msg_time.to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"

        with self._lock:
            msg.pose.pose.position.x = float(pose[0])
            msg.pose.pose.position.y = float(pose[1])
            msg.pose.pose.position.z = 0.0

            quat = self._quaternion_from_yaw(pose[2])
            msg.pose.pose.orientation.x = quat[0]
            msg.pose.pose.orientation.y = quat[1]
            msg.pose.pose.orientation.z = quat[2]
            msg.pose.pose.orientation.w = quat[3]

            # The ROS2 Odometry covariance is a 6x6 matrix (row-major, 36 elements)
            # for [x, y, z, roll, pitch, yaw]. Populate the [x, y, yaw] sub-block.
            cov = np.zeros(36, dtype=np.float64)
            cov[0] = pose_covariance[0, 0]  # x-x
            cov[1] = pose_covariance[0, 1]  # x-y
            cov[5] = pose_covariance[0, 2]  # x-yaw
            cov[6] = pose_covariance[1, 0]  # y-x
            cov[7] = pose_covariance[1, 1]  # y-y
            cov[11] = pose_covariance[1, 2]  # y-yaw
            cov[30] = pose_covariance[2, 0]  # yaw-x
            cov[31] = pose_covariance[2, 1]  # yaw-y
            cov[35] = pose_covariance[2, 2]  # yaw-yaw
            msg.pose.covariance = cov

        self.odom_publisher.publish(msg)


    def _publishLandmarkMap(self):
        landmark_poses = self._orchestrator._ekf.state_mean[3:].flatten()

        marker_array_msg = MarkerArray()

        for i in range(len(landmark_poses) // 2):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(landmark_poses[2 * i])
            marker.pose.position.y = float(landmark_poses[2 * i + 1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.color.r = 1.0
            marker.color.a = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.frame_locked = False
            marker_array_msg.markers.append(marker)

        self.map_publisher.publish(marker_array_msg)      

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _laserscan_to_rel_point(self, msg):
        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        ranges = np.array(msg.ranges, dtype=float)
        valid = (
            np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        )
        return np.column_stack(
            [
                ranges[valid] * np.cos(angles[valid]),
                ranges[valid] * np.sin(angles[valid]),
            ]
        )
    
    def _constructMotionWithCovariance(self, linear_vel: float, angular_vel: float, std_dev_linear_vel: float, std_dev_angular_vel: float, dt: float) -> Tuple[np.array, np.array]:
        # s_linear_vel_x = self.std_dev_linear_vel * linear_vel * dt #  5 cm / seg 
        # s_linear_vel_y = 0.000000001 # just a small value as there is no motion along y of the robot
        # s_angular_vel = self.std_dev_angular_vel * angular_vel * dt  # 2 deg / seg

        # Claude, added a floor for low speed 
        s_linear_vel_x = (self.std_dev_linear_vel * abs(linear_vel) + 0.005) * dt
        s_linear_vel_y = 1e-3 * dt
        s_angular_vel  = (self.std_dev_angular_vel * abs(angular_vel) + np.radians(2.0)) * dt

        # compute the motion command [dx, dy, dtheta]. On the real robot we dont add any perterbations
        # Note: this is an approximation but works as time steps are small          
        dx = linear_vel * dt #+ s_linear_vel_x
        dy = 0.0     # there is no motion along y of the robot
        dtheta = angular_vel * dt #+ s_angular_vel

        # Calculate motion command (u) and set it
        motion_command = np.array([[dx], [dy], [dtheta]])

        motion_covariance = np.array([[(s_linear_vel_x)**2, 0.0, 0.0],
                                           [0.0, (s_linear_vel_y)**2, 0.0],
                                           [0.0, 0.0, (s_angular_vel)**2]])

        return motion_command, motion_covariance
    

    @staticmethod
    def _quaternion_from_yaw(yaw) -> tuple:
        """Convert a yaw angle (numpy array of shape (1,)) to a (x, y, z, w) quaternion."""
        half_yaw = yaw[0] * 0.5
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))




