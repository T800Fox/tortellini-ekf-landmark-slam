import numpy as np

import rclpy
from rclpy.node import Node
from threading import Lock

from sensor_msgs.msg import PointCloud, LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TwistStamped

from turtlebot_landmark_slam.ekf import ExtendedKalmanFilter
from turtlebot_landmark_slam.ekf_orchestrator import EkfOrchestrator
from turtlebot_landmark_slam.types import ControlMeasurement

class EkfInterface(object):
    def __init__(self, node: Node, orchestrator: EkfOrchestrator) -> None:
        self._node = node
        self._orchestrator = orchestrator
        self._lock = Lock()

        self._last_motion_msg_time = None
        self._last_small_motion_time = None

        self.std_dev_linear_vel = float(self._node.declare_parameter("std_dev_linear_vel", 0.01).value)
        self.std_dev_angular_vel = float(self._node.declare_parameter("std_dev_angular_vel", (5 * np.pi) / 180).value)

        self._node.get_logger().info(
            f"[DataProvider] std_dev_linear_vel: {self.std_dev_linear_vel}"
        )
        self._node.get_logger().info(
            f"[DataProvider] std_dev_angular_vel: {self.std_dev_angular_vel}"
        )

        self.is_real = bool(self._node.declare_parameter("is_real", False).value)

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
            TwistStamped,
            'cmd_vel',
            self._motion_callback,
            qos_profile=10
        )

        # TODO: Add camera/image_raw subscription

        ## Publishers ## 
        self.odom_publisher = self._node.create_publisher(Odometry, "~/odom", 1)
        self.map_publisher = self._node.create_publisher(MarkerArray, "~/map", 5)


    def _motion_callback(self, msg):
        now = self._node.get_clock().now()

        if self._last_motion_msg_time is None:
            self._last_motion_msg_time = now
            return
        
        dt = (now - self._last_motion_msg_time).nanoseconds / 1e9
        self._last_motion_msg_time = now

        linear_vel = msg.twist.linear.x
        angular_vel = msg.twist.angular.z
        
        # ignore small motions and don't spam log with status messages
        if abs(linear_vel) < 0.009 and abs(angular_vel) < 0.09:
            should_log = (
                self._last_small_motion_time is None 
                or (now - self._last_small_motion_time).nanoseconds / 1e9 >= 10.0
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

        
    def _lidar_callback(self, msg):
        if self.is_real:
            points = np.array([[p.x, p.y] for p in msg.points], dtype=float)
        else:
            points = self._laserscan_to_rel_point(msg)

        self._orchestrator.lidar_handler(points)
        

    # def _image_callback(self, msg):
    #     pass

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
        s_linear_vel_x = self.std_dev_linear_vel * linear_vel * dt #  5 cm / seg 
        s_linear_vel_y = 0.000000001 # just a small value as there is no motion along y of the robot
        s_angular_vel = self.std_dev_angular_vel * angular_vel * dt  # 2 deg / seg

        # compute the motion command [dx, dy, dtheta]. On the real robot we dont add any perterbations
        # Note: this is an approximation but works as time steps are small          
        dx = linear_vel * dt + s_linear_vel_x
        dy = 0.0     # there is no motion along y of the robot
        dtheta = angular_vel * dt + s_angular_vel

        # Calculate motion command (u) and set it
        motion_command = np.array([[dx], [dy], [dtheta]])

        motion_covariance = np.array([[(s_linear_vel_x)**2, 0.0, 0.0],
                                           [0.0, (s_linear_vel_y)**2, 0.0],
                                           [0.0, 0.0, (s_angular_vel)**2]])

        return motion_command, motion_covariance




