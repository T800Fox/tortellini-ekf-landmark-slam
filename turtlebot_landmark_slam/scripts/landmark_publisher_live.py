#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud
from landmarks_msg.msg import LandmarkMsg, LandmarksMsg

import numpy as np
from sys import exit

from turtlebot_landmark_slam.landmark_observers import lidarCylinderObserver

class LandmarkPublisherLiveGazebo(Node):
    def __init__(self):
        super().__init__('live_landmarks_node')
        
        ## Subscribers ## 
        self.scan_subscription = self.create_subscription(
            PointCloud,
            '/pointcloud2d',
            self._scan_callback,
            10)
        self.odom_subscription =  self.create_subscription(
            Odometry, 
            "ekf/odom", 
            self._odom_callback, 
            10)
        
        ## Publishers ##
        self._landmarks_pub = self.create_publisher(LandmarksMsg, "~/landmarks", 10)
        self.landmarkObserver = lidarCylinderObserver(liveDisplay=True,
                                                      distance_threshold=0.05,        # 0.05
                                                      min_points=4,
                                                      max_radius=0.16,                # 0.2          -- higest reading was 0.18
                                                      min_radius=0.14,                # 0.1          -- lowest reading was 0.11
                                                      max_mse=1.0e-4,                 # 1.0e-4        -- annoying corner case
                                                      max_aspect_ratio=None,          # None
                                                      min_arc_angle=np.radians(90),   # np.radians(90)-- cleared out wall false positives
                                                      min_center_range=None,
                                                      polar=False)


    def _scan_callback(self, msg):
        points = laserscan_to_rel_point(msg)
        landmarks = self.landmarkObserver.attemptAssociation(points)

        if len(landmarks) == 0:
            return

        landmarks_msg = LandmarksMsg()
        for l in landmarks:
            lm = LandmarkMsg()
            lm.label = int(l['id'])
            lm.x = float(l['x'])
            lm.y = float(l['y'])
            lm.s_x =float(l['s_x'])
            lm.s_y = float(l['s_y'])
            landmarks_msg.landmarks.append(lm)

        self._landmarks_pub.publish(landmarks_msg)

    def _odom_callback(self, msg):
        _x = msg.pose.pose.position.x
        _y = msg.pose.pose.position.y
        _yaw = _yaw_from_quaternion(msg.pose.pose.orientation)

        self.landmarkObserver.updateLocationData(_x, _y, _yaw)
        pass


def laserscan_to_rel_point(msg):
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

def _yaw_from_quaternion(q) -> float:
    """Extract yaw (rotation about Z) from a geometry_msgs Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = LandmarkPublisherLiveGazebo()
    
    try:
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        minimal_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()