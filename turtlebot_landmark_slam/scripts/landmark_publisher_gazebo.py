#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry

from sensor_msgs.msg import LaserScan, PointCloud

from landmarks_msg.msg import LandmarkMsg, LandmarksMsg

import numpy as np
from sys import exit

from turtlebot_landmark_slam.landmark_observers import lidarLandmarkObserver

class LandmarkPublisherLiveGazebo(Node):
    def __init__(self):
        # Initialize node with the name 'minimal_publisher'
        super().__init__('live_gazebo_detection_node')
        

        ## Subscribers ## 
        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self._scan_callback,
            10)
        self.odom_subscription =  self.create_subscription(
            Odometry, 
            "~/odom", 
            self._odom_callback, 
            10)
        
        ## Publishers ##
        self._landmarks_pub = self.create_publisher(LandmarksMsg, "~/landmarks", 10)
        self.landmarkObserver = lidarLandmarkObserver(liveDisplay=True)

        self.max_landmarks = 4

    def _scan_callback(self, msg):
        landmarks = self.landmarkObserver.attemptAssociation(msg)

        if len(landmarks) == 0:
            return
        elif len(landmarks) > self.max_landmarks:
            self.get_logger().warning("Picked up too many landmarks! Aborting.")
            exit()
            

        landmarks_msg = LandmarksMsg()
        for l in landmarks:
            lm = LandmarkMsg()
            lm.label = int(l['id'])
            lm.x = float(l['x'])
            lm.y = float(l['y'])
            lm.s_x =float(l['s_x'])
            lm.s_y = float(l['s_y'])
            landmarks_msg.landmarks.append(lm)

        print("[raw] publishing!")
        self._landmarks_pub.publish(landmarks_msg)

    def _odom_callback(self, msg):
        _x = msg.pose.pose.position.x
        _y = msg.pose.pose.position.y
        _yaw = _yaw_from_quaternion(msg.pose.pose.orientation)

        self.landmarkObserver.updateLocationData(_x, _y, _yaw)
        pass


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