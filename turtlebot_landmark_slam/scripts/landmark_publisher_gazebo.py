#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud
from landmarks_msg.msg import LandmarkMsg, LandmarksMsg

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
        
        ## Publishers ##
        self._landmarks_pub = self.create_publisher(LandmarksMsg, "~/landmarks", 10)
        
        self.landmarkObserver = lidarLandmarkObserver(liveDisplay=True)


    def _scan_callback(self, msg):
        landmarks = self.landmarkObserver.attemptAssociation(msg)

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

        print("[raw] publishing!")
        self._landmarks_pub.publish(landmarks_msg)

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