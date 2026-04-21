#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud

from turtlebot_landmark_slam.landmark_observers import lidarLandmarkObserver

class LandmarkPublisherLive(Node):
    def __init__(self):
        # Initialize node with the name 'minimal_publisher'
        super().__init__('live_detection_node')
        
        # Create a publisher for 'topic' with queue size of 10
        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self._scan_callback,
            10)
        
        self.landmarkObserver = lidarLandmarkObserver(liveDisplay=True)
        # # live detections plot
        # self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def _scan_callback(self, msg):
        self.landmarkObserver.attemptAssociation(msg)

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = LandmarkPublisherLive()
    
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