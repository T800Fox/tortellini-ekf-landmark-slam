#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud

from matplotlib.patches import Circle as pltCircle
from turtlebot_landmark_slam.landmarks_circle_detector import extract_circular_objects

class MinimalPublisher(Node):
    def __init__(self):
        # Initialize node with the name 'minimal_publisher'
        super().__init__('live_detection_node')
        
        # Create a publisher for 'topic' with queue size of 10
        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self._scan_callback,
            10)
        
        # live detections plot
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        

    def _scan_callback(self, msg):
        print("[raw] Scan callback fired")
        points = self._laserscan_to_points(msg)
        detections = extract_circular_objects(points)

        self.ax.clear()
        self.ax.set_aspect("equal")
        # Robotic convention: x forward (up), y left (left).
        # Map: plot horizontal = robot Y (inverted), plot vertical = robot X.
        self.ax.set_xlabel("Y (meters)")
        self.ax.set_ylabel("X (meters)")
        self.ax.invert_xaxis()
        self.ax.set_title(f"Live Scan Data")
        self.ax.grid(True, linestyle=":", alpha=0.6)

        self.ax.plot(
            points[:, 1],
            points[:, 0],
            ".",
            color="lightgray",
            label="Raw scan",
            markersize=4,
            zorder=2,
        )
        self.ax.plot(
            0, 0, "^", color="black", markersize=10, label="Sensor origin", zorder=5
        )

        COLORS = [
        "tab:red",
        "tab:green",
        "tab:blue",
        "tab:purple",
        "tab:orange",
        "tab:cyan",
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "tab:gray",
        ]

        for i, c in enumerate(detections):
            color = COLORS[i % len(COLORS)]
            rng, bearing = c.center
            cx = c.center[0] # rng * np.cos(bearing)
            cy = c.center[1] # rng * np.sin(bearing)

            self.ax.plot(
                c.points[:, 1],
                c.points[:, 0],
                ".",
                color=color,
                markersize=8,
                label=f"Circle {i+1}: r={c.radius:.2f}m",
                zorder=3,
            )
            self.ax.add_patch(
                pltCircle(
                    (cy, cx), c.radius, color=color, fill=False, linewidth=2, zorder=4
                )
            )
            self.ax.plot(cy, cx, "+", color=color, markersize=10, zorder=5)

            print(
                f"Circle {i+1}: ({c.center[0]}, {c.center[1]}) --> range={rng:.3f} m, bearing={np.degrees(bearing):.2f} deg, "
                f"radius={c.radius:.3f} m, mse={c.mse:.2e} units, span={c.span:.3e}"
            )
        self.ax.legend(loc="upper right")
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

        return
        


    def _laserscan_to_points(self, msg):
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


def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    
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