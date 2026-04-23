from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="turtlebot_landmark_slam",
                executable="landmark_publisher_live.py",
                name="landmark_publisher_live",
                output="screen",
                remappings=[
                    ("~/odom", "/odom"),
                    ("~/landmarks", "/landmarks"),
                ]
            )
        ]
    )
