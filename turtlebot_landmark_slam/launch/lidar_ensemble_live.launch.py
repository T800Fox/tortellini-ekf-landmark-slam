import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.actions import TimerAction, ExecuteProcess

def generate_launch_description():

    control_republisher = Node(
                package="turtlebot_landmark_slam",
                executable="odom_to_control_republisher.py",
                name="odom_to_control_republisher",
                output="screen",
                remappings=[
                    ("~/odom", "/odom"),
                ]
            )
    
    live_landmark_publisher = Node(
                package="turtlebot_landmark_slam",
                executable="landmark_publisher_live.py",
                name="landmark_publisher_live",
                output="screen",
                remappings=[
                    ("~/odom", "/odom"),
                    ("~/landmarks", "/landmarks"),
                ]
            )
    
    ekf_pipeline = Node(
                package="turtlebot_landmark_slam",
                executable="ekf_pipeline_node.py",
                name="ekf",
                output="screen",
                parameters=[{"is_real": True}],
                remappings=[
                    ("~/landmarks", "/landmarks"),
                    ("~/control", "/cmd_vel"),
                ],
            )
    
    delayed_set = TimerAction(
        period=2.0,
        actions=[live_landmark_publisher]
    )

    return LaunchDescription([
        control_republisher,
        delayed_set
    ])
