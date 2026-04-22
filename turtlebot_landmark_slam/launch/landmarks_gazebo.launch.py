from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="turtlebot_landmark_slam",
                executable="landmark_publisher_gazebo.py",
                name="landmark_publisher_gazebo",
                output="screen",
                remappings=[
                    ("~/odom", "/odom"),
                    ("~/landmarks", "/landmarks"),
                ]
            )
        ]
    )
