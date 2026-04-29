from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    is_real = LaunchConfiguration("is_real")

    return LaunchDescription(
        [
            Node(
                package="turtlebot_landmark_slam",
                executable="ekf_telemetry_display_node.py",
                name="ekf_telemetry_display",
                output="screen",
            )
        ]
    )