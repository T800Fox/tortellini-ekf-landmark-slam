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
            DeclareLaunchArgument("is_real", default_value="false"),
            Node(
                package="turtlebot_landmark_slam",
                executable="ekf_interface_node.py",
                name="ekf",
                output="screen",
                parameters=[{"is_real": ParameterValue(is_real, value_type=bool)}],
            )
        ]
    )