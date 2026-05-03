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
                executable="ekf_pipeline_node.py",
                name="ekf",
                output="screen",
                parameters=[{"is_real": ParameterValue(is_real, value_type=bool)}],
                remappings=[
                    ("~/landmarks", "/landmarks"),
                    ("~/control", "/cmd_vel"),
                ],
                condition=IfCondition(is_real),
            ),
            Node(
                package="turtlebot_landmark_slam",
                executable="ekf_pipeline_node.py",
                name="ekf",
                output="screen",
                parameters=[{
                    "is_real": ParameterValue(is_real, value_type=bool),
                    "std_dev_linear_vel": 0.01,         # std dev lin v
                    "std_dev_angular_vel": 0.05,       # std dev ang v
                }],
                remappings=[
                    ("~/landmarks", "/landmarks"),
                    ("~/gt_odom", "/odom"),
                    ("~/control", "/odom_to_control_republisher/control"),
                ],
                condition=UnlessCondition(is_real),
            ),
        ]
    )
