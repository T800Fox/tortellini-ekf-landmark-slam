import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    turtlebot3_gazebo_launch = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"),
        "launch",
        "turtlebot3_dqn_stage2.launch.py",
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable(name="TURTLEBOT3_MODEL", value="burger"),
            DeclareLaunchArgument("is_real", default_value="false"),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(turtlebot3_gazebo_launch)),
            Node(
                package="turtlebot_landmark_slam",
                executable="odom_to_control_republisher.py",
                name="odom_to_control_republisher",
                output="screen",
                remappings=[
                    ("~/odom", "/odom"),
                ],
            ),
            Node(
                package="turtlebot_landmark_slam",
                executable="landmark_publisher_gazebo.py",
                name="landmark_publisher_gazebo",
                output="screen",
                remappings=[
                    ("~/odom", "/odom"),
                    ("~/landmarks", "/landmarks"),
                ]
            ),       
            Node(
                package="turtlebot_landmark_slam",
                executable="ekf_pipeline_node.py",
                name="ekf",
                output="screen",
                parameters=[{"is_real": False}],
                remappings=[
                    ("~/landmarks", "/landmarks"),
                    ("~/control", "/cmd_vel"),
                ],
            )
        ]
    )
