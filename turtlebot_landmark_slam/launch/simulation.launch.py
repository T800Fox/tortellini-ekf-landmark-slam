import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
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
            IncludeLaunchDescription(PythonLaunchDescriptionSource(turtlebot3_gazebo_launch))
        ]
    )
