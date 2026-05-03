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
                 executable="landmark_publisher_sim.py",
                 name="landmark_publisher_sim",
                 output="screen",
                 # std dev landmarks
                 parameters=[{"std_dev_landmark_x": 0.01, "std_dev_landmark_y": 0.01}],
                 remappings=[
                     ("~/odom", "/odom"),
                     ("~/landmarks", "/landmarks"),
                 ],
             ),
        ]
    )
