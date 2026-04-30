#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pickle

from std_msgs.msg import UInt8MultiArray

from turtlebot_landmark_slam.uncertainty_plotter import UncertaintyPlotter


class EkfInterfaceNode(Node):
    def __init__(self) -> None:
        super().__init__("ekf_telemetry_display")

        self.plotter = UncertaintyPlotter(
            log_file_path='ekf_values_log.txt'
        )

        self._ekf_telemetry = self.create_subscription(
            UInt8MultiArray,
            'ekf/telemetry',
            self._telemetry_callback,
            qos_profile=10
        )

    def _telemetry_callback(self, msg):
        serialised_data = bytes(msg.data)
        ekf_data = pickle.loads(serialised_data)

        self.plotter.plot_system(pose=ekf_data['pose'],
                                 pose_covar=ekf_data['pose_covar'],
                                 landmarks=ekf_data['landmarks'],
                                 t=ekf_data['time'])
        
        # TODO: add print to terminal for association!!!!!
        print("Landmark Identities ->")
        for l in ekf_data['landmarks']:
            pretty_id = "?"
            if l.aruco_id != -1:
                pretty_id = l.aruco_id
            print(f"    {l.mean} : {pretty_id}")





def main(args=None) -> None:
    rclpy.init(args=args)
    node = EkfInterfaceNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()