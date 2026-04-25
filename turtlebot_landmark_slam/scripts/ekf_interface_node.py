#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from turtlebot_landmark_slam.ekf_interface import EkfInterface
from turtlebot_landmark_slam.ekf_orchestrator import EkfOrchestrator


class EkfInterfaceNode(Node):
    def __init__(self) -> None:
        super().__init__("ekf_interface")
        self.interface = EkfInterface(self, EkfOrchestrator())


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