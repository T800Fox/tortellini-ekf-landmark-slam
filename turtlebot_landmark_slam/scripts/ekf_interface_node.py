#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from turtlebot_landmark_slam.ekf_interface import EkfInterface
from turtlebot_landmark_slam.ekf_orchestrator import EkfOrchestrator


class EkfInterfaceNode(Node):
    def __init__(self) -> None:
        super().__init__("ekf")

        self.real = bool(self.declare_parameter("is_real", False).value)

        self.interface = EkfInterface(self, EkfOrchestrator(self, self.real))




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