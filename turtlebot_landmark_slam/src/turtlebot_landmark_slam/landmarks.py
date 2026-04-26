import numpy as np

from turtlebot_landmark_slam.types import LandmarkMeasurement
from turtlebot_landmark_slam.utils import yaw_from_quaternion

STATIC_OBSTACLE_WORLD_POSITIONS: dict[int, tuple[float, float]] = {
    1: (-1.0, -1.0),  # obstacle_1
    2: (-1.0,  1.0),  # obstacle_2
    3: ( 1.0, -1.0),  # obstacle_3
    4: ( 1.0,  1.0),  # obstacle_4
}

class SimLandmarkObserver(object):

    def __init__(self) -> None:
        pass
        # super().__init__("landmark_publisher_sim")

        # Cached robot pose in the odom/world frame
        self._robot_x: float = 0.0
        self._robot_y: float = 0.0
        self._robot_yaw: float = 0.0
        self._robot_pose_received: bool = False

        # # Measurement variance (m²) on the diagonal of the 2×2 covariance matrix.
        # # types.py: covariance = diag(s_x, s_y), so 0.01 m² → ~0.1 m std dev.
        self.std_dev_landmark_x = 0.01
        # float(
        #     self.declare_parameter("std_dev_landmark_x", 0.01).value
        # )
        self.std_dev_landmark_y = 0.01
        # float(
        #     self.declare_parameter("std_dev_landmark_y", 0.01).value
        # )

        # self.get_logger().info(
        #     f"[LandmarkPublisherSim] std_dev_landmark_x: {self.std_dev_landmark_x}  "
        #     f"std_dev_landmark_y: {self.std_dev_landmark_y}"
        # )

        # self.create_subscription(Odometry, "~/odom", self._odom_callback, 10)

        # self._landmarks_pub = self.create_publisher(LandmarksMsg, "~/landmarks", 10)

        # # Publish at 2 Hz
        # self.create_timer(0.5, self._publish_landmarks)

        # self.get_logger().info("landmark_publisher_sim started")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def update_pose(self, pose) -> None:
        """Cache the robot pose from ground-truth odometry."""
        self._robot_x = pose[0]
        self._robot_y = pose[1]
        self._robot_yaw = pose[2]
        self._robot_pose_received = True

    # ------------------------------------------------------------------
    # Timer: transform obstacles to robot frame and publish
    # ------------------------------------------------------------------

    def measure_landmarks(self) -> list[LandmarkMeasurement]:
        if not self._robot_pose_received:
            print("No pose data")
            return []

        cos_yaw = np.cos(self._robot_yaw)
        sin_yaw = np.sin(self._robot_yaw)

        # landmarks_msg = LandmarksMsg()
        measurements = []
        for id, (wx, wy) in STATIC_OBSTACLE_WORLD_POSITIONS.items():
            # Translate then rotate into robot (base_link) frame
            dx = wx - self._robot_x
            dy = wy - self._robot_y
            rx =  cos_yaw * dx + sin_yaw * dy
            ry = -sin_yaw * dx + cos_yaw * dy

            co_var = np.array([[float(self.std_dev_landmark_x),0],
                               [0,float(self.std_dev_landmark_y)]])

            lm = LandmarkMeasurement(
                x=float(rx),
                y=float(ry),
                covariance=co_var,
                id=id
            )
            measurements.append(lm)
            # print("sim -> ", lm)

            # lm = LandmarkMsg()
            # lm.label = label
            # lm.x = float(rx)
            # lm.y = float(ry)
            # lm.s_x = float(self.std_dev_landmark_x)
            # lm.s_y = float(self.std_dev_landmark_y)
            # landmarks_msg.landmarks.append(lm)

        return measurements
        # self._landmarks_pub.publish(landmarks_msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

