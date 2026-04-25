from threading import Lock
import numpy as np

from geometry_msgs.msg import Twist

from turtlebot_landmark_slam.types import ControlMeasurement
from turtlebot_landmark_slam.ekf import ExtendedKalmanFilter
from turtlebot_landmark_slam.lidar_landmark_observers import CylinderObserver

class EkfOrchestrator(object):
    def __init__(self, is_real):
        self._ekf = ExtendedKalmanFilter()
        self.real_env = is_real
        self._lock = Lock()

        self.ignore_over_dist = 1.5

        if self.real_env:
            self.lidar_observer = CylinderObserver(
                show_display=True,
                max_landmark_dist=self.ignore_over_dist
                )
        else:
            self.lidar_observer = CylinderObserver(
                show_display=True,
                max_landmark_dist=self.ignore_over_dist,
                distance_threshold=0.05,        # 0.05
                min_points=4,
                max_radius=0.16,                # 0.2          -- higest reading was 0.18
                min_radius=0.14,                # 0.1          -- lowest reading was 0.11
                max_mse=1.0e-4,                 # 1.0e-4        -- annoying corner case
                min_arc_angle=np.radians(90)   # np.radians(90)-- cleared out wall false positives
                )

    def motion_handler(self, control_input: ControlMeasurement):
        with self._lock:
            self._ekf.predict(control_input)

    def lidar_handler(self, rel_points):
        with self._lock:
            landmarks = self._ekf.tracked_landmarks
            pose = self._ekf.pose

            lidar_observed_landmarks = self.lidar_observer.match_landmarks(pose, rel_points, landmarks)
            empty = self.lidar_observer.inital_landmark_label
            unlabeled = [lm for lm in lidar_observed_landmarks if lm.label == empty]
            labeled = [lm for lm in lidar_observed_landmarks if lm.label != empty]

            """
            camera_observed_landmarks = ...
            if there's a match between a camera observed landmark and an unlabeled one, 
            update the label and add to labeled set            
            """
            # this just keeps the ball rolling until then...
            nextLabel = len(landmarks)
            for lm in unlabeled:
                lm.label = str(nextLabel)
                nextLabel += 1
                labeled.append(lm)

            
            for llm in labeled:
                self._ekf.update(llm, llm.is_new)


        pass


