from threading import Lock
import numpy as np
from sys import exit

from geometry_msgs.msg import Twist

from turtlebot_landmark_slam.types import ControlMeasurement
from turtlebot_landmark_slam.ekf import ExtendedKalmanFilter
from turtlebot_landmark_slam.landmarks import SimLandmarkObserver

class EkfOrchestrator(object):
    def __init__(self, is_real):
        self._ekf = ExtendedKalmanFilter()
        self.real_env = is_real
        self._lock = Lock()

        self.ignore_over_dist = 1.5

        self.lidar_observer = SimLandmarkObserver()

        self.seen_landmark_ids = []

        if self.real_env:
            print('Not configured for real, exiting')
            exit()
            # self.lidar_observer = lidarCylinderObserver(
            #     show_display=True,
            #     max_landmark_dist=self.ignore_over_dist
            #     )
        else:
            self.lidar_observer = SimLandmarkObserver()
            # self.lidar_observer = lidarCylinderObserver(
            #     show_display=True,
            #     max_landmark_dist=5,
            #     max_landmark_count=4,
            #     distance_threshold=0.05,        # 0.05
            #     min_points=4,
            #     max_radius=0.16,                # 0.2          -- higest reading was 0.18
            #     min_radius=0.14,                # 0.1          -- lowest reading was 0.11
            #     max_mse=1.0e-4,                 # 1.0e-4        -- annoying corner case
            #     min_arc_angle=np.radians(90)   # np.radians(90)-- cleared out wall false positives
            #     )

    def motion_handler(self, control_input: ControlMeasurement):
        with self._lock:
            self._ekf.predict(control_input)

    def lidar_handler(self, rel_points):
        with self._lock:
            
            pose = self._ekf.pose

            print("Collecting Landmark Measurements")
            measurements = self.lidar_observer.measure_landmarks()
            if len(measurements) == 0:
                print('empty measurements')
                return
            
            
            # self.lidar_observer.updateLocationData(float(pose[0]), float(pose[1]), float(pose[2]))
            # lidar_observed_landmarks = self.lidar_observer.attemptAssociation(rel_points)
            # empty = self.lidar_observer.inital_landmark_label
            # unlabeled = [lm for lm in lidar_observed_landmarks if lm.label == empty]
            # labeled = [lm for lm in lidar_observed_landmarks if lm.label != empty]

            """
            camera_observed_landmarks = ...
            if there's a match between a camera observed landmark and an unlabeled one, 
            update the label and add to labeled set            
            """
            # this just keeps the ball rolling until then...
            # nextLabel = len(landmarks)
            # for lm in unlabeled:
            #     lm.label = str(nextLabel)
            #     nextLabel += 1
            #     labeled.append(lm)

            for llm in measurements:
                is_new = False
                if llm.id not in self.seen_landmark_ids:
                    self.seen_landmark_ids.append(llm.id)
                    is_new = True

                print("Feeding Measurement -> ", llm)
                self._ekf.update(llm, is_new)


        pass


