from threading import Lock

from geometry_msgs.msg import Twist

from turtlebot_landmark_slam.types import ControlMeasurement
from turtlebot_landmark_slam.ekf import ExtendedKalmanFilter

class EkfOrchestrator(object):
    def __init__(self):
        self._ekf = ExtendedKalmanFilter()
        self.real_env = True
        self._lock = Lock()

    def motion_handler(self, control_input: ControlMeasurement):
        with self._lock:
            self._ekf.predict(control_input)

    def lidar_handler(self, rel_points):
        with self._lock:
            landmarks = self._ekf.active_landmarks


        pass

def attempt_association(rel_points, current_landmarks):