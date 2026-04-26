import numpy as np
from turtlebot_landmark_slam.types import LandmarkMeasurement, ControlMeasurement, StoredLandmark
import turtlebot_landmark_slam.utils as utils
from copy import deepcopy


class ExtendedKalmanFilter(object):
    """EKF-SLAM filter that jointly estimates robot pose and landmark positions.

    The state vector has shape (3 + 2*M, 1) where M is the number of observed
    landmarks:  [x, y, yaw, lmk0_x, lmk0_y, lmk1_x, lmk1_y, ...]^T
    """

    def __init__(self) -> None:
        # State vector — grows as new landmarks are discovered (shape N x 1)
        self._state_vector = np.array([[0.0], [0.0], [0.0]])

        sigma_position = np.sqrt(10 ** (-3))
        sigma_orientation = np.sqrt(10 ** (-3))

        # Initial 3x3 robot-pose covariance block (grows to N x N with landmarks)
        self._state_covariance = np.array(
            [
                [sigma_position**2, 0.0, 0.0],
                [0.0, sigma_position**2, 0.0],
                [0.0, 0.0, sigma_orientation**2],
            ]
        )

        # Maps landmark label -> starting row index in the state vector
        # self._landmark_index = {}

        self._tracked_landmarks = []

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def x(self):
        return deepcopy(self._state_vector[0])

    @property
    def y(self):
        return deepcopy(self._state_vector[1])

    @property
    def yaw(self):
        return deepcopy(self._state_vector[2])

    @property
    def pose(self):
        """Robot pose as a (3,) array [x, y, yaw]."""
        return np.array([self.x, self.y, self.yaw], copy=True)

    @property
    def pose_covariance(self):
        """3x3 covariance block for the robot pose."""
        return np.array(self._state_covariance[0:3, 0:3], copy=True)

    @property
    def state_mean(self):
        """Full state vector (robot pose + landmark positions), shape (N, 1)."""
        return np.array(self._state_vector, copy=True)

    @property
    def state_covariance(self):
        """Full N x N state covariance matrix."""
        return np.array(self._state_covariance, copy=True)
    
    @property
    def tracked_landmarks(self) -> list[StoredLandmark]:
        return self._tracked_landmarks

    # ------------------------------------------------------------------
    # Helper Functions
    # ------------------------------------------------------------------

    def get_landmark_by_label(self, req_label):
        result = [l for l in self._tracked_landmarks if l.label == req_label]

        if len(result) == 0:
            raise RuntimeError(f"Landmark w/ label {req_label} not in stored landmarks")
        
        return result[0]

    def extract_landmark_from_state(self, label, state_mean):
        if label not in [l.label for l in self._tracked_landmarks]:
            return np.array([-99, -99])

        state_index = self.get_landmark_by_label(label).index
        return np.array([state_mean[state_index][0], state_mean[state_index+1][0]])

    # def _update_stored_landmarks(self, state_mean):
    #     for l in self._tracked_landmarks:
    #         state_x = float(state_mean[l.index][0])
    #         state_y = float(state_mean[l.index+1][0])

    #         l.abs_x = state_x
    #         l.abs_y = state_y

    def _active_landmarks(self):
        # self._update_stored_landmarks(self.state_mean)

        print("### Currently Tracking ###")
        for l in self._tracked_landmarks:
            print(f"\t{l.label} : ({l.abs_x},{l.abs_y})")


    @staticmethod
    def reguarlise_matrix(S, l=0.1):
        assert S.shape[0] == S.shape[1], "Matrix S must be square."

        # Create an identity matrix of the same size as S
        I = np.eye(S.shape[0])

        # Apply L2 regularization (add lambda * I)
        S_reg = S + l * I

        # Check if S_reg is invertible by computing its determinant
        if np.linalg.det(S_reg) == 0:
            raise ValueError("The regularized matrix is still singular.")

        return S_reg

    # ------------------------------------------------------------------
    # EKF predict step
    # ------------------------------------------------------------------

    def predict(self, control: ControlMeasurement):
        """Propagate the state forward using the motion model.

        Only the robot-pose block [0:3] of the state and covariance is updated;
        landmark estimates are unaffected by the motion model.
        """
        # print("Predict Called")

        motion_command = control.motion_vector
        motion_covariance = control.covariance
        pose = self._state_vector[0:3]
        pose_state_covariance = self.pose_covariance

        predicted_robot_pose, F, W = utils.Relative2AbsolutePose(pose, motion_command)
        np.copyto(self._state_vector[0:3], predicted_robot_pose)
        
        predicted_state_covariance = F @ self.pose_covariance @ F.T + W @ motion_covariance @ W.T
        np.copyto(self._state_covariance[0:3, 0:3], predicted_state_covariance)


    # ------------------------------------------------------------------
    # EKF update step
    # ------------------------------------------------------------------

    def update(self, landmark_measurement: LandmarkMeasurement, is_new: bool):
        """Correct the state estimate using a landmark measurement.

        If `is_new` is True the landmark is appended to the state vector and
        the covariance matrix is augmented before the standard EKF update.
        """
        print(f"Update Called --> tracking: {len(self._tracked_landmarks) }") 

        pose = self.pose
        state_covariance = self.state_covariance
        prior_state = self.state_mean
        pose_covar = self.pose_covariance
        pose = prior_state[0:3]

        if is_new:
            landmark_abs_pos, H1, H2 = utils.Relative2AbsoluteXY(pose, [landmark_measurement.x, landmark_measurement.y])

            insertion_index = prior_state.shape[0]

            new_landmark = StoredLandmark(
                abs_x=float(landmark_abs_pos[0]),
                abs_y=float(landmark_abs_pos[1]),
                index=insertion_index,
                label=landmark_measurement.label,
                covariance=landmark_measurement.covariance
            )
            self._tracked_landmarks.append(deepcopy(new_landmark))

            # self._landmark_index[landmark_measurement.label] = insertion_index
            print(f"Landmark {new_landmark.label} inserted at {new_landmark.index}")

            prior_state = np.vstack((prior_state, landmark_abs_pos))
            
            if len(self._tracked_landmarks) == 1:
                Plx = np.dot(H1, pose_covar)
            else:
                prior_state_covariance = deepcopy(state_covariance)
                Prm = prior_state_covariance[0:3, 3:]
                Plx = np.dot(H1, np.bmat([[pose_covar, Prm]]))
            
            Pll = np.dot(H1, np.dot(pose_covar, H1.T)) + np.dot(H2, np.dot(landmark_measurement.covariance, H2.T))

            P = np.bmat([[state_covariance, Plx.T], [Plx, Pll]])
            state_covariance = np.array(P, copy=True)

        identified_landmark = self.get_landmark_by_label(landmark_measurement.label)
        index = identified_landmark.index
        estimated_landmark = self.extract_landmark_from_state(landmark_measurement.label, prior_state)
        expected_measurement, Hr, Hl = utils.Absolute2RelativeXY(pose, estimated_landmark)

        Z = np.array([[landmark_measurement.x], [landmark_measurement.y]])
        R = landmark_measurement.covariance

        C_cols = state_covariance.shape[0]
        C = np.zeros((2, C_cols))
        C[:, 0:3] = Hr              # dependance on robot pose EXPERIMENTAL
        C[:, index:index+2] = Hl    # dependance on landmark postion

        y = Z - expected_measurement        # INNOVATION
        S = C @ state_covariance @ C.T + R

        if np.linalg.det(S) < 1e-6:
            print(f'WARNING!!! Non-invertible S Matrix {np.linalg.det(S)}')
            # hack by adding reguarlisaer
            S = ExtendedKalmanFilter.reguarlise_matrix(S)

        K = state_covariance @ C.T @ np.linalg.inv(S)
        posterior_state_mean = prior_state + K @ y
        I = np.eye(len(posterior_state_mean))
        posterior_state_covariance = (I - K @ C) @ state_covariance

        # Update state
        # np.copyto(self._state_vector, posterior_state_mean)
        
        self._state_vector = np.array(posterior_state_mean, copy=True)
        self._state_covariance = np.array(posterior_state_covariance, copy=True)
        # self._update_stored_landmarks(self.state_mean)

        state_x = float(self.state_mean[index][0])
        state_y = float(self.state_mean[index+1][0])

        identified_landmark.abs_x = state_x
        identified_landmark.abs_y = state_y

        self._active_landmarks()

    ### 

