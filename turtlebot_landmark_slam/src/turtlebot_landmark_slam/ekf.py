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
    def tracked_landmarks(self):
        return deepcopy(self._tracked_landmarks)

    # ------------------------------------------------------------------
    # Public Functions
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

        predicted_robot_pose, F, W = utils.Relative2AbsolutePose(pose, motion_command)

        N = self._state_covariance.shape[0]
        F_full = np.eye(N)
        F_full[0:3, 0:3] = F

        np.copyto(self._state_vector[0:3], predicted_robot_pose)

        # predicted_state_covariance = F @ self.pose_covariance @ F.T + W @ motion_covariance @ W.T
        # np.copyto(self._state_covariance[0:3, 0:3], predicted_state_covariance)

        # From Claude, only pose uncertainty was being updated.     
        self._state_covariance = F_full @ self._state_covariance @ F_full.T
        self._state_covariance[0:3, 0:3] += W @ motion_covariance @ W.T
        
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
            # self._landmark_index[landmark_measurement.id] = insertion_index
            print(f"Landmark {landmark_measurement.id} inserted at {insertion_index}")

            

            prior_state = np.vstack((prior_state, landmark_abs_pos))
            
            # if len(self._landmark_index) == 1:
            #     Plx = np.dot(H1, pose_covar)
            # else:
            #     prior_state_covariance = deepcopy(state_covariance)
            #     Prm = prior_state_covariance[0:3, 3:]
            #     Plx = np.dot(H1, np.bmat([[pose_covar, Prm]]))

            # From Claude, works for all landmark counts?
            Plx = H1 @ state_covariance[0:3, :]   
            
            Pll = np.dot(H1, np.dot(pose_covar, H1.T)) + np.dot(H2, np.dot(landmark_measurement.covariance, H2.T))

            P = np.bmat([[state_covariance, Plx.T], [Plx, Pll]])
            state_covariance = np.array(P, copy=True)

            landmark_covariance = np.array(state_covariance[insertion_index:insertion_index+2,
                                                    insertion_index:insertion_index+2], copy=True)

            new_stored = StoredLandmark(
                abs_x=landmark_abs_pos[0],
                abs_y=landmark_abs_pos[1],
                covariance=landmark_covariance,
                index=insertion_index,
                id=landmark_measurement.id
            )
            self._tracked_landmarks.append(new_stored)


        # index = self._landmark_index[landmark_measurement.id]
        identified_landmark = self._extract_landmark_from_stored(landmark_measurement.id)
        index = identified_landmark.index
        estimated_landmark_data = self._extract_landmark_from_state(landmark_measurement.id, prior_state)
        expected_measurement, Hr, Hl = utils.Absolute2RelativeXY(pose, estimated_landmark_data)

        Z = np.array([[landmark_measurement.x], [landmark_measurement.y]])
        R = landmark_measurement.covariance

        C_cols = state_covariance.shape[0]
        C = np.zeros((2, C_cols))
        C[:, 0:3] = Hr                     # From Claude, include pose uncertainty 
        C[:, index:index+2] = Hl       

        y = Z - expected_measurement
        S = C @ state_covariance @ C.T + R

        if np.linalg.det(S) < 1e-6:
            print(f'WARNING!!! Non-invertible S Matrix {np.linalg.det(S)}')
            # hack by adding reguarlisaer
            S = ExtendedKalmanFilter._reguarlise_matrix(S)

        # K = state_covariance @ C.T @ np.linalg.inv(S)
        # posterior_state_mean = prior_state + K @ y
        # I = np.eye(len(posterior_state_mean))
        # posterior_state_covariance = (I - K @ C) @ state_covariance

        # From Claude, apparently it will prevent errors from floating point arithmetic (Joseph Form)
        K = state_covariance @ C.T @ np.linalg.inv(S)
        posterior_state_mean = prior_state + K @ y
        I = np.eye(state_covariance.shape[0])   # size off state_covariance, not state_mean
        IKC = I - K @ C
        posterior_state_covariance = IKC @ state_covariance @ IKC.T + K @ R @ K.T

        # Update state
        # np.copyto(self._state_vector, posterior_state_mean)
        self._state_vector = np.array(posterior_state_mean, copy=True)
        self._state_covariance = np.array(posterior_state_covariance, copy=True)

        # Update tracked landmark data
        updated_landmark_coords = self._extract_landmark_from_state(landmark_measurement.id, self.state_mean)
        updated_landmark_covariance = self._get_landmark_covariance(landmark_measurement.id, self.state_covariance)
        identified_landmark.abs_x = updated_landmark_coords[0]
        identified_landmark.abs_y = updated_landmark_coords[1]
        identified_landmark.covariance = updated_landmark_covariance
        

        self._active_landmarks()

    def update_landmark_label(self, landmark_id, new_label):
        try:
            landmark_to_update = self._extract_landmark_from_stored(landmark_id)
        except RuntimeError:
            print(f"[ERROR] Issue pulling Landmark with id {landmark_id}")
            return

        landmark_to_update.label = new_label
        return

    # ------------------------------------------------------------------
    # Helper Functions
    # ------------------------------------------------------------------

    def _extract_landmark_from_stored(self, id) -> StoredLandmark:
        # print("## Pulling From Stored, Options Are... ##")
        # for l in self._tracked_landmarks:
        #     print(l)

        results = [l for l in self._tracked_landmarks if l.id == id]
        if len(results) != 1:
            raise RuntimeError(f"Issue pulling landmark {id}, found {len(results)}.")

        return results[0]

    def _extract_landmark_from_state(self, id, state_mean):
            state_index = self._extract_landmark_from_stored(id).index
            return np.array([state_mean[state_index][0], state_mean[state_index+1][0]])

    def _get_landmark_covariance(self, id, state_covariance):
        i = self._extract_landmark_from_stored(id).index
        return np.array(state_covariance[i:i+2, i:i+2], copy=True)

    def _active_landmarks(self):
        print("### EKF Currently Tracking ###")
        for l in self._tracked_landmarks:
            print(l)


    @staticmethod
    def _reguarlise_matrix(S, l=0.1):
        assert S.shape[0] == S.shape[1], "Matrix S must be square."

        # Create an identity matrix of the same size as S
        I = np.eye(S.shape[0])

        # Apply L2 regularization (add lambda * I)
        S_reg = S + l * I

        # Check if S_reg is invertible by computing its determinant
        if np.linalg.det(S_reg) == 0:
            raise ValueError("The regularized matrix is still singular.")

        return S_reg

