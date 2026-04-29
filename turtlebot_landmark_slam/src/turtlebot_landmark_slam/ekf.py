from copy import deepcopy
import numpy as np

from turtlebot_landmark_slam.types import se2, ControlMeasurement, LandmarkMeasurement, StoredLandmark
import turtlebot_landmark_slam.utils as utils

class ExtendedKalmanFilter(object):

    def __init__(self) -> None:
        self._state_vector = np.array([[0.0], [0.0], [0.0]])

        sigma_position = np.sqrt(10.0 ** -3)
        sigma_orientation = np.sqrt(10.0 ** -3)
        self._state_covariance = np.array(
          [
            [sigma_position ** 2, 0., 0.],
            [0., sigma_position ** 2, 0.],
            [0., 0., sigma_orientation ** 2]
          ]
        )

        self.innovation_ceiling = 0.1

        self._tracked_landmarks = []

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
        return np.array([self.x, self.y, self.yaw], copy=True)

    @property
    def pose_covariance(self):
        return np.array(self._state_covariance[0:3, 0:3], copy=True)

    @property
    def state_mean(self):
        return np.array(self._state_vector, copy=True)

    @property
    def num_landmarks(self):
        landmark_poses = self.state_mean[3:]
        return int(len(landmark_poses)/2)


    # @property
    # def landmarks(self):
    #     landmark_array = [self.landmark(label) for label in self._landmark_index.keys()]
    #     if len(landmark_array) == 0:
    #         return np.array([])
    #     return np.stack(landmark_array)


    # def landmark(self, label):
    #     return self.extract_landmark_from_state(label, self.state_mean)

    def pose_as_se2(self) -> se2:
        x,y,theta = self.pose.flatten()
        return se2(x, y, theta)

    def _extract_landmark_from_stored(self, lm_id) -> StoredLandmark:
        # print("## Pulling From Stored, Options Are... ##")
        # for l in self._tracked_landmarks:
        #     print(l)

        results = [l for l in self._tracked_landmarks if l.lm_id == lm_id]
        if len(results) != 1:
            raise RuntimeError(f"Issue pulling landmark {lm_id}, found {len(results)}.")

        return results[0]

    def extract_landmark_from_state(self, lm_id, state_mean):
        state_index = self._extract_landmark_from_stored(lm_id).index
        return np.array([state_mean[state_index][0], state_mean[state_index+1][0]])
    
    def get_landmark_covariance(self, lm_id, state_covariance):
            i = self._extract_landmark_from_stored(lm_id).index
            return np.array(state_covariance[i:i+2, i:i+2], copy=True)
    
    def active_landmarks(self):
        print("### EKF Currently Tracking ###")
        for l in self._tracked_landmarks:
            print(l)


    @property
    def state_covariance(self):
        return np.array(self._state_covariance, copy=True)
    
    @property
    def tracked_landmarks(self):
        return deepcopy(self._tracked_landmarks)

    def predict(self, control_meaurement: ControlMeasurement):
        motion_command = control_meaurement.motion_vector
        motion_covariance = control_meaurement.covariance
        pose = self._state_vector[0:3]
        predicted_robot_pose, F, W = utils.motion_model(pose, motion_command)
        np.copyto(self._state_vector[0:3], predicted_robot_pose)

        # # Update robot-robot covariance block
        # P_rr = self._state_covariance[0:3, 0:3]
        # self._state_covariance[0:3, 0:3] = F @ P_rr @ F.T + W @ motion_covariance @ W.T

        # # Update robot-landmark cross-covariance blocks (F_aug applied to full P)
        # n = self._state_covariance.shape[0]
        # if n > 3:
        #     P_rl = self._state_covariance[0:3, 3:]
        #     self._state_covariance[0:3, 3:] = F @ P_rl
        #     self._state_covariance[3:, 0:3] = (F @ P_rl).T
        

        # Claude, this chunk really helped with the weird decay of landmark uncertainty
        # Something about information getting leaked from the covariance at each run?
        # Snapshot the prior blocks BEFORE writing back (numpy slices return
        # views, so without .copy() the second use of P_rl would read the
        # already-updated values, applying F twice on one side of the
        # cross-covariance).
        P_rr = self._state_covariance[0:3, 0:3].copy()
        self._state_covariance[0:3, 0:3] = F @ P_rr @ F.T + W @ motion_covariance @ W.T

        n = self._state_covariance.shape[0]
        if n > 3:
            P_rl = self._state_covariance[0:3, 3:].copy()
            new_P_rl = F @ P_rl
            self._state_covariance[0:3, 3:] = new_P_rl
            self._state_covariance[3:, 0:3] = new_P_rl.T       


    def update(self, landmark_measurements: list[LandmarkMeasurement]):
        if len(landmark_measurements) == 0:
            return

        pose = self.pose
        state_covariance = self.state_covariance
        x = self.state_mean  # Prior state mean

        C_list = []
        Z_list = []
        expected_measurements_list = []
        R_list = []
        landmark_list = []

        # construct and update size of state cov if new landmarks needed
        for landmark_measurement in landmark_measurements:
            lm_id = landmark_measurement.lm_id

            if lm_id not in [l.lm_id for l in self._tracked_landmarks]:

                print(f"Gotten new landmark {landmark_measurement.lm_id}")
                # New landmark detected
                landmark_measured_abs, H1, H2 = utils.inverse_sensor_model(pose, [landmark_measurement.x, landmark_measurement.y])

                index = x.shape[0]
                print(f"Insertion index {index}")
                # self._landmark_index[label] = index
                x = np.vstack((x, landmark_measured_abs))

                Prr = self.pose_covariance
                Plx = H1 @ state_covariance[0:3, :]
                # if len(self._tracked_landmarks) == 1:
                #     Plx = np.dot(H1, Prr)
                # else:
                #     last_state_covariance = deepcopy(state_covariance)
                #     Prm = last_state_covariance[0:3, 3:]
                #     Plx = np.dot(H1, np.bmat([[Prr, Prm]]))

                # Pll = np.dot(H1, np.dot(Prr, H1.T)) + np.dot(H2, np.dot(landmark_measurement.covariance, H2.T))

                # P = np.bmat([[state_covariance, Plx.T], [Plx, Pll]])
                # state_covariance = np.array(P, copy=True)
                Pll = H1 @ Prr @ H1.T + H2 @ landmark_measurement.covariance @ H2.T

                state_covariance = np.block([
                    [state_covariance, Plx.T],
                    [Plx,              Pll  ],
                ])


                landmark_covariance = np.array(state_covariance[index:index+2,
                                                    index:index+2], copy=True)
                new_stored = StoredLandmark(
                    abs_x=landmark_measured_abs[0],
                    abs_y=landmark_measured_abs[1],
                    covariance=landmark_covariance,
                    index=index,
                    lm_id=landmark_measurement.lm_id
                )
                self._tracked_landmarks.append(new_stored)


        for landmark_measurement in landmark_measurements:
            lm_id = landmark_measurement.lm_id
            estimated_landmark = self.extract_landmark_from_state(lm_id, x)
            estimated_landmark_data = self._extract_landmark_from_stored(lm_id)

            expected_measurement, Hr, Hl = utils.sensor_model(pose, estimated_landmark)

            Z_l = np.array([[landmark_measurement.x], [landmark_measurement.y]])
            y_l = Z_l - expected_measurement

            # yc -> innovation for coordinate
            accept_observation = True
            for i, yc in enumerate(y_l):
                if abs(yc) > self.innovation_ceiling:
                    legend = ['x', 'y']
                    print(f"Rejecting Obs. of {estimated_landmark_data.lm_id};"
                          f" {legend[i]} innovation abs({yc}) > {self.innovation_ceiling}")
                    accept_observation = False

            if accept_observation:
                estimated_landmark_data.innovation_x = float(y_l[0])
                estimated_landmark_data.innovation_x = float(y_l[1])

                expected_measurements_list.append(expected_measurement)
                Z_list.append(Z_l)
                R_list.append(landmark_measurement.covariance)

                # Construct the C matrix
                index = estimated_landmark_data.index
                # landmark_list.append(identified_landmark)

                C_cols = state_covariance.shape[0]
                C = np.zeros((2, C_cols))
                C[:, :3] = Hr

                # add Hl at correct index
                C[:, index:index+2] = Hl
                C_list.append(C)

        if len(Z_list) == 0:
            print("All observations exceeded Innovation Ceiling; skipping update...")
            return

        # Stack all measurements and Jacobians
        C = np.vstack(C_list)
        Z = np.vstack(Z_list)

        R = ExtendedKalmanFilter.block_diag(R_list)

        expected_measurements = np.vstack(expected_measurements_list)

        y = Z - expected_measurements  # Innovation term
        S = C @ state_covariance @ C.T + R

        # # associate innovations to stored landmark objects for logging
        # for i, l in enumerate(landmark_list):
        #     l.innovation_x = y[2*i]
        #     l.innovation_y = y[2*i + 1]




        if np.linalg.det(S) < 1e-9:
            print(f'WARNING!!! Non-invertible S Matrix {np.linalg.det(S)}, Regularising...')
            # hack by adding reguarlisaer
            S = ExtendedKalmanFilter.reguarlise_matrix(S)

        # K = state_covariance @ C.T @ np.linalg.inv(S)
        # posterior_state_mean = x + K @ y
        # I = np.eye(len(posterior_state_mean))
        # posterior_state_covariance = (I - K @ C) @ state_covariance


        # From Claude, Joseph form ensures the covariance mat stays PSD and symmetic
        # was brought in during the landmark uncertainty decay, but not sure if it was IT for the fix
        K = state_covariance @ C.T @ np.linalg.inv(S)
        posterior_state_mean = x + K @ y

        I = np.eye(len(posterior_state_mean))
        IKC = I - K @ C

        # Joseph form: numerically stable, guaranteed symmetric & PSD as long
        # as inputs are. Replaces the simpler (I - KC) P form.
        posterior_state_covariance = IKC @ state_covariance @ IKC.T + K @ R @ K.T

        # Enforce symmetry against accumulated round-off
        posterior_state_covariance = 0.5 * (posterior_state_covariance + posterior_state_covariance.T)

        # wrap pi vals
        theta = posterior_state_mean[2]
        wrapped = self.wrap_to_minus_pi_pi(theta)
        posterior_state_mean[2] = wrapped

        # Update state
        self._state_vector = np.array(posterior_state_mean, copy=True)
        self._state_covariance = np.array(posterior_state_covariance, copy=True)

        # Resync ALL tracked landmarks from the updated state
        for stored in self._tracked_landmarks:
            coords = self.extract_landmark_from_state(stored.lm_id, self.state_mean)
            stored.abs_x = coords[0]
            stored.abs_y = coords[1]
            stored.covariance = self.get_landmark_covariance(stored.lm_id, self.state_covariance)       


    """
    I have a memory that this l value is the expected covariance of lidar values?
    """
    @staticmethod
    def reguarlise_matrix(S, l=0.0025):# l= 0.1
        assert S.shape[0] == S.shape[1], "Matrix S must be square."

        # Create an identity matrix of the same size as S
        I = np.eye(S.shape[0])

        # Apply L2 regularization (add lambda * I)
        S_reg = S + l * I

        # Check if S_reg is invertible by computing its determinant
        if np.linalg.det(S_reg) == 0:
            raise ValueError("The regularized matrix is still singular.")

        return S_reg

    @staticmethod
    def block_diag(R_list):
        """
        Constructs a block diagonal measurement covariance matrix R
        from a list of individual measurement covariance matrices using only NumPy.

        Args:
            R_list (list of np.ndarray): List of (2x2) covariance matrices R_i.

        Returns:
            np.ndarray: The block diagonal measurement covariance matrix.
        """
        # Compute total size of the resulting block matrix
        rows = sum(R.shape[0] for R in R_list)
        cols = sum(R.shape[1] for R in R_list)

        # Initialize a zero matrix of the required size
        R = np.zeros((rows, cols))

        # Fill the block diagonal positions
        row_start, col_start = 0, 0
        for R_i in R_list:
            r, c = R_i.shape
            R[row_start:row_start+r, col_start:col_start+c] = R_i
            row_start += r
            col_start += c

        return R
    
    @staticmethod
    def wrap_to_minus_pi_pi(x):
        """Wraps angles (in radians) to the range [-pi, pi]."""
        return np.arctan2(np.sin(x), np.cos(x))
     