"""
MTRX4701 2026 Assignment 3: Simultaneous Localisation and Mapping
File: landmarks.py
Author(s): 530 499 451

Module for generating relative measurements of some existing landmarks.

Simulation version makes up static ones and uses the ground truth of the turtlebot.

Lidar version attempts to associate point clouds (relative) taken from a supplied pose (gaussian), 
with supplied landmarks (gaussian). Detections are done with a tuned version the code provided.

Any possible detections of cylinders are expressed as a detection and converted into the absolute frame
according to the pose. For each detection, the landmark with the closest mahalanobis distance is idenified.
 - distance < 9.21 --> 99.1% certainty that the detection is that landmark.
 - distance > 13.21 --> 99.9% certainty that the detection is not the landmark --> new landmark
 - Anything in the middle is treated as a garbage reading and dumped.

These observation's relative distances from the robot are then passed on in a list of
LandmarkMeasurement's (also gaussian).

If the user wishes, a plot can be displayed of the landmark (cylinder) detections from the robot's
frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as pltCircle

from turtlebot_landmark_slam.types import LandmarkMeasurement, StoredLandmark
from turtlebot_landmark_slam.utils import mahalanobis_distance, euclidianDistance, Relative2AbsoluteXY
from turtlebot_landmark_slam.landmarks_circle_detector import extract_circular_objects

STATIC_OBSTACLE_WORLD_POSITIONS: dict[int, tuple[float, float]] = {
    1: (-1.0, -1.0),  # obstacle_1
    2: (-1.0,  1.0),  # obstacle_2
    3: ( 1.0, -1.0),  # obstacle_3
    4: ( 1.0,  1.0),  # obstacle_4
}

class lidarLandmarkObserver(object):
    def __init__(self,
                show_display,
                max_landmark_dist=None,
                max_landmark_count=None,
                distance_threshold=0.05,        # 0.05
                min_points=4,
                max_radius=0.09,                # 0.2          -- higest reading was 0.18
                min_radius=0.05,                # 0.1          -- lowest reading was 0.11
                max_mse=1.0e-4,                 # 1.0e-4        -- annoying corner case
                min_arc_angle=np.radians(90),   # np.radians(90)-- cleared out wall false positives
                max_aspect_ratio=None,          # None
                min_center_range=None,
                polar=False):
        
        self.show_display = show_display
        self.max_landmark_dist = max_landmark_dist
        self.max_landmark_count = max_landmark_count

        self.circle_distance_threshold = distance_threshold
        self.circle_min_points = min_points
        self.circle_max_radius = max_radius
        self.circle_min_radius = min_radius
        self.circle_max_mse = max_mse
        self.circle_max_aspect_ratio = max_aspect_ratio
        self.circle_min_arc_angle = min_arc_angle
        self.circle_min_center_range = min_center_range
        self.circle_polar = polar

        """
        2 d.o.f. Chi-Squared Table
        -------------------------------------------------------------
        d^2 < 5.99 --> 95% of the possible valid detections will pass
        d^2 < 9.21 --> 99% of the possible valid detections will pass
        d^2 < 13.82 --> 99.9% of possible valid detections will pass 
        -------------------------------------------------------------
        """
        self.mahal_associate_cutoff = 9.21
        self.mahal_new_base_val = 16 # 13.82 wasn't cutting it...

        if self.show_display:
            self.fig, self.ax0 = plt.subplots(figsize=(10, 8))

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def measure_landmarks(self, ekf_pose, 
                          ekf_pose_covariance, 
                          rel_points, 
                          ekf_landmarks):

        # real world lidar can see up to 8m, too much info
        if self.max_landmark_dist is not None:
            close_rel_points = [[point[0], point[1]] for point in rel_points 
                       if euclidianDistance(0,0, point[0], point[1]) < self.max_landmark_dist]
            close_rel_points = np.array(close_rel_points)

            points = close_rel_points
        else:
            points = rel_points

        detections = extract_circular_objects(points,
                distance_threshold=self.circle_distance_threshold,      
                min_points=self.circle_min_points,
                max_radius=self.circle_max_radius,           
                min_radius=self.circle_min_radius,                
                max_mse=self.circle_max_mse,             
                max_aspect_ratio=self.circle_max_aspect_ratio,        
                min_arc_angle=self.circle_min_arc_angle,  
                min_center_range=self.circle_min_center_range,
                polar=self.circle_polar)
        
        if self.show_display:
            self._updateLiveDisplay(points, detections)

        
        landmark_measurements = []
        next_new_id = len(ekf_landmarks)

        for d in detections:
            d_center_coord_abs, H1, H2 = Relative2AbsoluteXY(ekf_pose, d.center)
            d_xy_covariance = d.covariance[0:2, 0:2]

            # Claude
            # Enforce a realistic minimum sensor noise floor.
            # Lidar bearing/range error → ~2-5cm position noise on a circle fit.
            MIN_MEAS_VAR = 0.02 ** 2   # 2 cm std-dev floor; tune to your lidar
            d_xy_covariance = d_xy_covariance + MIN_MEAS_VAR * np.eye(2)

            pose_contribution = H1 @ ekf_pose_covariance @ H1.T
            # d_xy_w_pose_covariance= d_xy_covariance + pose_contribution

            meas_contribution = H2 @ d_xy_covariance @ H2.T 
            d_xy_w_pose_covariance = meas_contribution + pose_contribution

            print(f"Detection @ ({d_center_coord_abs[0]},"
                  f"{d_center_coord_abs[1]})")

            # first ever landmark case, just accept all the measurements
            if len(ekf_landmarks) == 0:
                # Landmark Measurements are relative
                measurement_of_initial_landmark = LandmarkMeasurement(
                    x=d.center[0],
                    y=d.center[1],
                    covariance=d_xy_covariance, # ignore theta vals
                    lm_id=next_new_id
                )
                landmark_measurements.append(measurement_of_initial_landmark)

                next_new_id += 1
                continue

            # find the landmark closest to the detection
            # consider the pose and xy uncertainty
            closest_dist = 999999
            for l in ekf_landmarks:
                mahal_dist = mahalanobis_distance(d_center_coord_abs, d_xy_w_pose_covariance,
                                                                    l.mean, l.covariance)

                if mahal_dist < closest_dist:
                    closest_landmark = l
                    closest_dist = mahal_dist

            print(f"\tClosest Landmark is {closest_landmark.lm_id} --> "
                  f"({closest_landmark.abs_x}, {closest_landmark.abs_y})"
                  f" w/ dist ({closest_dist})")
            
            # decide if the detection classifies as a Landmark Measurement
            if closest_dist < self.mahal_associate_cutoff: 
                print(f"Picked up {closest_landmark.lm_id} "
                      f"@ ABS->({d_center_coord_abs[0]}, {d_center_coord_abs[1]})")
                
                # Landmark Measurements are relative
                measurement_of_existing_landmark = LandmarkMeasurement(
                    x=d.center[0],
                    y=d.center[1],
                    covariance=d_xy_covariance, # ignore theta vals
                    lm_id=closest_landmark.lm_id
                )
                landmark_measurements.append(measurement_of_existing_landmark)

            elif self.mahal_new_base_val < closest_dist:
                # new landmark window --> not ludicriously big, 
                #                         definitely not misreading of existing landmark
                print(f"\tNew Landmark @ ({d_center_coord_abs[0]},{d_center_coord_abs[1]})")

                # Landmark Measurements are relative
                measurement_of_new_landmark = LandmarkMeasurement(
                    x=d.center[0],
                    y=d.center[1],
                    covariance=d_xy_covariance, # ignore theta vals
                    lm_id=next_new_id
                )
                landmark_measurements.append(measurement_of_new_landmark)
                next_new_id += 1
        
        return landmark_measurements

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _updateLiveDisplay(self, rel_points, rel_detections):
        # pretty hud for debugging

        COLORS = [
        "tab:red",
        "tab:green",
        "tab:blue",
        "tab:purple",
        "tab:orange",
        "tab:cyan",
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "tab:gray",
        ]

        ## Display detections in relative frame ##
        # Robotic convention: x forward (up), y left (left).
        # Map: plot horizontal = robot Y (inverted), plot vertical = robot X.
        self.ax0.clear()
        self.ax0.set_aspect("equal")
        self.ax0.set_xlabel("Y (meters)")
        self.ax0.set_ylabel("X (meters)")
        self.ax0.invert_xaxis()
        self.ax0.set_title(f"Live Scan Data - Relative Frame")
        self.ax0.grid(True, linestyle=":", alpha=0.6)

        self.ax0.plot(
            rel_points[:, 1],
            rel_points[:, 0],
            ".",
            color="lightgray",
            label="Raw scan",
            markersize=4,
            zorder=2,
        )
        self.ax0.plot(
            0, 0, "^", color="black", markersize=10, label="Sensor origin", zorder=5
        )

        for i, c in enumerate(rel_detections):
            color = COLORS[i % len(COLORS)]
            rng, bearing = c.center
            cx = c.center[0] # rng * np.cos(bearing)
            cy = c.center[1] # rng * np.sin(bearing)

            self.ax0.plot(
                c.points[:, 1],
                c.points[:, 0],
                ".",
                color=color,
                markersize=8,
                label=f"Circle {i+1}: r={c.radius:.2f}m",
                zorder=3,
            )
            self.ax0.add_patch(
                pltCircle(
                    (cy, cx), c.radius, color=color, fill=False, linewidth=2, zorder=4
                )
            )
            self.ax0.plot(cy, cx, "+", color=color, markersize=10, zorder=5)

            # print(
            #     f"Circle {i+1}: ({c.center[0]}, {c.center[1]}) --> range={rng:.3f} m, bearing={np.degrees(bearing):.2f} deg, "
            #     f"radius={c.radius:.3f} m, mse={c.mse:.2e} units, span={c.span:.3e}"
            # )

        self.ax0.legend(loc="upper right")



        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

class SimLandmarkObserver(object):

    def __init__(self) -> None:
        pass
        # Cached robot pose in the odom/world frame
        self._robot_x: float = 0.0
        self._robot_y: float = 0.0
        self._robot_yaw: float = 0.0
        self._robot_pose_received: bool = False

        # # Measurement variance (m²) on the diagonal of the 2×2 covariance matrix.
        # # types.py: covariance = diag(s_x, s_y), so 0.01 m² → ~0.1 m std dev.
        self.std_dev_landmark_x = 0.01

        self.std_dev_landmark_y = 0.01

    def update_pose(self, pose) -> None:
        """Cache the robot pose from ground-truth odometry."""
        self._robot_x = pose[0]
        self._robot_y = pose[1]
        self._robot_yaw = pose[2]
        self._robot_pose_received = True

    def measure_landmarks(self) -> list[LandmarkMeasurement]:
        if not self._robot_pose_received:
            print("No pose data")
            return []

        cos_yaw = np.cos(self._robot_yaw)
        sin_yaw = np.sin(self._robot_yaw)

        # landmarks_msg = LandmarksMsg()
        measurements = []
        for lm_id, (wx, wy) in STATIC_OBSTACLE_WORLD_POSITIONS.items():
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
                lm_id=lm_id
            )
            measurements.append(lm)

        return measurements
