import numpy as np
from math import dist
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as pltCircle
from turtlebot_landmark_slam.landmarks_circle_detector import extract_circular_objects, CircleFit
from turtlebot_landmark_slam.utils import Relative2AbsoluteXY
from dataclasses import dataclass
from copy import deepcopy
from sys import exit
from typing import Callable, Tuple

@dataclass
class landmark:
    def __init__(self,
                 mean=None,
                 covariance=None,
                 _id=-1):
        self.mean = np.array([-1,-1])
        self.covariance = np.array([[-1,-1],
                                    [-1,-1]])
        self.id = _id 
        

class lidarLandmarkObserver:
    def __init__(self, liveDisplay):
        assert type(liveDisplay) == bool 
        self.showDisplay = liveDisplay

        if self.showDisplay:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))

        self.storedLandmarks = []

        self.robot_x = -99
        self.robot_y = -99
        self.robot_yaw = -99
        
    def _updateLiveDisplay(self, points, detections):
        self.ax.clear()
        self.ax.set_aspect("equal")

        # Robotic convention: x forward (up), y left (left).
        # Map: plot horizontal = robot Y (inverted), plot vertical = robot X.
        self.ax.set_xlabel("Y (meters)")
        self.ax.set_ylabel("X (meters)")
        self.ax.invert_xaxis()
        self.ax.set_title(f"Live Scan Data - Relative Frame")
        self.ax.grid(True, linestyle=":", alpha=0.6)

        self.ax.plot(
            points[:, 1],
            points[:, 0],
            ".",
            color="lightgray",
            label="Raw scan",
            markersize=4,
            zorder=2,
        )
        self.ax.plot(
            0, 0, "^", color="black", markersize=10, label="Sensor origin", zorder=5
        )

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

        for i, c in enumerate(detections):
            color = COLORS[i % len(COLORS)]
            rng, bearing = c.center
            cx = c.center[0] # rng * np.cos(bearing)
            cy = c.center[1] # rng * np.sin(bearing)

            self.ax.plot(
                c.points[:, 1],
                c.points[:, 0],
                ".",
                color=color,
                markersize=8,
                label=f"Circle {i+1}: r={c.radius:.2f}m",
                zorder=3,
            )
            self.ax.add_patch(
                pltCircle(
                    (cy, cx), c.radius, color=color, fill=False, linewidth=2, zorder=4
                )
            )
            self.ax.plot(cy, cx, "+", color=color, markersize=10, zorder=5)

            # print(
            #     f"Circle {i+1}: ({c.center[0]}, {c.center[1]}) --> range={rng:.3f} m, bearing={np.degrees(bearing):.2f} deg, "
            #     f"radius={c.radius:.3f} m, mse={c.mse:.2e} units, span={c.span:.3e}"
            # )

        self.ax.legend(loc="upper right")
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def _laserscan_to_points(self, msg):
        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        ranges = np.array(msg.ranges, dtype=float)
        valid = (
            np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        )
        return np.column_stack(
            [
                ranges[valid] * np.cos(angles[valid]),
                ranges[valid] * np.sin(angles[valid]),
            ]
        )

    def _mahalanobisDistance(self, detection_mean, detection_covariance, stored_mean, stored_covariance):
        

        dx = float(detection_mean[0] - stored_mean[0])
        dy = float(detection_mean[1] - stored_mean[1])    

        # print('dx ->\n', dx)
        # print('dy ->\n', dy)
        mahalanobis_mean = np.sqrt(dx**2 + dy**2)

        mean_floor = 1e-3  # div. by 0 errors, need small but not zero
        safe_mahalanobis_mean = max(mahalanobis_mean, mean_floor)

        # state variable includes radius w/ x and y
        J_det = np.array([dx / safe_mahalanobis_mean, dy / safe_mahalanobis_mean, 0])
        J_stored = -J_det # Derivative w.r.t stored point is just the negative

        cov_block = np.block([
            [detection_covariance, np.zeros((3, 3))],
            [np.zeros((3, 3)),     stored_covariance]
        ])
        
        # 1x6 Combined Jacobian
        J_combined = np.hstack([J_det, J_stored]).reshape(1, 6)

        covariance_backup = 0.3
        mahalanobis_covariance = (J_combined @ cov_block @ J_combined.T).item()
        if mahalanobis_covariance == 0:
            safe_mahalanobis_covariance = covariance_backup
        else:
            safe_mahalanobis_covariance = mahalanobis_covariance
        
        # print("Detection Mean-->\n",detection_mean)
        # print("Detection Covariance-->\n",detection_covariance)
        # print("Stored Mean-->\n", stored_mean)
        # print("Stored Covariance-->\n", stored_covariance)
        # print("safe_mahalanobis_mean-->\n",safe_mahalanobis_mean)
        # print("safe_mahalandobis_covariance-->\n", safe_mahalanobis_covariance)

        return safe_mahalanobis_mean, safe_mahalanobis_covariance

    def _get_placeholder_id(self):
        # NOTE : REMOVE WHEN INTEGRATING WITH PERCEPTION CODEBASE
        return len(self.storedLandmarks)

    def _combineXYGaussians(self, new_mean, new_covariance, stored_mean, stored_covariance):
        # print('stored mean type -> ', type(stored_mean))
        # print(stored_mean)
        # print('stored covariance type -> ', type(stored_covariance))
        # print('new mean type -> ', type(new_mean))
        # print(new_mean)
        # print('new covariance type -> ', type(new_covariance))

        new_mean_vector = np.array([float(new_mean[0]), float(new_mean[1]), 0])
        stored_mean_vector = np.array([float(stored_mean[0]), float(stored_mean[1]), 0])

        # compute product of two gaussians w/ 'Kalman Gain Form' 
        K_gain = new_covariance @ np.linalg.inv(new_covariance + stored_covariance)

        updated_mean = new_mean_vector + K_gain @ (stored_mean_vector - new_mean_vector)
        updated_covariance = (np.eye(3) - K_gain) @ new_covariance

        # print("updated mean shape -> ", updated_mean.shape)
        # print(updated_mean)

        return updated_mean, updated_covariance

    def updateLocationData(self, new_x, new_y, new_yaw):
        self.robot_x = new_x
        self.robot_y = new_y
        self.robot_yaw = new_yaw
        return

    def attemptAssociation(self, msg):
        if self.robot_x == -99:
            print('[WARN] No position data skipping...')
            return []

        points = self._laserscan_to_points(msg)
        detections = extract_circular_objects(points,
                distance_threshold=0.05,        # 0.05
                min_points=4,
                max_radius=0.16,                 # 0.2          -- higest reading was 0.18
                min_radius=0.14,                 # 0.1          -- lowest reading was 0.11
                max_mse=1.0e-4,                 # 1.0e-4        -- annoying corner case
                max_aspect_ratio=None,          # None
                min_arc_angle=np.radians(90),   # np.radians(90)-- cleared out wall false positives
                min_center_range=None,
                polar=False,)

        if self.showDisplay:
            self._updateLiveDisplay(points, detections)

        # convert detections from relative into global frame
        for i_d in range(len(detections)):
            relative_dist = detections[i_d].center

            current_pose = np.array([[self.robot_x], [self.robot_y], [self.robot_yaw]])
            # print('pose shape -> ', current_pose.shape)

            abs_dist, _, _ = Relative2AbsoluteXY(current_pose, relative_dist)
            detections[i_d].center = abs_dist
            # print('Rel detection -> \n', relative_dist)
            # print('Abs detection -> \n', detections[i_d].center)

        # associate detections w/ with tag...
        associations = []
        for d in detections:

            # just assume the first ever detection isn't a false positive
            if len(self.storedLandmarks) == 0:
                newLandmark = landmark()
                newLandmark.covariance = d.covariance
                newLandmark.mean = np.array(d.center)
                newLandmark.id = self._get_placeholder_id()

                self.storedLandmarks.append(newLandmark)
            if len(self.storedLandmarks) > 8:
                print("Landmark runaway! aborting")
                exit()


            print(f"Detection @ ({d.center[0]},{d.center[1]}) v.s. {len(self.storedLandmarks)} Landmarks")


            # try use mahalanobis to associate w/ existing landmarks
            landmark_match = None
            is_match = False

            closest_landmark = landmark()
            closest_dist = 9999999

            for l in self.storedLandmarks:
                mahal_mean, mahal_covariance = self._mahalanobisDistance(d.center, d.covariance,
                                                                         l.mean, l.covariance)
                
                mahal_dist = mahal_mean * (1/mahal_covariance) * mahal_mean

                if mahal_dist < closest_dist:
                    closest_landmark = l
                    closest_dist = mahal_dist
                    

            print(f"\tClosest Landmark is {closest_landmark.id} --> ({closest_landmark.mean[0]}, {closest_landmark.mean[1]})"
                  f" w/ dist ({closest_dist})")

            if 0 < closest_dist and closest_dist < 100:
                # association window
                landmark_match = closest_landmark

                # update landmark position
                updated_mean, updated_covariance = self._combineXYGaussians(d.center, 
                                                                            d.covariance, 
                                                                            landmark_match.mean, 
                                                                            landmark_match.covariance)
                print(f"\t({landmark_match.mean[0]},{landmark_match.mean[1]})"
                      f" --(becomes)--> ({updated_mean[0]}, {updated_mean[1]})")

                landmark_match.mean = updated_mean
                landmark_match.covariance = updated_covariance

                

            elif 10000 < closest_dist and closest_dist < 30000:
                # new landmark window

                print(f"\tNew Landmark @ ({d.center[0]},{d.center[1]})")
                newLandmark = landmark()
                newLandmark.covariance = d.covariance
                newLandmark.mean = np.array(d.center)
                newLandmark.id = self._get_placeholder_id()

                self.storedLandmarks.append(newLandmark)
                landmark_match = newLandmark

                print('### LANDMARKS ###')
                for l in self.storedLandmarks:
                    print(f"\t\t{l.id} : ({l.mean[0]},{l.mean[1]})")


            if landmark_match != None:
                if landmark_match.id != -1:
                    associations.append({'id':landmark_match.id,
                                        'x':landmark_match.mean[0],
                                        'y':landmark_match.mean[1],
                                        's_x':landmark_match.covariance[0,0],
                                        's_y':landmark_match.covariance[1,1]
                                        })

        return associations



                

                





