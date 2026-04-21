import numpy as np
from math import dist
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as pltCircle
from turtlebot_landmark_slam.landmarks_circle_detector import extract_circular_objects
from dataclasses import dataclass
from copy import deepcopy

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



    def _updateLiveDisplay(self, points, detections):
        self.ax.clear()
        self.ax.set_aspect("equal")

        # Robotic convention: x forward (up), y left (left).
        # Map: plot horizontal = robot Y (inverted), plot vertical = robot X.
        self.ax.set_xlabel("Y (meters)")
        self.ax.set_ylabel("X (meters)")
        self.ax.invert_xaxis()
        self.ax.set_title(f"Live Scan Data")
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

            print(
                f"Circle {i+1}: ({c.center[0]}, {c.center[1]}) --> range={rng:.3f} m, bearing={np.degrees(bearing):.2f} deg, "
                f"radius={c.radius:.3f} m, mse={c.mse:.2e} units, span={c.span:.3e}"
            )

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
        # j --> detection
        # k --> stored
        eps = 1e-3

        # dx, dy is the vector from stored landmark to detection
        dx = detection_mean[0] - stored_mean[0]
        dy = detection_mean[1] - stored_mean[1]
        
        # This is our "innovation" (1D distance)
        mahalanobis_mean = np.sqrt(dx**2 + dy**2)
        
        # Prevent division by zero for the Jacobian
        safe_mahalanobis_mean = max(mahalanobis_mean, eps)

        # Standard Euclidean Jacobians: [dx/D, dy/D, dr/D]
        # The derivative w.r.t radius is 0 because center distance doesn't depend on radius
        J_det = np.array([dx / safe_mahalanobis_mean, dy / safe_mahalanobis_mean, 0])
        J_stored = -J_det # Derivative w.r.t stored point is just the negative

        # 6x6 Block Covariance
        cov_block = np.block([
            [detection_covariance, np.zeros((3, 3))],
            [np.zeros((3, 3)),     stored_covariance]
        ])
        
        # 1x6 Combined Jacobian
        J_combined = np.hstack([J_det, J_stored]).reshape(1, 6)

        # Project the 6x6 uncertainty down to 1x1 distance uncertainty
        # Variance of the distance measurement
        mahalanobis_covariance = (J_combined @ cov_block @ J_combined.T).item()
        safe_mahalanobis_covariance  = max(mahalanobis_covariance, eps)
        
        print("Detection Mean-->\n",detection_mean)
        print("Detection Covariance-->\n",detection_covariance)
        print("Stored Mean-->\n", stored_mean)
        print("Stored Covariance-->\n", stored_covariance)
        print("safe_mahalanobis_mean-->\n",safe_mahalanobis_mean)
        print("safe_mahalandobis_covariance-->\n", safe_mahalanobis_covariance)

        return safe_mahalanobis_mean, safe_mahalanobis_covariance

        # print("Detection Mean -->\n", detection_mean)
        # print("Stored Mean -->\n", stored_mean)

        # mahalanobis_mean = ((detection_mean[0] - stored_mean[0])**2 + (detection_mean[1] - stored_mean[1])**2)**(1/2)
        
        # # perfect overlap edge case
        # if mahalanobis_mean == 0:
        #     mahalanobis_mean = 0.001 # basically zero, but still a number
        #     denom = mahalanobis_mean**(3/2) # extra **(1/2) already inside mahalanobis_mean
        # else:
        #     denom = ((detection_mean[0] - stored_mean[0])**2 + (detection_mean[1] - stored_mean[1])**2)**(3/4)

        # dD_dx_detection = - (detection_mean[0] - stored_mean[0]) / denom
        # dD_dy_detection = - (detection_mean[1] - stored_mean[1]) / denom
        # J_detection = np.array([dD_dx_detection, dD_dy_detection, 0])
        
        # dD_dx_stored = - (stored_mean[0] - detection_mean[0]) / denom
        # dD_dy_stored = - (stored_mean[1] - detection_mean[1]) / denom
        # J_stored = np.array([dD_dx_stored, dD_dy_stored, 0])

        # # covariance from circle detector is s_x, s_y, s_radius
        # covariance_matrix = np.block([[detection_covariance, np.zeros((3,3))], 
        #                               [np.zeros((3,3)), stored_covariance]])
        # J_combined = np.block([J_detection, J_stored])

        # print("Covariance has shape --> ", covariance_matrix.shape)
        # print("Combined Jacobian has shape --> ", J_combined.shape)

        # mahalanobis_covariance = J_combined @ covariance_matrix @ J_combined.T


    def _get_placeholder_id(self):
        # NOTE : REMOVE WHEN INTEGRATING WITH PERCEPTION CODEBASE
        return len(self.storedLandmarks)

    def attemptAssociation(self, msg):
        points = self._laserscan_to_points(msg)
        detections = extract_circular_objects(points,
                distance_threshold=0.05,        # 0.05
                min_points=4,
                max_radius=0.2,                 # 0.2          -- higest reading was 0.18
                min_radius=0.1,                 # 0.1          -- lowest reading was 0.11
                max_mse=1.0e-4,                 # 1.0e-4        -- annoying corner case
                max_aspect_ratio=None,          # None
                min_arc_angle=np.radians(90),   # np.radians(90)-- cleared out wall false positives
                min_center_range=None,
                polar=False,)

        if self.showDisplay:
            self._updateLiveDisplay(points, detections)

        # associate detections w/ with tag...
        associations = []
        for d in detections:

            # just assume the first ever detection isn't a false positive
            if len(self.storedLandmarks) == 0:
                newLandmark = landmark()
                newLandmark.covariance = d.covariance
                newLandmark.mean = d.center
                newLandmark.id = self._get_placeholder_id()

                self.storedLandmarks.append(newLandmark)

            # try use mahalanobis to associate w/ existing landmarks
            landmarkMatch = landmark()
            isMatch = False
            for l in self.storedLandmarks:
                mahal_mean, mahal_covariance = self._mahalanobisDistance(d.center, d.covariance,
                                                                         l.mean, l.covariance)
                

                mahal_dist = mahal_mean * (1/mahal_covariance) * mahal_mean
                crit_chi_squared = 9.21     # 99% certainty @ 2 dof

                print("mahal_mean -->\n ", mahal_mean)
                print("mahal_covariance -->\n ", mahal_covariance)
                print("mahal_dist -->\n ", mahal_dist)

                if mahal_dist < crit_chi_squared:
                    landmarkMatch = deepcopy(l)
                    isMatch = True
                    break

            # not an existing landmark, add to list and run  w/ it
            if isMatch == False:
                
                newLandmark = landmark()
                newLandmark.covariance = d.covariance
                newLandmark.mean = d.center
                newLandmark.id = self._get_placeholder_id()

                self.storedLandmarks.append(newLandmark)
                landmarkMatch = deepcopy(newLandmark)

            # landmark only worthwhile if it has an id
            if landmarkMatch.id != -1:
                associations.append({'id':landmarkMatch.id,
                                     'x':landmarkMatch.mean[0],
                                     'y':landmarkMatch.mean[1],
                                     's_x':landmarkMatch.covariance[0,0],
                                     's_y':landmarkMatch.covariance[1,1]
                                     })
                
        if len(associations) != 0:
            print("### Associations ###")
        for a in associations:
            print('\t',a)

        return associations



                

                





