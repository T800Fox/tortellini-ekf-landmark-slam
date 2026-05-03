import numpy as np
from sys import exit
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as pltCircle
from dataclasses import dataclass

from turtlebot_landmark_slam.landmarks_circle_detector import extract_circular_objects
from turtlebot_landmark_slam.utils import Relative2AbsoluteXY, Absolute2RelativeXY

@dataclass
class landmark:
    # struct for storing landmarks
    # all vals. init. to -1 by default

    def __init__(self,
                 mean=None,
                 covariance=None,
                 _id=-1):
        self.mean = np.array([-1,-1])
        self.covariance = np.array([[-1,-1],
                                    [-1,-1]])
        self.id = _id 
        
class lidarCylinderObserver:
    def __init__(self, 
                 abortCount=4,
                 maxLandmarkDistance=5,
                 liveDisplay=False,
                 distance_threshold=0.05,        # 0.05
                 min_points=4,
                 max_radius=0.09,                # 0.2          -- higest reading was 0.18
                 min_radius=0.05,                # 0.1          -- lowest reading was 0.11
                 max_mse=1.0e-4,                 # 1.0e-4        -- annoying corner case
                 max_aspect_ratio=None,          # None
                 min_arc_angle=np.radians(90),   # np.radians(90)-- cleared out wall false positives
                 min_center_range=None,
                 polar=False):
        
        assert type(liveDisplay) == bool 
        self.showDisplay = liveDisplay

        self.maxLandmarks = abortCount

        # yes this is ugly, but the same job is being done in both envs. (live vs. gazebo)
        # circle detection may need to be tuned for each one
        self.circle_distance_threshold = distance_threshold
        self.circle_min_points = min_points
        self.circle_max_radius = max_radius
        self.circle_min_radius = min_radius
        self.circle_max_mse = max_mse
        self.circle_max_aspect_ratio = max_aspect_ratio
        self.circle_min_arc_angle = min_arc_angle
        self.circle_min_center_range = min_center_range
        self.circle_polar = polar

        self.storedLandmarks = []   # list of landmarks, LOL
        self.max_landmark_distance = maxLandmarkDistance # ignore landmarks over (m)

        # robot state data for (rel. --> abs.) conversions
        self.robot_x = 0
        self.robot_y =  0
        self.robot_yaw = 0
        self.odom_recieved = False

        if self.showDisplay:
            self.fig, (self.ax0, self.ax1) = plt.subplots(1, 2, figsize=(10, 8))

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def updateLocationData(self, new_x, new_y, new_yaw):
        # Take in current state data 

        self.robot_x = new_x
        self.robot_y = new_y
        self.robot_yaw = new_yaw
        self.odom_recieved = True
        return

    def attemptAssociation(self, rel_points):
        # For some points (relative frame) pull out circles and try 
        # associate them to landmarks. 
        # Landmark Id's are based on order of discovery, NOT camera data

        if not self.odom_recieved:
            print('[WARN] No position data skipping...')
            return []

        close_points = [[point[0], point[1]] for point in rel_points 
                       if self._euclidianDistance(0,0, point[0], point[1]) < self.max_landmark_distance]
        close_points = np.array(close_points)

        detections = extract_circular_objects(close_points,
                distance_threshold=self.circle_distance_threshold,      
                min_points=self.circle_min_points,
                max_radius=self.circle_max_radius,           
                min_radius=self.circle_min_radius,                
                max_mse=self.circle_max_mse,             
                max_aspect_ratio=self.circle_max_aspect_ratio,        
                min_arc_angle=self.circle_min_arc_angle,  
                min_center_range=self.circle_min_center_range,
                polar=self.circle_polar)

        if self.showDisplay:
            self._updateLiveDisplay(close_points, detections)

        # convert detections from relative into global frame
        current_pose = np.array([[self.robot_x], [self.robot_y], [self.robot_yaw]])
        for i_d in range(len(detections)):
            relative_dist = detections[i_d].center
            # print('pose shape -> ', current_pose.shape)

            abs_dist, _, _ = Relative2AbsoluteXY(current_pose, relative_dist)
            detections[i_d].center = abs_dist
            # print('Rel detection -> \n', relative_dist)
            # print('Abs detection -> \n', detections[i_d].center)

        # try associate detections w/ with tag...
        associations = []
        for d in detections:
            # just assume the first ever detection isn't a false positive
            if len(self.storedLandmarks) == 0:
                newLandmark = landmark()
                newLandmark.covariance = d.covariance
                newLandmark.mean = np.array(d.center)
                newLandmark.id = self._get_placeholder_id()

                self.storedLandmarks.append(newLandmark)
                
            if len(self.storedLandmarks) > self.maxLandmarks:
                print("Landmark runaway! aborting")
                exit()

            print(f"Detection @ ({d.center[0]},{d.center[1]}) v.s. {len(self.storedLandmarks)} Landmarks")

            # closest existing landmark to detection w/ mahalanobis
            closest_landmark = landmark()
            closest_dist = 9999999
            landmark_match = None

            for l in self.storedLandmarks:
                mahal_mean, mahal_covariance = self._mahalanobisDistance(d.center, d.covariance,
                                                                         l.mean, l.covariance)
                
                mahal_dist = mahal_mean * (1/mahal_covariance) * mahal_mean

                if mahal_dist < closest_dist:
                    closest_landmark = l
                    closest_dist = mahal_dist
                    
            print(f"\tClosest Landmark is {closest_landmark.id} --> ({closest_landmark.mean[0]}, {closest_landmark.mean[1]})"
                  f" w/ dist ({closest_dist})")


            if closest_dist < 9.21: # 99.9% Certianty chi-squard val for 2 d.o.f.
                # within association window --> definitley the landmark
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


            elif 40000 < closest_dist and closest_dist < 60000:
                # new landmark window --> not ludicriously big, 
                #                         definitely not misreading of existing landmark

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

            # only give landmarks that are associated with an id / label
            if landmark_match != None:
                if landmark_match.id != -1:
                    relativeXY, _, _  = Absolute2RelativeXY(current_pose, 
                                                            landmark_match.mean.flatten())

                    associations.append({'id':landmark_match.id,
                                        'x':float(relativeXY[0,0]),
                                        'y':float(relativeXY[1,0]),
                                        's_x':landmark_match.covariance[0,0],
                                        's_y':landmark_match.covariance[1,1]
                                        })

        return associations

    # ------------------------------------------------------------------
    # Helper Functions
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

        ## Stored Landmarks in Absolute Frame
        self.ax1.clear()
        self.ax1.set_aspect("equal")
        self.ax1.set_xlabel("Y (meters)")
        self.ax1.set_ylabel("X (meters)")
        self.ax1.invert_xaxis()
        self.ax1.set_title(f"Stored Landmarks - Absolute Frame")
        self.ax1.grid(True, linestyle=":", alpha=0.6)

        landmark_size = ( self.circle_min_radius + self.circle_max_radius ) / 2

        if self.odom_recieved:
            self.ax1.plot(self.robot_x, self.robot_y, "o", 
                          color="black", 
                          markersize=10, 
                          label="Sensor origin")

        for i, l in enumerate(self.storedLandmarks):
            color = COLORS[i % len(COLORS)]
            
            self.ax1.plot(l.mean[0], l.mean[1], "x",
                          color=color,
                          label=f"({round(float(l.mean[0]), 4)}, {round(float(l.mean[1]), 4)})")

            self.ax1.add_patch(
                pltCircle((l.mean[0], l.mean[1]), landmark_size, color=color, fill=False)
                )

        self.ax1.legend(loc="upper center")

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def _euclidianDistance(self, x1, y1, x2, y2):
        dx = float(x2 - x1)
        dy = float(y2 - y1)    

        return np.sqrt(dx**2 + dy**2)

    def _mahalanobisDistance(self, detection_mean, detection_covariance, stored_mean, stored_covariance):
        # Distance between Gaussians, w/ safety! 
        # if mean is too small defaults to, mean = 1e-3 and covariance = 0.3

        mahalanobis_mean = self._euclidianDistance(detection_mean[0], detection_mean[1],
                                                   stored_mean[0], stored_mean[1])


        # # print('dx ->\n', dx)
        # # print('dy ->\n', dy)
        # mahalanobis_mean = np.sqrt(dx**2 + dy**2)

        mean_floor = 1e-3  # for div. by 0 errors, need small but not zero
        safe_mahalanobis_mean = max(mahalanobis_mean, mean_floor)

        # state variable includes radius w/ x and y
        dx = float(detection_mean[0] - stored_mean[0])
        dy = float(detection_mean[1] - stored_mean[1])   

        J_det = np.array([dx / safe_mahalanobis_mean, dy / safe_mahalanobis_mean, 0])
        J_stored = -J_det # Derivative w.r.t stored point is just the negative

        cov_block = np.block([
            [detection_covariance, np.zeros((3, 3))],
            [np.zeros((3, 3)),     stored_covariance]
        ])
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
        # Compute average of two Gaussians

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





 