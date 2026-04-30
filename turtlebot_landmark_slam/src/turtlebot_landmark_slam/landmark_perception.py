import numpy as np
import cv2

from cv_bridge import CvBridge

import turtlebot_landmark_slam.cone_detection as cd
import turtlebot_landmark_slam.lidar_project_to_image as lpi
import turtlebot_landmark_slam.aruco_detection as ad
from turtlebot_landmark_slam.landmarks_circle_detector import extract_circular_objects
from turtlebot_landmark_slam.types import LandmarkMeasurement, StoredLandmark
from turtlebot_landmark_slam.utils import mahalanobis_distance, euclidianDistance, Relative2AbsoluteXY



class LandMarkPerception():
    def __init__(self, img, lidar, ekf_pose, ekf_pose_covariance, ekf_landmarks, image_publisher):
        self.img_pub = image_publisher

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
        

        self.total_detection = self._perform_perception(img, lidar)
        
        self.landmark_measurement = self._landmark_detection(ekf_pose, ekf_pose_covariance, self.total_detection, ekf_landmarks)

        

        pass
    
    def _perform_perception(self, img, lidar):
        # Expected img subcribe from /camera/image_raw
        # Expected lidar subcribe from /laser or /pointcloud2d
        cone_detect = cd.ConeDetection(img) #.debug_image --> ros2 topic
        lidar_project = lpi.LidarProject(cone_detect.debug_img, lidar)

        lidar_projected_img = lidar_project.img

        aruco_detect_img = ad.ArucoDetection(lidar_projected_img).img

        # msg = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg = CvBridge().cv2_to_imgmsg(aruco_detect_img, encoding="bgr8")
        self.img_pub.publish(img_msg)
        
        r_boxes = [
            box
            for box, colour in cone_detect.boxes
            if colour == "red"
        ]
        
        r_warps = [
            warp
            for warp, colour in cone_detect.warped_images
            if colour == "red"
        ]
        
        g_boxes = [
            box
            for box, colour in cone_detect.boxes
            if colour == "green"
        ]
        
        g_warps = [
            warp
            for warp, colour in cone_detect.warped_images
            if colour == "green"
        ]
        
        b_boxes = [
            box
            for box, colour in cone_detect.boxes
            if colour == "blue"
        ]
        
        b_warps = [
            warp
            for warp, colour in cone_detect.warped_images
            if colour == "blue"
        ]
        
        y_boxes = [
            box
            for box, colour in cone_detect.boxes
            if colour == "yellow"
        ]
        
        y_warps = [
            warp
            for warp, colour in cone_detect.warped_images
            if colour == "yellow"
        ]
        
        r_points = lidar_project.extract_depth_in_box(r_boxes)
        g_points = lidar_project.extract_depth_in_box(g_boxes)
        y_points = lidar_project.extract_depth_in_box(y_boxes)
        b_points = lidar_project.extract_depth_in_box(b_boxes)
        
        r_fit = extract_circular_objects(r_points)
        g_fit = extract_circular_objects(g_points)
        y_fit = extract_circular_objects(y_points)
        b_fit = extract_circular_objects(b_points)
        
        total_fit = []

        for c in r_fit:
            total_fit.append({"colour": "red", "fit": c})

        for c in g_fit:
            total_fit.append({"colour": "green", "fit": c})

        for c in y_fit:
            total_fit.append({"colour": "yellow", "fit": c})

        for c in b_fit:
            total_fit.append({"colour": "blue", "fit": c})
            
        return total_fit
        
    def _landmark_detection(self,  ekf_pose, 
                          ekf_pose_covariance, 
                          detections, 
                          ekf_landmarks):
        landmark_measurements = []
        next_new_id = len(ekf_landmarks)
        
        for det in detections:
            d = det["fit"]
            colour = det["colour"]
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

            print(f"Image -> Detection @ ({d_center_coord_abs[0]},"
                  f"{d_center_coord_abs[1]})")

            # first ever landmark case, just accept all the measurements
            if len(ekf_landmarks) == 0:
                # Landmark Measurements are relative
                measurement_of_initial_landmark = LandmarkMeasurement(
                    x=d.center[0],
                    y=d.center[1],
                    covariance=d_xy_covariance, # ignore theta vals
                    lm_id=next_new_id,
                    colour= colour
                )
                landmark_measurements.append(measurement_of_initial_landmark)

                next_new_id += 1
                continue
            
            # find the landmark closest to the detection
            # consider the pose and xy uncertainty
            closest_dist = 999999
            closest_landmark = None
            for l in ekf_landmarks:
                if l.colour != colour:
                    continue
                
                mahal_dist = mahalanobis_distance(d_center_coord_abs, d_xy_w_pose_covariance,
                                                                    l.mean, l.covariance)
                if mahal_dist < closest_dist:
                    closest_landmark = l
                    closest_dist = mahal_dist

            if closest_landmark == None:
                print("Image -> No closest landmark :(")
                continue

            print(f"\tImage -> Closest Landmark is {closest_landmark.lm_id} --> "
                  f"({closest_landmark.abs_x}, {closest_landmark.abs_y})"
                  f" w/ dist ({closest_dist})")
            
            # decide if the detection classifies as a Landmark Measurement
            if closest_dist < self.mahal_associate_cutoff: 
                print(f"Image -> Picked up {closest_landmark.lm_id} "
                      f"@ ABS->({d_center_coord_abs[0]}, {d_center_coord_abs[1]})")
                
                # Landmark Measurements are relative
                measurement_of_existing_landmark = LandmarkMeasurement(
                    x=d.center[0],
                    y=d.center[1],
                    covariance=d_xy_covariance, # ignore theta vals
                    lm_id=closest_landmark.lm_id,
                    colour= colour
                )
                landmark_measurements.append(measurement_of_existing_landmark)

            elif self.mahal_new_base_val < closest_dist:
                # new landmark window --> not ludicriously big, 
                #                         definitely not misreading of existing landmark
                print(f"\tImage -> New Landmark @ ({d_center_coord_abs[0]},{d_center_coord_abs[1]})")

                # Landmark Measurements are relative
                measurement_of_new_landmark = LandmarkMeasurement(
                    x=d.center[0],
                    y=d.center[1],
                    covariance=d_xy_covariance, # ignore theta vals
                    lm_id=next_new_id,
                    colour= colour
                )
                landmark_measurements.append(measurement_of_new_landmark)
                next_new_id += 1
        
        return landmark_measurements