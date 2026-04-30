import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import json
from pathlib import Path


class LidarProject():
    def __init__(self, img, lidar):
        self.camera_k = np.array([(503.4, 0.0, 319.3),
                        (0.0, 505.2, 233.4),
                        (0.0, 0.0, 1.0)])
        self.camera_dist = np.array([(0.2031, -0.4606, 0.0002772, 0.0006714, 0.3447)])

        self.icp_result = np.array([[ 0.99999723, -0.00235341, -0.07921775],
                        [0.00235341,  0.99999723,  0.00144672],
                        [ 0., 0., 1.]])

        # Assign img FIRST, then use it
        self.img = img

        h, w = self.img.shape[:2]

        self.new_camera_k, _ = cv2.getOptimalNewCameraMatrix(
        self.camera_k, self.camera_dist, (w, h), 1, (w, h))
        self.new_camera_dist = np.zeros((1, 5))

        self.img = cv2.undistort(
            self.img, self.camera_k, self.camera_dist, None, self.new_camera_k)

        self.img_pts, _, self.lidar_pts = self.lidar_projection_pipeline(lidar)
        self.draw_lidar_points()
    
    def plot_top_down_view(self, cam_lidar_h):
        # extract coordinates
        x = cam_lidar_h[:, 0]
        y = cam_lidar_h[:, 1]
        
        plt.figure(figsize=(6,6))
        plt.scatter(x, y, s=2)

        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.title("LiDAR Top View")

        plt.axis("equal")   # VERY IMPORTANT → preserves geometry
        plt.grid()

        plt.show()

    def lidar_to_image_projection(self, rel_points, image):
        h, w = self.img.shape[:2]
        
        pts = rel_points 
        
        lidar_pts = pts[:, :2]
        
        lidar_h = np.hstack((lidar_pts, np.zeros((lidar_pts.shape[0], 1))))
        depth_mask = (lidar_h[:, 0] > 0) & (lidar_h[:, 0] < 2.5)
        side_mask  = (lidar_h[:, 1] > -1.5) & (lidar_h[:, 1] < 1.5)

        front_mask = depth_mask & side_mask
        cam_lidar_h = (self.icp_result @ lidar_h.T).T
        #cam_lidar_h[:, 2] = 0.05 # set all points to be at a constant height of 0.05m, as we are only interested in the top-down view of the LiDAR points, and the height information is not crucial for this task. This also helps to reduce the noise and outliers in the depth direction, as the LiDAR points can be very sparse and noisy in the vertical direction.
        
        # set all points to be at a constant height of 0.05m 
        #cam_lidar_h[:, 2] = 0.05 
        
        # change from LiDAR-camera frame (x forward, y left, z up) to image frame (x right, y down, z forward)
        R_lidar_to_cam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        cam_pts = (R_lidar_to_cam @ cam_lidar_h.T).T
        
        cam_pts = cam_pts[front_mask]
        lidar_h = lidar_h[front_mask]

        img_pts, _ = cv2.projectPoints(cam_pts, np.zeros(3), np.zeros(3), self.new_camera_k, self.new_camera_dist)
        
        img_pts = img_pts.reshape(-1, 2)

        valid_mask = (
            (img_pts[:, 0] >= 0) & (img_pts[:, 0] < w) &
            (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h)
        )

        img_pts = img_pts[valid_mask]
        cam_pts = cam_pts[valid_mask]
        lidar_h = lidar_h[valid_mask]
        
        return img_pts, cam_pts, lidar_h

    def lidar_projection_pipeline(self, lidar):

        # Project LiDAR to camera frame
        img_pts, cam_pts, lidar_pts = self.lidar_to_image_projection(
            lidar,
            self.img.shape
        )

        return img_pts, cam_pts, lidar_pts

    def draw_lidar_points(self):
        for p in self.img_pts:
            u, v = p.ravel().astype(int)
            cv2.circle(self.img, (u, v), 1, (0, 0, 255), -1)
        return self.img
    
    def extract_depth_in_box(self, boxes, min_points=4):
        
        filtered_points = []

        for box in boxes:
            # if boxes contain colour: (box, colour)
            if isinstance(box, tuple):
                box = box[0]

            selected_points = []

            for (p, P3D) in zip(self.img_pts, self.lidar_pts):
                u, v = map(float, p.ravel())

                if cv2.pointPolygonTest(box, (u, v), False) >= 0:
                    selected_points.append(P3D)

            if len(selected_points) >= min_points:
                filtered_points.extend(selected_points)

        if len(filtered_points) == 0:
            return np.empty((0, self.lidar_pts.shape[1]))

        return np.array(filtered_points)