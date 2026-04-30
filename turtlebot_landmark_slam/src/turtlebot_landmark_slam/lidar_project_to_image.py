import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import json
from pathlib import Path
from sklearn.cluster import DBSCAN

def plot_top_down_view(cam_lidar_h):
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

def lidar_to_image_projection(lidar, icp_result, camera_k, camera_dist, img_shape):
    h, w = img_shape[:2]
    
    pts = np.asarray(lidar.points) 
    
    lidar_pts = pts[:, :2]
    
    lidar_h = np.hstack((lidar_pts, np.zeros((lidar_pts.shape[0], 1))))
    depth_mask = (lidar_h[:, 0] > 0) & (lidar_h[:, 0] < 2.5)
    side_mask  = (lidar_h[:, 1] > -1.5) & (lidar_h[:, 1] < 1.5)

    front_mask = depth_mask & side_mask
    cam_lidar_h = (icp_result @ lidar_h.T).T
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

    img_pts, _ = cv2.projectPoints(cam_pts, np.zeros(3), np.zeros(3), camera_k, camera_dist)
    
    img_pts = img_pts.reshape(-1, 2)

    valid_mask = (
        (img_pts[:, 0] >= 0) & (img_pts[:, 0] < w) &
        (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h)
    )

    img_pts = img_pts[valid_mask]
    cam_pts = cam_pts[valid_mask]
    lidar_h = lidar_h[valid_mask]
    
    return img_pts, cam_pts, lidar_h

def lidar_projection_pipeline(img, lidar, icp_result, camera_k, camera_dist):
    h, w = img.shape[:2]

    # 1. Undistort camera
    new_camera_k, _ = cv2.getOptimalNewCameraMatrix(camera_k, camera_dist, (w, h), 1, (w, h))
    new_camera_dist = np.zeros((1, 5))

    undistorted_img = cv2.undistort(img, camera_k, camera_dist, None, new_camera_k)

    # 2. Project LiDAR to camera frame
    img_pts, cam_pts, lidar_pts = lidar_to_image_projection(
        lidar,
        icp_result,
        new_camera_k,
        new_camera_dist,
        undistorted_img.shape
    )

    return undistorted_img, img_pts, cam_pts, lidar_pts

def extract_depth_in_box(
        boxes,
        img_pts,
        lidar_pts,
        filename=None,
        save_dir=None,
        min_points=4,
        save=False   # <-- NEW
    ):
    # cam_pts follow opencv convention: x right, y down, z forward
    valid_points = []
    result = {
        "image": filename,
        "boxes": [],
        "valid_box_indices": []
    }

    for j, box in enumerate(boxes):
        selected_points = []

        for (p, P3D) in zip(img_pts, lidar_pts):
            u, v = map(float, p.ravel())

            # check if inside contour
            inside = cv2.pointPolygonTest(box, (u, v), False)

            if inside >= 0:  # inside or on edge
                selected_points.append(P3D)

        selected_points = np.array(selected_points)
        
        if len(selected_points) < min_points:
            continue
        
        xy = selected_points[:, [0, 1]]   # (X, Y) in lidar frame, where X is forward and Y is right
        
        # Each boxes might contain more than 1 cones
        # We can use clustering to separate them, and then analyze the depth distribution of each cluster to estimate the depth of each cone.
        clustering = DBSCAN(eps=0.05, min_samples=4).fit(xy)

        labels = clustering.labels_
        
        unique_labels = set(labels)

        box_data = {
            "box_id": j,
            "box_coords": box.tolist(),
            "clusters": []
        }
        
        #plt.figure()
        # This one check for each cluster in the box
        for label in unique_labels:
            if label == -1:
                continue  # skip noise

            cluster_pts = selected_points[labels == label]
            
            depths = cluster_pts[:, 0]
            mean_depth = np.mean(depths) 
            std_depth = np.std(depths)

            cluster_info = {
                "cluster_id": int(label),
                "num_points": int(len(cluster_pts)),
                "mean_depth": float(np.mean(depths)),
                "std_depth": float(np.std(depths)),
                "points": cluster_pts.tolist()   # optional (can remove if too large)
            }

            box_data["clusters"].append(cluster_info)
            #print(f"Cluster {label}:") 
            #print(f" Num points: {len(cluster_pts)}") 
            #print(f" Mean depth: {mean_depth}") 
            #print(f" Std depth: {std_depth}") 
            #pts = xy[labels == label] 
            #plt.scatter(pts[:,0], pts[:,1], label=f"cluster {label}")
        #plt.legend() 
        #plt.axis("equal") 
        #plt.title("LiDAR clustering") 
        #plt.show()
            
        result["boxes"].append(box_data)
        result["valid_box_indices"].append(j)
        
    # --- SAVE FILE ---
    if save:
        if filename is None or save_dir is None:
            raise ValueError("filename and save_dir must be provided if save=True")

        save_path = os.path.join(save_dir, f"{filename}.json")

        with open(save_path, "w") as f:
            json.dump(result, f, indent=4)

        print(f"Saved: {save_path}")
    return result