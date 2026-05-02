"""
MTRX4701 2026 Assignment 3: Simultaneous Localisation and Mapping
File: aruco_perception.py
Author(s): 530 499 451

ArUco-tag-based landmark perception module.

Drop-in replacement for LandMarkPerception. Instead of using HSV colour
segmentation to find cones, this module detects ArUco tags on the cone bodies,
takes the lidar points whose projected pixel falls within each tag's lateral
(horizontal) image bounds, fits a circle to those points and emits a
LandmarkMeasurement.

Data association is done directly via ArUco tag IDs:
 - if a tag's ID matches the aruco_id of an existing StoredLandmark, the
   measurement reuses that landmark's lm_id.
 - otherwise the tag is treated as a brand new landmark; a fresh lm_id is
   minted and the aruco_id is stamped onto the measurement so the EKF can
   record it on the new StoredLandmark.

The colour pipeline's Mahalanobis gating is intentionally skipped here -
tag IDs resolve identity directly. The EKF's own innovation ceiling still
catches genuinely broken measurements downstream.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as pltCircle

from cv_bridge import CvBridge

import turtlebot_landmark_slam.lidar_project_to_image as lpi
from turtlebot_landmark_slam.landmarks_circle_detector import extract_circular_objects
from turtlebot_landmark_slam.types import LandmarkMeasurement, StoredLandmark
from turtlebot_landmark_slam.utils import Relative2AbsoluteXY


class ArucoPerception(object):
    def __init__(self,
                 img,
                 lidar,
                 ekf_pose,
                 ekf_pose_covariance,
                 ekf_landmarks,
                 image_publisher,
                 debugging=False,
                 aruco_dict=cv2.aruco.DICT_4X4_50,
                 min_points_per_tag=4,
                 cylinder_growth_gap=0.05):
        """
        Args:
            img:                 BGR image from /camera/image_raw.
            lidar:               Nx2 relative lidar points (robot frame).
            ekf_pose:            3x1 [x, y, theta] in absolute frame.
            ekf_pose_covariance: 3x3 covariance of the pose.
            ekf_landmarks:       list[StoredLandmark] currently tracked by EKF.
            image_publisher:     ROS2 sensor_msgs/Image publisher for the
                                 annotated debug frame.
            debugging:           if True, also pop up a non-blocking
                                 matplotlib top-down plot of the lidar points
                                 with per-tag clusters highlighted.
            aruco_dict:          cv2.aruco predefined dictionary id.
                                 Default DICT_4X4_50.
            min_points_per_tag:  minimum number of in-band lidar points
                                 required to attempt a circle fit.
            cylinder_growth_gap: max radial gap (m) between consecutive
                                 lidar returns when growing a tag's seed
                                 cluster outward to capture the full
                                 cylinder surface. Same scale as
                                 LidarLandmarkObserver's distance_threshold.
        """
        self.img_pub = image_publisher
        self.debugging = debugging
        self.min_points_per_tag = min_points_per_tag
        self.cylinder_growth_gap = cylinder_growth_gap

        # Same 2 cm std-dev measurement noise floor as LandMarkPerception.
        self.MIN_MEAS_VAR = 0.02 ** 2

        # ArUco detector setup. The OpenCV API changed names between 4.6 and
        # 4.7+; support both so this works on whatever turtlebot ship has.
        self._aruco_dict_id = aruco_dict
        self._aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        if hasattr(cv2.aruco, "ArucoDetector"):
            params = cv2.aruco.DetectorParameters()
            self._aruco_detector = cv2.aruco.ArucoDetector(
                self._aruco_dictionary, params
            )
            self._use_new_api = True
        else:
            self._aruco_params = cv2.aruco.DetectorParameters_create()
            self._aruco_detector = None
            self._use_new_api = False

 

        # Run the pipeline.
        self.landmark_measurements = self._perform_perception(
            img, lidar, ekf_pose, ekf_pose_covariance, ekf_landmarks
        )

    # ------------------------------------------------------------------
    # Public convenience wrapper - matches LidarLandmarkObserver's call style.
    # ------------------------------------------------------------------

    @classmethod
    def measure_landmarks(cls,
                          img,
                          lidar,
                          ekf_pose,
                          ekf_pose_covariance,
                          ekf_landmarks,
                          image_publisher,
                          debugging=False,
                          aruco_dict=cv2.aruco.DICT_4X4_50,
                          min_points_per_tag=4,
                          cylinder_growth_gap=0.05):
        """Convenience wrapper. Returns the list of LandmarkMeasurements."""
        instance = cls(
            img=img,
            lidar=lidar,
            ekf_pose=ekf_pose,
            ekf_pose_covariance=ekf_pose_covariance,
            ekf_landmarks=ekf_landmarks,
            image_publisher=image_publisher,
            debugging=debugging,
            aruco_dict=aruco_dict,
            min_points_per_tag=min_points_per_tag,
            cylinder_growth_gap=cylinder_growth_gap,
        )
        return instance.landmark_measurement

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _perform_perception(self, img, lidar, ekf_pose,
                            ekf_pose_covariance, ekf_landmarks):
        # ----- 1. Project lidar points into the (undistorted) image -----
        # Reuse LidarProject so the ICP transform + intrinsics live in one
        # place. lidar_project.img is already undistorted.
        lidar_project = lpi.LidarProject(img, lidar)
        undistorted_img = lidar_project.img
        img_pts = lidar_project.img_pts        # (M, 2) pixel (u, v)
        lidar_pts = lidar_project.lidar_pts    # (M, 3) lidar frame

        # ----- 2. Detect ArUco tags on the undistorted frame -----
        gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
        if self._use_new_api:
            corners_list, ids, _ = self._aruco_detector.detectMarkers(gray)
        else:
            corners_list, ids, _ = cv2.aruco.detectMarkers(
                gray, self._aruco_dictionary, parameters=self._aruco_params
            )

        if ids is None:
            print("ArUco -> no tags detected in frame")
            self._publish_debug_image(
                undistorted_img, img_pts, [], [], []
            )
            if self.debugging:
                self._update_topdown_plot(lidar_pts, [])
            return []

        ids = ids.flatten().tolist()
        print(f"ArUco -> detected {len(ids)} tag(s): {ids}")

        # Pre-sort the projected lidar points by bearing in the lidar frame.
        # The sweep that grows each tag's seed cluster outward walks this
        # ordering, so consecutive entries are angular neighbours.
        if lidar_pts.size:
            lidar_xy_full = lidar_pts[:, :2]
            bearings_full = np.arctan2(lidar_xy_full[:, 1], lidar_xy_full[:, 0])
            sort_order = np.argsort(bearings_full)
            sorted_xy = lidar_xy_full[sort_order]
            sorted_img_pts = img_pts[sort_order]
        else:
            sorted_xy = np.empty((0, 2))
            sorted_img_pts = np.empty((0, 2))

        # ----- 3. Per-tag association of lidar points + circle fit -----
        # Each entry: dict(aruco_id, corners, u_min, u_max, seed_xy,
        #                  cylinder_xy, fit_or_None)
        tag_records = []

        for tag_corners, tag_id in zip(corners_list, ids):
            # tag_corners shape: (1, 4, 2) -> 4 (u, v) corner pixels
            corners = tag_corners.reshape(-1, 2)
            u_min = float(corners[:, 0].min())
            u_max = float(corners[:, 0].max())

            # Seed indices: projected pixels whose u falls in the tag's
            # lateral bounds. These anchor us to the right physical cone.
            if sorted_img_pts.shape[0] == 0:
                seed_mask = np.zeros((0,), dtype=bool)
            else:
                u_coords = sorted_img_pts[:, 0]
                seed_mask = (u_coords >= u_min) & (u_coords <= u_max)

            seed_xy = sorted_xy[seed_mask]

            # Grow each seed outward (in bearing order) along the cylinder
            # until the radial gap to the next neighbour exceeds the
            # threshold. Captures the full cone surface, not just the
            # narrow strip behind the tag.
            cylinder_xy = self._grow_cylinder_cluster(
                sorted_xy, seed_mask, self.cylinder_growth_gap
            )

            record = {
                "aruco_id": int(tag_id),
                "corners": corners,
                "u_min": u_min,
                "u_max": u_max,
                "seed_xy": seed_xy,
                "cylinder_xy": cylinder_xy,
                # Kept under the old key so the debug image highlight code
                # below still works without conditional branches.
                "band_lidar_xy": cylinder_xy,
                "fit": None,
            }

            if cylinder_xy.shape[0] < self.min_points_per_tag:
                print(f"\tTag {tag_id}: only {cylinder_xy.shape[0]} "
                      f"cylinder pts after growth (seeds={seed_xy.shape[0]}, "
                      f"need {self.min_points_per_tag}); skipping.")
                tag_records.append(record)
                continue

            fits = extract_circular_objects(cylinder_xy,
                                            distance_threshold=0.1, # 0.05
                                            min_points=4,
                                            max_radius=0.09,                 # 0.12          -- higest reading was 0.18
                                            min_radius=0.05,                 # 0.06          -- lowest reading was 0.11
                                            max_mse=1.0e-4,                      # 1.0e-4        -- annoying corner case
                                            max_aspect_ratio=None,          # None
                                            min_arc_angle=np.radians(30),   # np.radians(90)-- cleared out wall false positives
                                            min_center_range=None,
                                            polar=False)
            if not fits:
                print(f"\tTag {tag_id}: no circular fit on "
                      f"{cylinder_xy.shape[0]} pts (seeds={seed_xy.shape[0]});"
                      " skipping.")
                tag_records.append(record)
                continue

            # Pick the fit with the lowest MSE - most likely to be the cone.
            best_fit = min(fits, key=lambda c: c.mse)
            record["fit"] = best_fit
            tag_records.append(record)
            print(f"\tTag {tag_id}: fit cylinder w/ "
                  f"{cylinder_xy.shape[0]} pts (seeds={seed_xy.shape[0]}), "
                  f"r={best_fit.radius:.3f} m, mse={best_fit.mse:.2e}")

        # ----- 4. Build LandmarkMeasurements with ArUco-ID-based association
        landmark_measurements = self._associate_and_build_measurements(
            tag_records, ekf_pose, ekf_pose_covariance, ekf_landmarks
        )

        # ----- 5. Always publish the annotated debug image -----
        self._publish_debug_image(
            undistorted_img, img_pts, tag_records, landmark_measurements, lidar_pts
        )

        # ----- 6. Optional matplotlib top-down plot -----
        if self.debugging:
            self._update_topdown_plot(lidar_pts, tag_records)

        return landmark_measurements

    def _associate_and_build_measurements(self, tag_records, ekf_pose,
                                          ekf_pose_covariance, ekf_landmarks):
        """Map each tag with a valid fit to a LandmarkMeasurement.

        Direct association by aruco_id - no Mahalanobis gating.
        """
        measurements = []

        # Build a lookup from aruco_id -> existing StoredLandmark (only those
        # that have been previously seen via this observer).
        existing_by_aruco = {
            l.aruco_id: l for l in ekf_landmarks if l.aruco_id != -1
        }

        # lm_id namespace: keep the same convention as LandMarkPerception
        # (start from len(ekf_landmarks), increment as we mint new ones).
        next_new_id = len(ekf_landmarks)

        for rec in tag_records:
            fit = rec["fit"]
            if fit is None:
                continue

            tag_id = rec["aruco_id"]

            # Relative measurement = circle centre in robot frame.
            rel_x = float(fit.center[0])
            rel_y = float(fit.center[1])

            # 2x2 covariance with sensor noise floor (matches LandMarkPerception)
            xy_cov = fit.covariance[0:2, 0:2] + self.MIN_MEAS_VAR * np.eye(2)

            # Absolute position - useful for the debug print.
            abs_xy, _, _ = Relative2AbsoluteXY(ekf_pose, [rel_x, rel_y])

            if tag_id in existing_by_aruco:
                stored = existing_by_aruco[tag_id]
                lm_id = stored.lm_id
                print(f"\tArUco -> tag {tag_id} matched stored landmark "
                      f"lm_id={lm_id} @ ABS({stored.abs_x:.2f},"
                      f"{stored.abs_y:.2f}); meas ABS({float(abs_xy[0]):.2f},"
                      f"{float(abs_xy[1]):.2f})")
            else:
                lm_id = next_new_id
                next_new_id += 1
                print(f"\tArUco -> tag {tag_id} is new; minted lm_id={lm_id} "
                      f"@ ABS({float(abs_xy[0]):.2f},{float(abs_xy[1]):.2f})")

            measurement = LandmarkMeasurement(
                x=rel_x,
                y=rel_y,
                covariance=xy_cov,
                lm_id=lm_id,
                aruco_id=tag_id,
            )
            measurements.append(measurement)

        return measurements

    # ------------------------------------------------------------------
    # Cylinder cluster growth
    # ------------------------------------------------------------------

    @staticmethod
    def _grow_cylinder_cluster(sorted_xy, seed_mask, max_gap):
        """Grow each seed outward along the bearing-sorted scan until the
        Euclidean gap to the next neighbour exceeds `max_gap`.

        The seed indices anchor us to the cone associated with the tag; the
        outward sweep then captures the rest of the cylinder surface (which
        extends past the tag's narrow lateral strip).

        Args:
            sorted_xy:  (N, 2) lidar points pre-sorted by bearing.
            seed_mask:  (N,) bool mask flagging which points fall inside the
                        tag's lateral u-band.
            max_gap:    radial distance threshold between consecutive
                        accepted points.

        Returns:
            (M, 2) array of cluster points, M >= seeds.shape[0].
        """
        n = sorted_xy.shape[0]
        if n == 0 or not seed_mask.any():
            return np.empty((0, 2))

        accepted = seed_mask.copy()
        seed_indices = np.where(seed_mask)[0]

        # Sweep right from each seed.
        last = seed_indices[-1]
        i = last
        while i + 1 < n:
            gap = np.linalg.norm(sorted_xy[i + 1] - sorted_xy[i])
            if gap > max_gap:
                break
            accepted[i + 1] = True
            i += 1

        # Sweep left from each seed.
        first = seed_indices[0]
        i = first
        while i - 1 >= 0:
            gap = np.linalg.norm(sorted_xy[i - 1] - sorted_xy[i])
            if gap > max_gap:
                break
            accepted[i - 1] = True
            i -= 1

        return sorted_xy[accepted]



    def _publish_debug_image(self, undistorted_img, img_pts, tag_records,
                             measurements, lidar_pts):
        """Annotate the frame and publish over the supplied image_publisher.

        Always runs (not gated on `debugging`) - matches LandMarkPerception's
        behaviour of always feeding the debug topic.
        """
        debug_img = undistorted_img.copy()

        # All projected lidar points - faint red.
        for p in img_pts:
            u, v = p.ravel().astype(int)
            cv2.circle(debug_img, (u, v), 1, (0, 0, 180), -1)

        # Per-tag overlays.
        per_tag_colours = [
            (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255),
            (255, 255, 0), (255, 128, 0), (128, 0, 255), (0, 128, 255),
        ]

        # Reproject helper - fit centre is in the lidar frame, so we send it
        # back through the same pipeline as the raw scan to draw it.
        for i, rec in enumerate(tag_records):
            colour = per_tag_colours[i % len(per_tag_colours)]
            corners = rec["corners"].astype(int)

            # Tag polygon.
            cv2.polylines(debug_img, [corners], isClosed=True,
                          color=colour, thickness=2)

            # Vertical lateral bounds.
            u_min_i = int(rec["u_min"])
            u_max_i = int(rec["u_max"])
            h = debug_img.shape[0]
            cv2.line(debug_img, (u_min_i, 0), (u_min_i, h - 1), colour, 1)
            cv2.line(debug_img, (u_max_i, 0), (u_max_i, h - 1), colour, 1)

            # ID label.
            label_pos = tuple(corners[0])
            fit = rec["fit"]
            status = "OK" if fit is not None else "NO FIT"
            cv2.putText(debug_img,
                        f"id={rec['aruco_id']} ({status})",
                        (label_pos[0], max(label_pos[1] - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

            # In-band / grown lidar points overlay.
            #   - Seed pts (inside the tag's u-band) drawn solid.
            #   - Grown pts (outside the band but stitched to the cylinder
            #     via the bearing sweep) drawn as outlines so you can see
            #     how far the seed was extended.
            if img_pts.shape[0] > 0 and lidar_pts.size:
                u_coords = img_pts[:, 0]
                seed_mask_dbg = ((u_coords >= rec["u_min"]) &
                                 (u_coords <= rec["u_max"]))

                # Build "is in cylinder cluster" mask over img_pts by
                # checking each point's lidar (x, y) against cluster set.
                cyl = rec["cylinder_xy"]
                if cyl.shape[0] > 0:
                    cyl_set = {(float(p[0]), float(p[1])) for p in cyl}
                    cyl_mask_dbg = np.array([
                        (float(p[0]), float(p[1])) in cyl_set
                        for p in lidar_pts[:, :2]
                    ])
                else:
                    cyl_mask_dbg = np.zeros(img_pts.shape[0], dtype=bool)

                grown_only_mask = cyl_mask_dbg & ~seed_mask_dbg

                # Solid for seeds.
                for p in img_pts[seed_mask_dbg]:
                    u, v = p.ravel().astype(int)
                    cv2.circle(debug_img, (u, v), 3, colour, -1)
                # Outline for grown-but-not-seed points.
                for p in img_pts[grown_only_mask]:
                    u, v = p.ravel().astype(int)
                    cv2.circle(debug_img, (u, v), 3, colour, 1)

        try:
            img_msg = CvBridge().cv2_to_imgmsg(debug_img, encoding="bgr8")
            self.img_pub.publish(img_msg)
        except Exception as e:
            print(f"ArUco -> debug image publish failed: {e}")

    def _update_topdown_plot(self, lidar_pts, tag_records):
        """Non-blocking matplotlib top-down view of the lidar scan with
        per-tag clusters highlighted and fitted circles drawn.
        """
        cls = type(self)
        if getattr(cls, "_debug_fig", None) is None or \
           not plt.fignum_exists(cls._debug_fig.number):
            cls._debug_fig, cls._debug_ax = plt.subplots(
                figsize=(8, 8), num="ArucoPerception"
            )

        ax = cls._debug_ax

        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("X (m)")
        ax.invert_xaxis()  # robot Y -> left
        ax.set_title("ArucoPerception - Top Down (lidar frame)")
        ax.grid(True, linestyle=":", alpha=0.6)

        # All raw points in grey.
        if lidar_pts.size:
            ax.plot(lidar_pts[:, 1], lidar_pts[:, 0],
                    ".", color="lightgray", markersize=3, zorder=1,
                    label="raw scan")

        # Sensor origin.
        ax.plot(0, 0, "^", color="black", markersize=10, zorder=5,
                label="robot")

        cmap = plt.cm.tab10.colors
        for i, rec in enumerate(tag_records):
            colour = cmap[i % len(cmap)]
            seeds = rec["seed_xy"]
            cyl = rec["cylinder_xy"]
            if cyl.shape[0] == 0:
                continue

            # Grown-only = cylinder pts that aren't seeds. Set-difference
            # by row.
            if seeds.shape[0]:
                seed_set = {(float(p[0]), float(p[1])) for p in seeds}
                grown_only = np.array([
                    p for p in cyl
                    if (float(p[0]), float(p[1])) not in seed_set
                ])
            else:
                grown_only = cyl

            # Faded outline for grown-only.
            if grown_only.size:
                ax.plot(grown_only[:, 1], grown_only[:, 0],
                        "o", markerfacecolor="none", markeredgecolor=colour,
                        markersize=6, zorder=2,
                        label=f"tag {rec['aruco_id']} grown "
                              f"(n={grown_only.shape[0]})")
            # Solid for seeds.
            if seeds.size:
                ax.plot(seeds[:, 1], seeds[:, 0], ".", color=colour,
                        markersize=8, zorder=3,
                        label=f"tag {rec['aruco_id']} seed "
                              f"(n={seeds.shape[0]})")

            fit = rec["fit"]
            if fit is not None:
                cx, cy = fit.center[0], fit.center[1]
                ax.add_patch(pltCircle((cy, cx), fit.radius, color=colour,
                                       fill=False, linewidth=2, zorder=4))
                ax.plot(cy, cx, "+", color=colour, markersize=12, zorder=5)
                ax.annotate(f"id={rec['aruco_id']}", (cy, cx),
                            textcoords="offset points", xytext=(8, 8),
                            color=colour, fontweight="bold")

        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)