from threading import Lock
import numpy as np
from sys import exit
from copy import deepcopy
import pickle

from geometry_msgs.msg import Twist
from std_msgs.msg import UInt8MultiArray
from cv_bridge import CvBridge
import cv2

from turtlebot_landmark_slam.types import ControlMeasurement
from turtlebot_landmark_slam.ekf import ExtendedKalmanFilter
from turtlebot_landmark_slam.landmarks import SimLandmarkObserver, LidarLandmarkObserver
from turtlebot_landmark_slam.aruco_perception import ArucoPerception
from turtlebot_landmark_slam.uncertainty_plotter import UncertaintyPlotter
from turtlebot_landmark_slam.utils import mahalanobis_distance

from turtlebot_landmark_slam.landmark_perception import LandMarkPerception

class EkfOrchestrator(object):
    def __init__(self, node, is_real):
        self._ekf = ExtendedKalmanFilter()
        self.real_env = is_real
        self._lock = Lock()
        self._node = node

        self._bridge = CvBridge()

        self.telemetry_publisher_set = False
        self.visible_landmark_publisher_set = False

        self.ignore_over_dist = 1.5
        self.landmark_cap = -1



        self.last_image_data = None
        self.last_image_nanoseconds = -1

        self.seen_landmark_ids = []

        self.u_plotter = UncertaintyPlotter(
            log_file_path='ekf_values_log.txt'
            )

        self.landmark_cap = 20
        self.lidar_observer = LidarLandmarkObserver(
            show_display=False,
            max_landmark_dist=5,
            max_landmark_count=self.landmark_cap,
            distance_threshold=0.05,    
            min_points=4,
            max_radius=0.09,        
            min_radius=0.005,           
            max_mse=1.0e-4,              
            min_arc_angle=np.radians(90)  
            )


        if not self.real_env:
            print("not configured for gazebo")
            exit()

            
            # self.lidar_observer = SimLandmarkObserver()
            # self.landmark_cap = 4
            # self.lidar_observer = LidarLandmarkObserver(
            #     show_display=True,
            #     max_landmark_dist=5,
            #     max_landmark_count=self.landmark_cap,
            #     distance_threshold=0.05,        # 0.05
            #     min_points=4,
            #     max_radius=0.16,                # 0.2          -- higest reading was 0.18
            #     min_radius=0.14,                # 0.1          -- lowest reading was 0.11
            #     max_mse=1.0e-4,                 # 1.0e-4        -- annoying corner case
            #     min_arc_angle=np.radians(90)   # np.radians(90)-- cleared out wall false positives
            #     )


    def image_handler(self, image_data):
        import cv2


        try:
            self.last_image_data = self._bridge.imgmsg_to_cv2(image_data, desired_encoding="bgr8")
            self.last_image_nanoseconds = self._node.get_clock().now().nanoseconds
        except Exception as e:
            print(f"cv_bridge conversion failed: {e}")

        # cv2.imshow("Tag Detections", self.last_image_data)
        # cv2.waitKey(1) # Refresh window

        # pose = self._ekf.pose
        # pose_covariance = self._ekf.pose_covariance
        # stored_landmarks = self._ekf.tracked_landmarks
        # rel_points = [0,0]
        
        # m_camera = self.aruco_observer.measure_landmarks(ekf_pose=pose,
        #                                                      ekf_covariance=pose_covariance,
        #                                                      image=self.last_image_data,
        #                                                      lidar_points=rel_points)

    def motion_handler(self, control_input: ControlMeasurement):
        with self._lock:
            self._ekf.predict(control_input)

    def lidar_handler(self, rel_points):
        print('lidar!')
        with self._lock:
            
            pose = self._ekf.pose
            pose_covariance = self._ekf.pose_covariance
            stored_landmarks = self._ekf.tracked_landmarks

            if len(stored_landmarks) > self.landmark_cap:
                raise RuntimeError(f"+{self.landmark_cap} Landmarks, Aborting.")

            print("Collecting Landmark Measurements....")
            measurements = []
            # measurements = self.lidar_observer.measure_landmarks()

            print("Checking LiDAR data...")
            m_lidar = self.lidar_observer.measure_landmarks(ekf_pose=pose,
                                                            ekf_pose_covariance=pose_covariance, 
                                                            rel_points=rel_points, 
                                                            ekf_landmarks=stored_landmarks)
            
            stored_ids = [l_s.lm_id for l_s in stored_landmarks]
            l_existing = []
            l_new = []

            for l in m_lidar:
                if l.lm_id in stored_ids:
                    l_existing.append(l)
                else:
                    l_new.append(l)

            print(f"Got {len(l_new)} new landmarks.")
            print(f"Got {len(l_existing)} existing landmarks.")
            
            if len(m_lidar) == 0:
                return
            
            if self.last_image_data is None:
                print("No camera frame yet; skipping.")
                return

            if not self.visible_landmark_publisher_set:
                print("No camera debug publisher set.")
                return

            frame_age = round((self._node.get_clock().now().nanoseconds - self.last_image_nanoseconds)*1e-9, 4)
            print(f"Checking last frame; is {frame_age} seconds old")

            m_camera = ArucoPerception(self.last_image_data,
                                       rel_points,
                                       pose,
                                       pose_covariance,
                                       stored_landmarks,
                                       self.visible_landmark_publisher,
                                       debugging=True,
                                       aruco_dict=cv2.aruco.DICT_4X4_50).landmark_measurements

            print(f"Got {len(m_camera)} from frame...")

            for c_m in m_camera:
                l_closest_dist = 9999999
                for l_m in l_new:
                    dist = mahalanobis_distance(c_m.mean, c_m.covariance,
                                                l_m.mean, l_m.covariance)
                    if dist < l_closest_dist:
                        l_closest = l_m
                        l_closest_dist = dist

                if l_closest_dist < 9.21:
                    # 99.9% ceritanty @ 2 dof --> transfer data from camera to lidar measurement
                    l_m.aruco_id = c_m.aruco_id

                    measurements.append(l_m)
                else:
                    print(f"Could not match camera meas. @ ({c_m.mean}) w/ lidar meas.")


            # currently has new values that got associated through camera
            measurements += l_existing 

            print(f"Feeding {len(measurements)} into ekf update...")

            self._ekf.update(measurements)

            t = self._node.get_clock().now().nanoseconds
            print('t -> ',t)

            telemetry_package = {}
            telemetry_package['time'] = t
            telemetry_package['pose'] = self._ekf.pose
            telemetry_package['pose_covar'] = self._ekf.pose_covariance
            telemetry_package['landmarks'] = self._ekf.tracked_landmarks

            serialized_telemetry = pickle.dumps(telemetry_package)

            if self.telemetry_publisher_set:
                telem_msg = UInt8MultiArray()
                telem_msg.data = list(serialized_telemetry)
                self.telemetry_publisher.publish(telem_msg)

            # self.u_plotter.plot_system(pose=self._ekf.pose,
            #                            pose_covar=self._ekf.pose_covariance,
            #                            landmarks=self._ekf.tracked_landmarks, 
            #                            t=t)

    def handover_telem_publisher(self, publisher):
        self.telemetry_publisher = publisher
        self.telemetry_publisher_set = True

    def handover_visible_landmark_publisher(self, publisher):
        self.visible_landmark_publisher = publisher
        self.visible_landmark_publisher_set = True

    


