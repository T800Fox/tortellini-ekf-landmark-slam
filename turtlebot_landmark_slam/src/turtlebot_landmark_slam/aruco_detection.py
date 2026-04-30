import cv2
import numpy as np


class ArucoDetection():
    def __init__(self, img):
        self.img = self.aruco_detect(img)
        pass

    def aruco_detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters_create()

        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=parameters
        )

        print("Detected markers:", ids)

        # draw custom pink bounding boxes
        if ids is not None:
            for c in corners:
                pts = c.reshape((4, 2)).astype(int)

                # draw 4 edges
                for i in range(4):
                    pt1 = tuple(pts[i])
                    pt2 = tuple(pts[(i + 1) % 4])

                    cv2.line(img, pt1, pt2, (255, 0, 255), 3)  # pink (BGR)

                # optional: center dot
                center = pts.mean(axis=0).astype(int)
                cv2.circle(img, tuple(center), 5, (255, 0, 255), -1)

        return img