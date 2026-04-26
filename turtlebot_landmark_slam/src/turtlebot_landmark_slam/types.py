from dataclasses import dataclass
import numpy as np
from landmarks_msg.msg import LandmarkMsg

@dataclass
class LandmarkMeasurement:
    x: float
    y: float
    label: int
    covariance: np.array # [2x2]
    is_new: bool

    def __str__(self):
        return f"x: {self.x} y: {self.y} label: {self.label} cov: {self.covariance}"

    # @classmethod
    # def from_landmark_msg(cls, msg: LandmarkMsg):
    #     covariance = np.array([[msg.s_x, 0.0],
    #                                 [0.0, msg.s_y]])
    #     # Deal with measurement covariance close to zero
    #     if msg.s_x < 10**(-4) and msg.s_y < 10**(-4):
    #         print("Measurement covariance is close to zero.")
    #         covariance = np.array([[0.01,0.0], [0.0, 0.01]])   # 10 cm ??  
    #     measurement = cls(msg.x, msg.y, msg.label, covariance)
    #     return measurement

@dataclass
class ControlMeasurement:
    dx: float
    dy: float
    dtheta: float
    covariance: np.array # [3x3]

    @property
    def motion_vector(self):
        return np.array([[self.dx], [self.dy], [self.dtheta]], copy=True)

    def __str__(self):
        return f"sx: {self.dx} dy: {self.dy} dtheta: {self.dtheta} cov: {self.covariance}"
    
@dataclass
class StoredLandmark:
    abs_x: float    # x component of abs. landmark coords.
    abs_y: float    # y component of abs. landmark coords.
    index: int      # index in state variable
    id: int      # name of landmark
    covariance: np.array # [2x2]

    colour='r'      # plot colour
    radius=0.1      # plot patch radius

    @property
    def mean(self):
        return (self.abs_x, self.abs_y)
    
    def __str__(self):
        return f"label: {self.id}, index: {self.index}, coords: ({self.abs_x},{self.abs_y})"