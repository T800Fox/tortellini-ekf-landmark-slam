from dataclasses import dataclass
import numpy as np

@dataclass
class LandmarkMeasurement:
    x: float
    y: float
    covariance: np.array # [2x2]

    lm_id: int = -1
    colour: str = "unknown"

    def __str__(self):
        return f"x: {self.x} y: {self.y} id: {self.id} cov: {self.covariance}"

    @property
    def mean(self):
        return np.array([[self.x], [self.y]])

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
    covariance: np.array # [2x2]

    innovation_x=0.0
    innovation_y=0.0

    index: int      # index in state variable
    lm_id: int      # name of landmark
    label="NULL"
    

    colour='r'      # plot colour
    radius=0.1      # plot patch radius

    @property
    def mean(self):
        return (self.abs_x, self.abs_y)
    
    def __str__(self):
        return f"label: {self.id}, index: {self.index}, coords: ({self.abs_x},{self.abs_y})"

class se2:
    def __init__(self, x: float = None, y: float = None, theta: float = None, data=None):
        """Initialize SE(2) transform with either:
        - explicit x, y, theta (radians), or
        - a list/array [x, y, theta].
        """
        if data is not None:
            if isinstance(data, np.ndarray):
                data = list(data.flatten())
            if isinstance(data, list) and len(data) == 3:
                self.x, self.y, self.theta = data
                assert isinstance(self.x, float), f"x is of type {type(self.x)}, {self.x}"
                assert isinstance(self.y, float), f"y is of type {type(self.y)}"
                assert isinstance(self.theta, float), f"theta is of type {type(self.theta)}"
            else:
                raise ValueError("data must be a list or array of length 3.")
        elif x is not None and y is not None and theta is not None:
            self.x = x
            self.y = y
            self.theta = theta
        else:
            raise ValueError("Provide either (x, y, theta) or a list/array of size 3.")
        # store cos(theta)
        self._c = np.cos(self.theta)
        # store sin(theta)
        self._s = np.sin(self.theta)

    def matrix(self):
        return np.array([
            [self._c, -self._s, self.x],
            [self._s,  self._c, self.y],
            [0,  0, 1]
        ])

    def compose(self, other):
        """Compose this transform with another SE(2) transform."""
        result_matrix = self.matrix() @ other.matrix()
        x, y = result_matrix[:2, 2]
        theta = np.arctan2(result_matrix[1, 0], result_matrix[0, 0])
        return se2(x, y, theta)

    def inverse(self):
        """Compute the inverse of this transform."""
        inv_x = -self._c * self.x - self._s * self.y
        inv_y =  self._s * self.x - self._c * self.y
        return se2(inv_x, inv_y, -self.theta)

    def __mul__(self, other: "se2") -> "se2":
        """Overload the multiplication operator for composition."""
        return self.compose(other)