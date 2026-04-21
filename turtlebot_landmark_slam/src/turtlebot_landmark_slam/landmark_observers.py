import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as pltCircle
from turtlebot_landmark_slam.landmarks_circle_detector import extract_circular_objects

class lidarLandmarkObserver:
    def __init__(self, liveDisplay):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        assert type(liveDisplay) == bool 
        self.showDisplay= liveDisplay

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

        
        # assoicate detections w/ with tag...



