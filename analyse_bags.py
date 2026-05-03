from pathlib import Path
import yaml
import math
import numpy as np
import matplotlib.pyplot as plt
import rosbag2_py
import csv
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan


def get_storage_id(bag_path):
    metadata_path = bag_path / "metadata.yaml"

    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    return metadata["rosbag2_bagfile_information"]["storage_identifier"]



bags = ["0p5F", "0p5R", "0p5L", "1p0F", "1p0R", "1p0L", "1p5F", "1p5R", "1p5L"]

view_labels = {
    "L": "Facing left (-π/2 rad)",
    "F": "Facing forward (0 rad)",
    "R": "Facing right (+π/2 rad)",
}

view_order = ["L", "F", "R"]

distance_order = [0.5, 1.0, 1.5]
results= []

SCRIPT_DIR = Path(__file__).resolve().parent
TOPIC_NAME = "/scan"

#Cycle through each bag
for bag in range(len(bags)):
    BAG_NAME = bags[bag]
    BAG_PATH = SCRIPT_DIR / BAG_NAME
    match BAG_NAME[0]:
        case "1":
            exp_range = 1
        case "0":
            exp_range = 0
    match BAG_NAME[2]:
        case "5":
            exp_range += 0.5
    match BAG_NAME[3]:
        case "F":
            exp_angle = 0
        case "L":
            exp_angle = math.radians(270)
        case "R":
            exp_angle = math.radians(90)

    MIN_RANGE = exp_range - 0.07
    MAX_RANGE =exp_range +0.07
    MAX_ANGLE = exp_angle + 0.5
    MIN_ANGLE = exp_angle - 0.5 

    storage_id = get_storage_id(BAG_PATH)

    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=str(BAG_PATH),
        storage_id=storage_id,
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(storage_options, converter_options)

    scan_count = 0

    distances = []
    angles = []
    #take each lidar reading in each bag
    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if topic == TOPIC_NAME:
            scan_msg = deserialize_message(data, LaserScan)

            scan_count += 1
            distance_filtured = []
            angle_filtured = []

            #Cycle through each measurement in lidar reading
            for i in range(len(scan_msg.ranges)):
                if math.isnan(scan_msg.ranges[i]):
                    continue
                
                r = scan_msg.ranges[i]
                theta = scan_msg.angle_min + i * scan_msg.angle_increment
                
                #wrap around logic
                if theta > 5.4:
                    theta = theta - (360 *math.pi/180)
                
                #Cone logic
                if MIN_RANGE < r < MAX_RANGE and (MIN_ANGLE < theta  < MAX_ANGLE): 
                    
                    distance_filtured.append(r)
                    angle_filtured.append(theta)

            distances.append(np.mean(distance_filtured))
            angles.append(np.mean(angle_filtured))
    total_dist_mean = np.mean(distances)
    total_angle_mean = np.mean(angles)

    distance_sd = np.std(distances, ddof=1)
    angle_sd = np.std(angles, ddof=1)

    cov_matrix = np.cov(np.array([distances, angles]), ddof=1)

    range_variance = cov_matrix[0, 0]
    range_angle_covariance = cov_matrix[0, 1]
    angle_variance = cov_matrix[1, 1]

    results.append({
        "bag_name": BAG_NAME,
        "expected_range_m": exp_range,
        "view": BAG_NAME[3],
        "view_label": view_labels[BAG_NAME[3]],
        "scan_count": scan_count,

        "mean_range_m": total_dist_mean,
        "mean_angle_rad": total_angle_mean,

        "range_sd_m": distance_sd,
        "angle_sd_rad": angle_sd,

        "range_variance": range_variance,
        "range_angle_covariance": range_angle_covariance,
        "angle_variance": angle_variance,
    })

#Save to csv
csv_path = SCRIPT_DIR / "landmark_uncertainty_results.csv"

with open(csv_path, "w", newline="") as csvfile:
    fieldnames = [
        "bag_name",
        "expected_range_m",
        "view",
        "view_label",
        "scan_count",
        "mean_range_m",
        "mean_angle_rad",
        "range_sd_m",
        "angle_sd_rad",
        "range_variance",
        "range_angle_covariance",
        "angle_variance",
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in results:
        writer.writerow(row)

x_positions = np.arange(len(distance_order))
bar_width = 0.25

#Range uncertainty ploy
fig_range, ax_range = plt.subplots(figsize=(10, 6))

for i, distance in enumerate(distance_order):
    y_values = []

    for view in view_order:
        matching_rows = [
            row for row in results
            if row["view"] == view and math.isclose(row["expected_range_m"], distance)
        ]

        if len(matching_rows) == 0:
            y_values.append(np.nan)
        else:
            y_values.append(matching_rows[0]["range_sd_m"])

    offset = (i - 1) * bar_width

    ax_range.bar(
        x_positions + offset,
        y_values,
        width=bar_width,
        label=f"{distance:.1f} m"
    )

ax_range.set_xticks(x_positions)
ax_range.set_xticklabels([view_labels[v] for v in view_order])

ax_range.set_xlabel("LiDAR viewing direction")
ax_range.set_ylabel("Range standard deviation [m]")
ax_range.set_title("Range uncertainty by viewing direction and distance")
ax_range.grid(True, axis="y")
ax_range.legend(title="Landmark distance")

fig_range.tight_layout()
fig_range.savefig(SCRIPT_DIR / "range_sd_bar_plot.png", dpi=300)


# Bearing uncertainty Plot
fig_bearing, ax_bearing = plt.subplots(figsize=(10, 6))

for i, distance in enumerate(distance_order):
    y_values = []

    for view in view_order:
        matching_rows = [
            row for row in results
            if row["view"] == view and math.isclose(row["expected_range_m"], distance)
        ]

        if len(matching_rows) == 0:
            y_values.append(np.nan)
        else:
            y_values.append(matching_rows[0]["angle_sd_rad"])

    offset = (i - 1) * bar_width

    ax_bearing.bar(
        x_positions + offset,
        y_values,
        width=bar_width,
        label=f"{distance:.1f} m"
    )

ax_bearing.set_xticks(x_positions)
ax_bearing.set_xticklabels([view_labels[v] for v in view_order])

ax_bearing.set_xlabel("LiDAR viewing direction")
ax_bearing.set_ylabel("Bearing standard deviation [rad]")
ax_bearing.set_title("Bearing uncertainty by viewing direction and distance")
ax_bearing.grid(True, axis="y")
ax_bearing.legend(title="Landmark distance")

fig_bearing.tight_layout()
fig_bearing.savefig(SCRIPT_DIR / "bearing_sd_bar_plot.png", dpi=300)

plt.show()