# MTRX5700 Assignment 3 — Landmark-Based EKF SLAM on TurtleBot3


## Running
**Choose between Relative XY or Range Bearing:**
* Adjust the bool USE_RANGE_BEARING at the top of ekf.py to select model and range bearing noise
* adjust linear and angular velocity parameters in ekf_pipeline_launch.py
* adjust landmark noise in simulation_launch.py

**Simulation:**
```bash
# Terminal 1 — start Gazebo
ros2 launch turtlebot_landmark_slam simulation.launch.py

# Terminal 2 — start EKF
ros2 launch turtlebot_landmark_slam ekf_pipeline.launch.py is_real:=false
```


**Move the robot (teleoperation):**
```bash
# Terminal 3 — use keyboard to drive, Ctrl+C to stop
export TURTLEBOT3_MODEL=burger; ros2 run turtlebot3_teleop teleop_keyboard
```

**Save the map:**
```bash
ros2 run turtlebot_landmark_slam map_writer.py
```

**Evaluate against ground truth:**
```bash
python3 scripts/evaluate_map.py --solution map_slam.txt --gt <ground_truth_file>.txt
```
