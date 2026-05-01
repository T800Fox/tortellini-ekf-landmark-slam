# MTRX5700 Assignment 3 : Tortellini - Demo Branch
## Using the Repo From Scratch
### Making Sure Your Terminal is Setup
Go into ```~/.bashrc``` with your editor of choice and make sure that these things have been included towards the end of the file...
- You're sourcing ROS.\
```source /opt/ros/jazzy/setup.bash```
- The ROS domain id is the same as the turtlebot's (ie. 13 for bot #13),\
```export ROS_DOMAIN_ID=13```
- You're using cyclonedds for your topics,\
```export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp```
- The turtlebot model is set to Burger,\
```export TURTLEBOT3_MODEL=burger```

### Running the System off a ROS Bag
1. Get your bag; either a .mcap file or a .mcap and .yaml file
2. Make sure your ros2 workspace has been setup.
    - It's using a folder in the home directory (~/)
    - Within that folder, you've placed this repo and named the file source
    ```git clone <this-repo> src```
    - If the project's already been built source it...\
    ```source install/setup.bash```
    - If not build and source the workspace.\
    ```colcon build --symlink-install && source install/setup.bash```)
    - The workspace folder show now have ```/src```, ```/log```, ```/build``` and  ```/install``` folders within it.
3. If you want the telemetry plotter to run, launch it, though it's not critical.\
```ros2 launch turtlebot_landmark_slam ekf_telemetry.launch.py  ```
4. Launch the ekf_interface node...\
```ros2 launch turtlebot_landmark_slam ekf_interface.launch.py is_real:=true```
5. With the ekf_interface ready to start taking in data, now play the ros bag\
```ros2 bag play <recorded_bag>.mcap``` or  ```ros2 bag play <recorded_bag>/```
6. Sit back and watch the data come through on rviz2 or whatever you may be using.

**Steps 3 to 5 would ideally be done in their own window's on sessions that are sitting in the workspace folder and have sourced install/setup.bash**

