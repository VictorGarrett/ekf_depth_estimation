# EKF Depth Estimation

The objective of this package is to estimate the distance between a robot and an object of unknown size.

This code whas developed as part of a Globaling Research Internship project under the mentorship of professor 
Krishna Vijayaraghavan at Simon Fraser University.

## Dependencies
This system is build for ROS and uses the Gazebo simulator.
Specifically, it was built with ROS Noetic and Gazebo 11 (Gazebo Classic)
The ROS tutorials cover installation of the system, that comes with Gazebo already. https://wiki.ros.org/ROS/Tutorials

Also, to simulate the motors, this plugin was used: https://github.com/VictorGarrett/gazebo_ros_motors_euler

Tmux is also necessary to run the startup scripts.

## Run Instructions

1. Setup a ROS workspace. (http://wiki.ros.org/catkin/Tutorials/create_a_workspace)
2. Clone the motor plugin into src (https://github.com/VictorGarrett/gazebo_ros_motors_euler).
3. Clone this repository into src.
4. Build the workspace (`catkin build` if using ROS Noetic).
5. Run the start_slip.sh script inside the tmux folder.


Relevant resources:
https://wiki.ros.org/Packages
http://wiki.ros.org/roslaunch
https://wiki.ros.org/rosbash#rosrun


## Project Structure
### Scripts
The executable code for the EKF estimation

### Launch
Launchfiles to run the ROS nodes. Sets up the initialization variables for Gazebo, such as world and GUI usage. Also spawns the robot model.

### Media and Models
Contains the used models and textures. The robot model can be modified there.

### Tmux
Launch script to run everything using tmux. Highly recommended.

