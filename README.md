# EKF Depth Estimation

The objective of this package is to estimate the distance between a robot and an object of unknown size.

## Dependencies
This system is build for ROS and uses the Gazebo simulator.
The ROS tutorials cover installation of the system, that comes with Gazebo already. https://wiki.ros.org/ROS/Tutorials

Also, to simulate the motors, this plugin was used: https://github.com/nilseuropa/gazebo_ros_motors
So installing that is necessary,

Installing Tmux is also recommended.

## Run Instructions

This is a ROS package and must be installed as such (https://wiki.ros.org/Packages).


After that, this package must be cloned inside a ROS workspace src folder, then the workspace must be built (`catkin build` if using ROS Noetic).
After the package is built, if the script is not being found by rosrun, it may be necessary to run `chmod +x https://wiki.ros.org/Packages` inside the package folder.
With that working it is possible to run the script inside tmux folter (start_slip.sh), it will start both the simulation and the estimation. Alternatively, everithing can be lauched using rosrun and roslaunch manually.
Here is the relevant documentation about that:
http://wiki.ros.org/roslaunch
https://wiki.ros.org/rosbash#rosrun


## Project Structure
### Scripts
The executable code for the EKF estimation

### Media and Models
Contains the used models and textures. The robot model can be modified there.

### Tmux
Launch script to run everything using tmux. Highly recommended.

