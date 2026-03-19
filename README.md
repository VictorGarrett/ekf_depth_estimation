# EKF Depth Estimation

The objective of this package is to estimate the distance between a robot and an object of unknown size.

This code whas developed as part of a Globaling Research Internship project under the mentorship of professor 
Krishna Vijayaraghavan at Simon Fraser University.

# Tracker Control Simulation & EKF Controller

This repository contains a ROS 2 package for simulating a slip-steer robot in Gazebo (Harmonic) and controlling it using an Extended Kalman Filter (EKF) and custom PI motor controllers. 

Because this package combines C++ build structures (`ament_cmake`) with Python execution libraries (`ament_cmake_python`), it requires a specific folder structure and build process.

## Prerequisites

Ensure your system has ROS 2 and Gazebo Harmonic installed, along with the necessary bridge and coordinate transform packages.

**Install Gazebo Harmonic:**
```bash
sudo apt update
sudo apt install curl lsb-release gnupg

# Add the Gazebo package repository
curl [https://packages.osrfoundation.org/gazebo.gpg](https://packages.osrfoundation.org/gazebo.gpg) --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] [http://packages.osrfoundation.org/gazebo/ubuntu-stable](http://packages.osrfoundation.org/gazebo/ubuntu-stable) $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
        
# Install Harmonic
sudo apt update
sudo apt install gz-harmonic
```

**Additional ROS Dependencies:**
```bash
sudo apt update
sudo apt install ros-dev-tools
sudo apt install ros-$ROS_DISTRO-ros-gz-sim ros-$ROS_DISTRO-ros-gz-bridge
sudo apt install ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-tf2-ros ros-$ROS_DISTRO-tf2-geometry-msgs
```

## 1. Setup Workspace & Clone Repositories

Create a new ROS 2 workspace (if you haven't already) and clone this repository into the `src` directory.

```bash
mkdir -p ~/workspace/src
cd ~/workspace/src
# Clone this repository (replace with your actual repo URL)
git clone [https://github.com/YOUR_USERNAME/tracker_control.git](https://github.com/YOUR_USERNAME/tracker_control.git)
```

## 2. Clone and Build the Motor Plugin

The robot's SDF relies on a custom motor plugin (`libDCMotorPlugin.so`). You must build this plugin separately so Gazebo can load it during runtime.

```bash
cd ~/workspace/src
# Clone the motor plugin repository
git clone https://github.com/VictorGarrett/gz_dc_motor.git

# Build the plugin using standard CMake
cd gz_dc_motor
mkdir build && cd build
cmake ..
make
```

## 3. Verify Directory Structure

**Top Level Workspace Structure**

```text
~/workspace/src
├── ekf_depth_estimation/                  
│   └── [CONTENT]
└── gz_dc_motor/
    └── [CONTENT]     
```


**tracker_control Package Structure**
```text
~/workspace/src/ekf_depth_estimation/
├── CMakeLists.txt
├── package.xml
├── launch/                  
│   └── ground_slip.launch.py        # ROS 2 Launch file
├── worlds/
│   └── test1.world          
├── models/
│   └── sliping_robot/
│       └── model.sdf        
│   
├── tracker_control/         # Inner Python module
│   ├── __init__.py          # Required for Python to recognize the module
│   ├── image_analyzer_ekf_slip_vel_absdepth_dyn.py  # Main executable
│   └── motor_controller.py  # Contains the PIController for the motors
│
└── tmux/                   # Startup scripts
    ├── session_slip.yml    # Tmux session definition
    └── start_slip.sh       # Startup script
```

## 4. Build the ROS 2 Package

Build from the root of your workspace. Using `--symlink-install` allows you to edit your Python files without having to rebuild the package every time.

```bash
cd ~/workspace

# Clean old build artifacts if needed
rm -rf build/tracker_control install/tracker_control

# Build the package
colcon build --symlink-install

# Source the workspace (you can also add this to .bashrc so you don't need to do it everytime)
source install/setup.bash
```

## 5. Running the System

The whole system is run by the tmux script, simply run ./start_slip.sh inside the tmux folder.


