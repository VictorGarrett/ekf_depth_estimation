cmake_minimum_required(VERSION 3.8)
project(tracker_control)

# Default to C++14 if any C++ is added later
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find ament_cmake and ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# --- INSTALLING PYTHON LIBRARY MODULE ---
# <-- ADDED: This installs the folder containing your __init__.py and motor_controller.py
ament_python_install_package(${PROJECT_NAME}) 

# --- INSTALLING PYTHON SCRIPTS ---
# In ROS 2, Python scripts are usually placed in a 'scripts' or 'nodes' folder.
# This command makes them executable and places them in the lib folder of the install space.
install(PROGRAMS
  tracker_control/image_analyzer_ekf_slip_vel_absdepth_dyn.py  # Replace with your actual filenames
  DESTINATION lib/${PROJECT_NAME}
)

# Install other directories (launch, config, etc.)
install(DIRECTORY launch config worlds models
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()