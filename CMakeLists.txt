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

# If you have custom messages/services, you'd use rosidl_generate_interfaces here.
# Since they were commented out in your original file, I've omitted them.

# --- INSTALLING PYTHON SCRIPTS ---
# In ROS 2, Python scripts are usually placed in a 'scripts' or 'nodes' folder.
# This command makes them executable and places them in the lib folder of the install space.
install(PROGRAMS
  scripts/image_analyzer_ekf_slip_vel_absdepth_dyn.py  # Replace with your actual filenames
  DESTINATION lib/${PROJECT_NAME}
)

# Install other directories (launch, config, etc.)
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()