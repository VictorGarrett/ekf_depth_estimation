cmake_minimum_required(VERSION 3.0.2)
project(tracker_control)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  std_srvs
  tf2_ros
  tf2_geometry_msgs
  sensor_msgs
  cv_bridge
  #message_generation
  #mavros_msgs
)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
##add_service_files(
  ## FILES
  ## MoveToPosition.srv
##)

## Generate added messages and services with any dependencies listed here
#generate_messages(
#   DEPENDENCIES
#   std_msgs
#   geometry_msgs
#   nav_msgs
#   #mavros_msgs
#)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs sensor_msgs message_runtime nav_msgs std_srvs tf2_geometry_msgs cv_bridge
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
  include/
)


#add_executable(${PROJECT_NAME}_node src/main.cpp src/drone.cpp src/drone_state.cpp ${STATE_SRC_FILES})


## Link libraries
#target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES})
