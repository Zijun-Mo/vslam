cmake_minimum_required(VERSION 3.10)
project(orb_slam3_tracking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(orb_slam3_lib REQUIRED)
find_package(vslam_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(tracking_component SHARED
  src/tracking_node.cpp
)
target_include_directories(tracking_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(tracking_component
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  orb_slam3_lib
  vslam_msgs
  OpenCV
  Eigen3
)

rclcpp_components_register_node(tracking_component
  PLUGIN "orb_slam3_tracking::TrackingNode"
  EXECUTABLE tracking_node_exe
)

install(TARGETS tracking_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
