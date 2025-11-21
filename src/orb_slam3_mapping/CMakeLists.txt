cmake_minimum_required(VERSION 3.10)
project(orb_slam3_mapping)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(orb_slam3_lib REQUIRED)
find_package(vslam_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(mapping_component SHARED
  src/mapping_node.cpp
)
target_include_directories(mapping_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(mapping_component
  rclcpp
  rclcpp_components
  orb_slam3_lib
  vslam_msgs
  OpenCV
  Eigen3
)

rclcpp_components_register_node(mapping_component
  PLUGIN "orb_slam3_mapping::MappingNode"
  EXECUTABLE mapping_node_exe
)

install(TARGETS mapping_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_package()
