cmake_minimum_required(VERSION 3.5)
project(orb_slam3_vggt_frontend)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(orb_slam3_lib REQUIRED)
find_package(orb_slam3_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(GLEW REQUIRED)

# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${orb_slam3_lib_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${GLEW_INCLUDE_DIRS}
)

# Create the node executable
add_executable(vggt_frontend_node src/vggt_frontend_node.cpp)
ament_target_dependencies(vggt_frontend_node
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
  image_transport
  message_filters
  orb_slam3_msgs
)

# Link against ORB_SLAM3 library
target_link_libraries(vggt_frontend_node
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${orb_slam3_lib_LIBRARIES}
)

# Install
install(TARGETS vggt_frontend_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
