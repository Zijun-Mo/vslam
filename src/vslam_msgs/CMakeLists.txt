cmake_minimum_required(VERSION 3.10)
project(vslam_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/KeyFramePtr.msg
  msg/MapPointPtr.msg
  msg/SystemPtr.msg
  msg/VggtOutput.msg
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_package()
