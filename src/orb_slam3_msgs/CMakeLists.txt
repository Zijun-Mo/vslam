cmake_minimum_required(VERSION 3.10)
project(orb_slam3_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/KeyFramePtr.msg
  msg/MapPointPtr.msg
  msg/SystemPtr.msg
)

ament_package()
