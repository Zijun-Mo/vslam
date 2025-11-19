cmake_minimum_required(VERSION 3.10)
project(orb_slam3_lib)

set(ORB_SLAM3_SRC_DIR ${PROJECT_SOURCE_DIR}/src/orb_slam3/src)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

find_package(ament_cmake REQUIRED)
find_package(OpenCV 4.2 REQUIRED)
find_package(Eigen3 3.3.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/orb_slam3
  ${PROJECT_SOURCE_DIR}/orb_slam3/include
  ${PROJECT_SOURCE_DIR}/orb_slam3/include/CameraModels
  ${PROJECT_SOURCE_DIR}/orb_slam3/Thirdparty
  ${PROJECT_SOURCE_DIR}/orb_slam3/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
add_library(orb_slam3_lib SHARED
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/System.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/Tracking.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/LocalMapping.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/LoopClosing.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/ORBextractor.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/ORBmatcher.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/FrameDrawer.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/Converter.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/MapPoint.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/KeyFrame.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/Atlas.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/Map.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/MapDrawer.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/Optimizer.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/Frame.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/KeyFrameDatabase.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/Sim3Solver.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/Viewer.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/ImuTypes.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/G2oTypes.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/CameraModels/Pinhole.cpp
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/CameraModels/KannalaBrandt8.cpp
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/OptimizableTypes.cpp
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/MLPnPsolver.cpp
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/GeometricTools.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/TwoViewReconstruction.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/Config.cc
  ${PROJECT_SOURCE_DIR}/orb_slam3/src/Settings.cc
)

# Find the .so files provided in orb_slam3/Thirdparty projects
set(DBoW2_PATH "${PROJECT_SOURCE_DIR}/orb_slam3/Thirdparty/DBoW2/lib/libDBoW2.so")
set(g2o_PATH "${PROJECT_SOURCE_DIR}/orb_slam3/Thirdparty/g2o/lib/libg2o.so")

target_link_libraries(orb_slam3_lib
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${Pangolin_LIBRARIES}
  ${DBoW2_PATH}
  ${g2o_PATH}
  -lboost_system
  -lboost_serialization
  -lcrypto 
)

# Install .so files
install(FILES 
  ${DBoW2_PATH}
  ${g2o_PATH}
  DESTINATION lib
)

# Install headers
install(DIRECTORY orb_slam3/include/
  DESTINATION include
)
install(DIRECTORY orb_slam3/Thirdparty/Sophus/sophus
  DESTINATION include/Thirdparty/Sophus
)
install(DIRECTORY orb_slam3/Thirdparty/DBoW2/DBoW2
  DESTINATION include/Thirdparty/DBoW2
)
install(DIRECTORY orb_slam3/Thirdparty/DBoW2/DUtils
  DESTINATION include/Thirdparty/DBoW2
)
install(DIRECTORY orb_slam3/Thirdparty/g2o/g2o
  DESTINATION include/Thirdparty/g2o
)
install(FILES orb_slam3/Thirdparty/g2o/config.h
  DESTINATION include/Thirdparty/g2o
)

# Install library
install(TARGETS orb_slam3_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

set(THIS_PACKAGE_INCLUDE_DEPENDS
  OpenCV
  Eigen3
  Pangolin
)

ament_export_include_directories(include include/Thirdparty/Sophus include/CameraModels)
ament_export_libraries(orb_slam3_lib)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})
ament_package()
