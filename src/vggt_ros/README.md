# VGGT ROS2 Package

This package provides a ROS2 node for VGGT (Visual Geometry Ground Truth?).

## Installation

1. Ensure you have ROS2 installed (e.g., Humble).
2. Install dependencies:
   ```bash
   pip install torch numpy opencv-python
   sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge
   ```
3. Build the package:
   ```bash
   cd /home/jun/vslam/vggt_ros
   colcon build
   source install/setup.bash
   ```

## Usage

Run the node using the launch file:

```bash
ros2 launch vggt_ros vggt.launch.py
```

You can modify the parameters in `config/vggt_params.yaml` or pass them via command line:

```bash
ros2 launch vggt_ros vggt.launch.py image_topic:=/my_camera/image
```

## Topics

- `vggt/world_points` (sensor_msgs/PointCloud2): Reconstructed 3D point cloud.
- `vggt/camera_poses` (geometry_msgs/PoseArray): Estimated camera poses.
- `vggt/tracks` (visualization_msgs/MarkerArray): 3D tracks of query points.
- `vggt/depth` (sensor_msgs/Image): Depth maps.
- `vggt/camera_info` (sensor_msgs/CameraInfo): Camera intrinsics.

## Parameters

- `model_name`: HuggingFace model name (default: "facebook/VGGT-1B").
- `device`: "cuda" or "cpu".
- `window_size`: Number of frames in the sliding window (default: 8).
- `image_topic`: Topic to subscribe to (default: "/camera/image_raw").
- `publish_rate`: Frequency of inference and publishing in Hz (default: 1.0).
