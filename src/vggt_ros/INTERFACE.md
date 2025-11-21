# VGGT Node Interface Documentation

This document describes the interface for the `vggt_node` in the `vggt_ros` package, specifically focusing on the published message format and data organization.

## Published Topics

### `vggt/output`

*   **Topic Name**: `vggt/output`
*   **Message Type**: `vslam_msgs/msg/VggtOutput`
*   **Description**: Publishes the inference results from the VGGT model, including camera poses and dense 3D tracks for a sliding window of frames.

## Message Definition (`vslam_msgs/msg/VggtOutput`)

```
std_msgs/Header header
uint64 vggt_frame_id
geometry_msgs/PoseArray camera_poses
std_msgs/Float32MultiArray tracks_3d
std_msgs/Float32MultiArray tracks_mask
```

## Field Details

### 1. `header`
*   **Type**: `std_msgs/Header`
*   **Description**: Standard ROS header.
    *   `stamp`: Corresponds to the timestamp of the **latest** frame in the processing window.
    *   `frame_id`: Typically "map" or the reference frame for the poses.

### 2. `vggt_frame_id`
*   **Type**: `uint64`
*   **Description**: A monotonically increasing identifier for each published message, useful for detecting dropped messages or sequencing. 下游的 `orb_slam3_vggt_frontend` 依赖该字段判断 VGGT 滑窗是否连续：若 frame_id 连续，则维持上一窗口的全局 Track ID 映射；一旦出现跳变，则为新轨迹点分配新的全局 ID，以便映射到 ORB-SLAM3 的 `MapPoint`。

### 3. `camera_poses`
*   **Type**: `geometry_msgs/PoseArray`
*   **Description**: Contains the estimated camera poses for all frames in the current sliding window.
*   **Size**: Equal to the window size `S`.
*   **Coordinate System**: Poses are relative to the frame specified in `header.frame_id`.

### 4. `tracks_3d`
*   **Type**: `std_msgs/Float32MultiArray`
*   **Description**: A flattened tensor representing dense 3D tracks。下游节点会基于 `(s, n)` 的索引构造 `vTrackIds`，并借助 `vggt_frame_id` 延续这些 ID，从而跨窗口复用同一物理点的 `MapPoint`。
*   **Logical Shape**: `(S, N, 3)`
    *   `S`: Window size (number of frames).
    *   `N`: Number of query points. 为了避免显存占用过大，节点对像素网格做了 `stride = 4` 的子采样，因此 `N = (Height / 4) * (Width / 4)`（向下取整后相乘）。
    *   `3`: Coordinates `(x, y, z)` in the world frame.
*   **Layout**:
    *   `dim[0]`: Label "S", Size `S`, Stride `S * N * 3`
    *   `dim[1]`: Label "N", Size `N`, Stride `N * 3`
    *   `dim[2]`: Label "C", Size `3`, Stride `3`
*   **Pixel Mapping**: 第 `n` 个采样点对应的像素坐标为 `(u, v)`，其中 `u = (n % (Width/4)) * 4`，`v = (n / (Width/4)) * 4`（行优先顺序，同样假设能被 4 整除）。
    *   `tracks_3d[s, n]` 表示在帧 `s` 上该下采样像素 `(u, v)` 的 3D 位置；相邻帧中相同的 `n` 代表同一物理点。
    *   This structure implicitly encodes the optical flow/alignment: `tracks_3d[s, n]` and `tracks_3d[s+1, n]` refer to the **same physical point** (identified by pixel index `n` in the query frame) across time.

### 5. `tracks_mask`
*   **Type**: `std_msgs/Float32MultiArray`
*   **Description**: A flattened tensor representing the validity of each 3D track point.
*   **Logical Shape**: `(S, N)`
    *   `S`: Window size.
    *   `N`: Number of下采样查询点（与 `tracks_3d` 相同，约等于 `(H/4)*(W/4)`）。
*   **Layout**:
    *   `dim[0]`: Label "S", Size `S`, Stride `S * N`
    *   `dim[1]`: Label "N", Size `N`, Stride `N`
*   **Values**: `1.0` indicates a valid track point, `0.0` indicates invalid/occluded.

## Parsing Guide

### Python Example

```python
import numpy as np
from vslam_msgs.msg import VggtOutput

def callback(msg: VggtOutput):
    # --- Parse tracks_3d ---
    # Get dimensions from layout
    S = msg.tracks_3d.layout.dim[0].size
    N = msg.tracks_3d.layout.dim[1].size
    C = msg.tracks_3d.layout.dim[2].size
    
    # Reshape data to (S, N, 3)
    tracks_3d = np.array(msg.tracks_3d.data).reshape((S, N, C))
    
    # --- Parse tracks_mask ---
    # Reshape data to (S, N)
    tracks_mask = np.array(msg.tracks_mask.data).reshape((S, N))
    
    # --- Accessing a specific point ---
    # Assuming you know the image resolution H x W
    # N = H * W
    # To get the 3D point for pixel (u, v) at frame s:
    # idx = v * W + u
    # point = tracks_3d[s, idx]
```

### C++ Example

```cpp
#include "vslam_msgs/msg/vggt_output.hpp"

void callback(const vslam_msgs::msg::VggtOutput::SharedPtr msg) {
    // Get dimensions
    int S = msg->tracks_3d.layout.dim[0].size;
    int N = msg->tracks_3d.layout.dim[1].size;
    int C = msg->tracks_3d.layout.dim[2].size;
    
    const auto& data = msg->tracks_3d.data;
    const auto& mask = msg->tracks_mask.data;
    
    // Accessing data
    // To get point at frame s, pixel index n:
    int index_3d = s * (N * C) + n * C;
    int index_mask = s * N + n;
    
    if (mask[index_mask] > 0.5) {
        float x = data[index_3d + 0];
        float y = data[index_3d + 1];
        float z = data[index_3d + 2];
        // Process valid 3D point...
    }
}
```
