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
uint32 query_grid_width
uint32 query_grid_height
uint32 query_stride
uint32 original_image_width
uint32 original_image_height
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
    *   `N`: Number of query points. 等于 `query_grid_width * query_grid_height`，由 VGGT 模型在预处理后的图像上生成。
    *   `3`: Coordinates `(x, y, z)` in the world frame.
*   **Layout**:
    *   `dim[0]`: Label "S", Size `S`, Stride `S * N * 3`
    *   `dim[1]`: Label "N", Size `N`, Stride `N * 3`
    *   `dim[2]`: Label "C", Size `3`, Stride `3`
*   **Pixel Mapping**: 第 `n` 个查询点在原始图像上的坐标可通过以下方式计算：
    *   网格坐标: `grid_x = n % query_grid_width`, `grid_y = n / query_grid_width`
    *   原图坐标缩放比例: `scale_x = original_image_width / (query_grid_width * query_stride)`, `scale_y = original_image_height / (query_grid_height * query_stride)`
    *   原图像素坐标: `u = (grid_x * query_stride) * scale_x`, `v = (grid_y * query_stride) * scale_y`
    *   `tracks_3d[s, n]` 表示在帧 `s` 上该查询点对应的 3D 位置；相邻帧中相同的 `n` 代表同一物理点。
    *   This structure implicitly encodes the optical flow/alignment: `tracks_3d[s, n]` and `tracks_3d[s+1, n]` refer to the **same physical point** (identified by query index `n`) across time.

### 5. `tracks_mask`
*   **Type**: `std_msgs/Float32MultiArray`
*   **Description**: A flattened tensor representing the validity of each 3D track point.
*   **Logical Shape**: `(S, N)`
    *   `S`: Window size.
    *   `N`: Number of query points（与 `tracks_3d` 相同，等于 `query_grid_width * query_grid_height`）。
*   **Layout**:
    *   `dim[0]`: Label "S", Size `S`, Stride `S * N`
    *   `dim[1]`: Label "N", Size `N`, Stride `N`
*   **Values**: `1.0` indicates a valid track point, `0.0` indicates invalid/occluded.

### 6. `query_grid_width`
*   **Type**: `uint32`
*   **Description**: Width of the query grid used by VGGT model (after downsampling from preprocessed image).
*   **Note**: `N = query_grid_width * query_grid_height`

### 7. `query_grid_height`
*   **Type**: `uint32`
*   **Description**: Height of the query grid used by VGGT model (after downsampling from preprocessed image).

### 8. `query_stride`
*   **Type**: `uint32`
*   **Description**: Stride used when downsampling the preprocessed image to generate query points (typically 4).
*   **Note**: 查询点在预处理后图像上的间距，用于将查询网格坐标映射回原始图像坐标。

### 9. `original_image_width`
*   **Type**: `uint32`
*   **Description**: Width of the original input image before preprocessing.

### 10. `original_image_height`
*   **Type**: `uint32`
*   **Description**: Height of the original input image before preprocessing.

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
    
    # --- Get query grid dimensions ---
    grid_width = msg.query_grid_width
    grid_height = msg.query_grid_height
    query_stride = msg.query_stride
    orig_w = msg.original_image_width
    orig_h = msg.original_image_height
    
    # --- Accessing a specific point ---
    # To get the 3D point for query index n at frame s:
    # point = tracks_3d[s, n]
    
    # To map query index n back to original image coordinates:
    grid_x = n % grid_width
    grid_y = n // grid_width
    scale_x = orig_w / (grid_width * query_stride)
    scale_y = orig_h / (grid_height * query_stride)
    u = int((grid_x * query_stride) * scale_x)
    v = int((grid_y * query_stride) * scale_y)
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
    
    // Get query grid dimensions
    int grid_width = msg->query_grid_width;
    int grid_height = msg->query_grid_height;
    int query_stride = msg->query_stride;
    int orig_w = msg->original_image_width;
    int orig_h = msg->original_image_height;
    
    // Accessing data
    // To get point at frame s, query index n:
    int index_3d = s * (N * C) + n * C;
    int index_mask = s * N + n;
    
    if (mask[index_mask] > 0.5) {
        float x = data[index_3d + 0];
        float y = data[index_3d + 1];
        float z = data[index_3d + 2];
        
        // Map query index to original image coordinates
        int grid_x = n % grid_width;
        int grid_y = n / grid_width;
        float scale_x = static_cast<float>(orig_w) / (grid_width * query_stride);
        float scale_y = static_cast<float>(orig_h) / (grid_height * query_stride);
        int u = static_cast<int>((grid_x * query_stride) * scale_x);
        int v = static_cast<int>((grid_y * query_stride) * scale_y);
        
        // Process valid 3D point at pixel (u, v)...
    }
}
```
