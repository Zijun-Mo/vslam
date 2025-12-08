std_msgs/Header header
uint64 vggt_frame_id
geometry_msgs/PoseArray camera_poses
# Optional list of keyframe IDs corresponding to camera_poses order
uint64[] keyframe_ids
# Visibility ratios for each camera pose (normalized by latest frame visibility)
float32[] visibility_ratios
# Latest raw image (same frame as camera_poses[-1])
sensor_msgs/Image latest_image
# Dense 3D tracks tensor with shape (S, N, 3) where N = query_grid_height * query_grid_width
std_msgs/Float32MultiArray tracks_3d
# Dense 2D tracks tensor with shape (S, N, 2) in preprocessed image coordinates
# (aligned with tracks_mask layout)
std_msgs/Float32MultiArray tracks_2d
# Validity mask with shape (S, N)
std_msgs/Float32MultiArray tracks_mask
# Per-track RGB colors stored as uint8 with layout (S, N, 3)
std_msgs/UInt8MultiArray tracks_colors
# Dense RGB+XYZ point cloud flattened to (N, 6) with N = S * H * W.
# Each row stores [r, g, b, x, y, z] (colors in 0-255) in the same
# world frame as tracks_3d; invalid samples are zeros.
std_msgs/Float32MultiArray window_point_cloud
# Query grid dimensions (downsampled from original image)
uint32 query_grid_width
uint32 query_grid_height
uint32 query_stride
# Original image dimensions
uint32 original_image_width
uint32 original_image_height

# Full RGB window (preprocessed, aligned with depth) flattened to [S, H, W, 3]
std_msgs/UInt8MultiArray window_images
# Full depth window flattened to [S, H, W] (meters, same scale as window_point_cloud/depth_np)
std_msgs/Float32MultiArray window_depths
# Dimensions shared by all window frames
uint32 window_image_width
uint32 window_image_height

# Average intrinsic matrix (row-major 3x3) accumulated over all model inferences so far
float32[9] intrinsic_avg
# Number of samples used in intrinsic_avg
uint64 intrinsic_samples
