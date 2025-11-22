std_msgs/Header header
uint64 vggt_frame_id
geometry_msgs/PoseArray camera_poses
# Dense 3D tracks tensor with shape (S, N, 3) where N = query_grid_height * query_grid_width
std_msgs/Float32MultiArray tracks_3d
# Validity mask with shape (S, N)
std_msgs/Float32MultiArray tracks_mask
# Per-track RGB colors stored as uint8 with layout (S, N, 3)
std_msgs/UInt8MultiArray tracks_colors
# Query grid dimensions (downsampled from original image)
uint32 query_grid_width
uint32 query_grid_height
uint32 query_stride
# Original image dimensions
uint32 original_image_width
uint32 original_image_height
