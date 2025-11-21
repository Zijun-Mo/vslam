std_msgs/Header header
uint64 vggt_frame_id
geometry_msgs/PoseArray camera_poses
# Dense 3D tracks tensor with shape (S, H, W, 3)
std_msgs/Float32MultiArray tracks_3d
# Validity mask with shape (S, H, W)
std_msgs/Float32MultiArray tracks_mask
