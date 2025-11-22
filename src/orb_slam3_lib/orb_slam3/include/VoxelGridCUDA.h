#ifndef ORB_SLAM3_VOXELGRIDCUDA_H
#define ORB_SLAM3_VOXELGRIDCUDA_H

#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace ORB_SLAM3
{

struct VoxelGridStats
{
    size_t input_points{0};
    size_t output_points{0};
    float voxel_size{0.0f};
    bool used_cuda{false};
};

bool VoxelGridDownsample(const std::vector<Eigen::Vector3f>& points,
                         const std::vector<cv::Vec3b>& colors,
                         float voxel_size,
                         std::vector<Eigen::Vector3f>& out_points,
                         std::vector<cv::Vec3b>& out_colors,
                         std::vector<uint64_t>* out_keys = nullptr,
                         Eigen::Vector3f* out_min_bound = nullptr,
                         VoxelGridStats* stats = nullptr);

bool IsCudaVoxelGridAvailable();

} // namespace ORB_SLAM3

#endif // ORB_SLAM3_VOXELGRIDCUDA_H
