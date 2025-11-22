#include "VoxelGridCUDA.h"

#include <unordered_map>
#include <cmath>

namespace ORB_SLAM3
{
namespace
{
struct GridCoord
{
    int x;
    int y;
    int z;

    bool operator==(const GridCoord& other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct GridCoordHash
{
    size_t operator()(const GridCoord& coord) const noexcept
    {
        const size_t h1 = std::hash<int>{}(coord.x);
        const size_t h2 = std::hash<int>{}(coord.y);
        const size_t h3 = std::hash<int>{}(coord.z);
        return ((h1 * 73856093u) ^ (h2 * 19349663u) ^ (h3 * 83492791u));
    }
};
}

static bool CPUVoxelGrid(const std::vector<Eigen::Vector3f>& points,
                         const std::vector<cv::Vec3b>& colors,
                         float voxel_size,
                         std::vector<Eigen::Vector3f>& out_points,
                         std::vector<cv::Vec3b>& out_colors,
                         std::vector<uint64_t>* out_keys,
                         Eigen::Vector3f* out_min_bound,
                         VoxelGridStats* stats)
{
    if(points.empty() || voxel_size <= 0.f)
        return false;

    const float inv_voxel = 1.0f / voxel_size;
    Eigen::Vector3f min_bound = points.front();
    for(const auto& p : points)
    {
        min_bound = min_bound.cwiseMin(p);
    }

    if(out_min_bound)
        *out_min_bound = min_bound;

    const int bias = 1 << 20;
    const uint64_t mask = (1ull << 21) - 1ull;
    auto compute_key = [&](const Eigen::Vector3f& p)
    {
        int ix = static_cast<int>(std::floor((p.x() - min_bound.x()) * inv_voxel)) + bias;
        int iy = static_cast<int>(std::floor((p.y() - min_bound.y()) * inv_voxel)) + bias;
        int iz = static_cast<int>(std::floor((p.z() - min_bound.z()) * inv_voxel)) + bias;
        ix = std::max(0, std::min(ix, (1 << 21) - 1));
        iy = std::max(0, std::min(iy, (1 << 21) - 1));
        iz = std::max(0, std::min(iz, (1 << 21) - 1));
        return ((static_cast<uint64_t>(ix) & mask) << 42) |
               ((static_cast<uint64_t>(iy) & mask) << 21) |
               (static_cast<uint64_t>(iz) & mask);
    };

    std::unordered_map<uint64_t, Eigen::Vector3f> pos_sum;
    std::unordered_map<uint64_t, Eigen::Vector3f> color_sum;
    std::unordered_map<uint64_t, size_t> counts;
    pos_sum.reserve(points.size());
    color_sum.reserve(points.size());
    counts.reserve(points.size());

    for(size_t i = 0; i < points.size(); ++i)
    {
        const Eigen::Vector3f& p = points[i];
        const uint64_t key = compute_key(p);

        pos_sum[key] += p;
        Eigen::Vector3f c_vec(255.f, 255.f, 255.f);
        if(i < colors.size())
        {
            const cv::Vec3b& c = colors[i];
            c_vec = Eigen::Vector3f(static_cast<float>(c[0]), static_cast<float>(c[1]), static_cast<float>(c[2]));
        }
        color_sum[key] += c_vec;
        counts[key] += 1;
    }

    out_points.clear();
    out_colors.clear();
    out_points.reserve(pos_sum.size());
    out_colors.reserve(pos_sum.size());
    if(out_keys)
        out_keys->clear();

    for(const auto& kv : pos_sum)
    {
        const uint64_t key = kv.first;
        const size_t count = counts[key];
        if(count == 0)
            continue;

        const float inv = 1.0f / static_cast<float>(count);
        Eigen::Vector3f avg_pos = kv.second * inv;
        Eigen::Vector3f avg_color = color_sum[key] * inv;
        cv::Vec3b color(
            static_cast<uchar>(std::round(std::min(255.f, std::max(0.f, avg_color[0])))),
            static_cast<uchar>(std::round(std::min(255.f, std::max(0.f, avg_color[1])))),
            static_cast<uchar>(std::round(std::min(255.f, std::max(0.f, avg_color[2]))))
        );

        out_points.push_back(avg_pos);
        out_colors.push_back(color);
        if(out_keys)
            out_keys->push_back(key);
    }

    if(stats)
    {
        stats->input_points = points.size();
        stats->output_points = out_points.size();
        stats->voxel_size = voxel_size;
        stats->used_cuda = false;
    }

    return true;
}

bool IsCudaVoxelGridAvailable()
{
    return false;
}

bool VoxelGridDownsample(const std::vector<Eigen::Vector3f>& points,
                         const std::vector<cv::Vec3b>& colors,
                         float voxel_size,
                         std::vector<Eigen::Vector3f>& out_points,
                         std::vector<cv::Vec3b>& out_colors,
                         std::vector<uint64_t>* out_keys,
                         Eigen::Vector3f* out_min_bound,
                         VoxelGridStats* stats)
{
    return CPUVoxelGrid(points, colors, voxel_size, out_points, out_colors, out_keys, out_min_bound, stats);
}

} // namespace ORB_SLAM3
