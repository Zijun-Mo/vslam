#include "VoxelGridCUDA.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/system_error.h>

#include <unordered_map>
#include <cmath>
#include <cstdio>

namespace ORB_SLAM3
{
namespace
{
constexpr int kVoxelBias = 1 << 20;
constexpr int kMaxCoord = (1 << 21) - 1;
constexpr uint64_t kCoordMask = (1ull << 21) - 1ull;

__device__ __host__ inline uint64_t PackKey(int ix, int iy, int iz)
{
	ix = max(0, min(ix, kMaxCoord));
	iy = max(0, min(iy, kMaxCoord));
	iz = max(0, min(iz, kMaxCoord));
	return ((static_cast<uint64_t>(ix) & kCoordMask) << 42) |
		   ((static_cast<uint64_t>(iy) & kCoordMask) << 21) |
		   (static_cast<uint64_t>(iz) & kCoordMask);
}

__global__ void ComputeVoxelIndices(const float3* points,
									size_t count,
									float3 min_bound,
									float inv_voxel,
									uint64_t* keys)
{
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= count)
		return;

	const float3 p = points[idx];
	const int ix = static_cast<int>(floorf((p.x - min_bound.x) * inv_voxel)) + kVoxelBias;
	const int iy = static_cast<int>(floorf((p.y - min_bound.y) * inv_voxel)) + kVoxelBias;
	const int iz = static_cast<int>(floorf((p.z - min_bound.z) * inv_voxel)) + kVoxelBias;
	keys[idx] = PackKey(ix, iy, iz);
}

__global__ void AveragePointsInVoxels(const float3* points,
									  const uchar3* colors,
									  const int* offsets,
									  const int* counts,
									  size_t voxel_count,
									  float3* pos_sums,
									  float3* color_sums)
{
	const size_t voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(voxel_idx >= voxel_count)
		return;

	const int start = offsets[voxel_idx];
	const int length = counts[voxel_idx];

	float3 pos_sum = make_float3(0.f, 0.f, 0.f);
	float3 col_sum = make_float3(0.f, 0.f, 0.f);

	for(int i = 0; i < length; ++i)
	{
		const float3 p = points[start + i];
		const uchar3 c = colors[start + i];
		pos_sum.x += p.x;
		pos_sum.y += p.y;
		pos_sum.z += p.z;
		col_sum.x += static_cast<float>(c.x);
		col_sum.y += static_cast<float>(c.y);
		col_sum.z += static_cast<float>(c.z);
	}

	pos_sums[voxel_idx] = pos_sum;
	color_sums[voxel_idx] = col_sum;
}

__global__ void CompactResults(const uint64_t* keys,
							   const float3* pos_sums,
							   const float3* color_sums,
							   const int* counts,
							   size_t voxel_count,
							   float3* out_points,
							   uchar3* out_colors)
{
	const size_t voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(voxel_idx >= voxel_count)
		return;

	const int sample_count = max(1, counts[voxel_idx]);
	const float inv = 1.0f / static_cast<float>(sample_count);

	float3 avg_pos;
	avg_pos.x = pos_sums[voxel_idx].x * inv;
	avg_pos.y = pos_sums[voxel_idx].y * inv;
	avg_pos.z = pos_sums[voxel_idx].z * inv;

	float3 avg_col;
	avg_col.x = color_sums[voxel_idx].x * inv;
	avg_col.y = color_sums[voxel_idx].y * inv;
	avg_col.z = color_sums[voxel_idx].z * inv;

	out_points[voxel_idx] = avg_pos;
	out_colors[voxel_idx] = make_uchar3(
		static_cast<unsigned char>(fminf(255.f, fmaxf(0.f, avg_col.x))),
		static_cast<unsigned char>(fminf(255.f, fmaxf(0.f, avg_col.y))),
		static_cast<unsigned char>(fminf(255.f, fmaxf(0.f, avg_col.z))));
}

struct CPUAccumulator
{
	Eigen::Vector3f pos_sum = Eigen::Vector3f::Zero();
	Eigen::Vector3f color_sum = Eigen::Vector3f::Zero();
	size_t count = 0;
};

bool CPUFallbackVoxelGrid(const std::vector<Eigen::Vector3f>& points,
						  const std::vector<cv::Vec3b>& colors,
						  float voxel_size,
						  std::vector<Eigen::Vector3f>& out_points,
						  std::vector<cv::Vec3b>& out_colors,
						  std::vector<uint64_t>* out_keys,
						  Eigen::Vector3f* out_min_bound,
						  VoxelGridStats* stats)
{
	if(voxel_size <= 0.f || points.empty())
		return false;

	const float inv_voxel = 1.0f / voxel_size;
	Eigen::Vector3f min_bound = points.front();
	for(const auto& p : points)
		min_bound = min_bound.cwiseMin(p);

	if(out_min_bound)
		*out_min_bound = min_bound;

	auto compute_key = [&](const Eigen::Vector3f& p) {
		const int ix = static_cast<int>(std::floor((p.x() - min_bound.x()) * inv_voxel)) + kVoxelBias;
		const int iy = static_cast<int>(std::floor((p.y() - min_bound.y()) * inv_voxel)) + kVoxelBias;
		const int iz = static_cast<int>(std::floor((p.z() - min_bound.z()) * inv_voxel)) + kVoxelBias;
		return PackKey(ix, iy, iz);
	};

	std::unordered_map<uint64_t, CPUAccumulator> grid;
	grid.reserve(points.size());

	for(size_t i = 0; i < points.size(); ++i)
	{
		const Eigen::Vector3f& p = points[i];
		CPUAccumulator& acc = grid[compute_key(p)];
		acc.pos_sum += p;
		const cv::Vec3b color = (i < colors.size()) ? colors[i] : cv::Vec3b(255,255,255);
		acc.color_sum += Eigen::Vector3f(static_cast<float>(color[0]),
										 static_cast<float>(color[1]),
										 static_cast<float>(color[2]));
		acc.count++;
	}

	out_points.clear();
	out_colors.clear();
	out_points.reserve(grid.size());
	out_colors.reserve(grid.size());
	if(out_keys)
		out_keys->clear();

	for(const auto& entry : grid)
	{
		if(entry.second.count == 0)
			continue;
		const float inv = 1.0f / static_cast<float>(entry.second.count);
		const Eigen::Vector3f avg_pos = entry.second.pos_sum * inv;
		const Eigen::Vector3f avg_color = entry.second.color_sum * inv;
		cv::Vec3b color(
			static_cast<uchar>(std::round(std::min(255.f, std::max(0.f, avg_color[0])))),
			static_cast<uchar>(std::round(std::min(255.f, std::max(0.f, avg_color[1])))),
			static_cast<uchar>(std::round(std::min(255.f, std::max(0.f, avg_color[2]))))
		);
		out_points.push_back(avg_pos);
		out_colors.push_back(color);
		if(out_keys)
			out_keys->push_back(entry.first);
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

inline bool ShouldUseCuda(size_t point_count)
{
	constexpr size_t kMinPointsForCuda = 2000;
	return point_count >= kMinPointsForCuda;
}

inline bool CheckCuda(cudaError_t err, const char* context)
{
	if(err == cudaSuccess)
		return true;
	std::fprintf(stderr, "[VoxelGridCUDA] CUDA error at %s: %s\n", context, cudaGetErrorString(err));
	return false;
}

} // namespace

bool IsCudaVoxelGridAvailable()
{
#ifdef __CUDACC__
	static int cached = -1;
	if(cached != -1)
		return cached == 1;

	int device_count = 0;
	const cudaError_t err = cudaGetDeviceCount(&device_count);
	if(err != cudaSuccess || device_count <= 0)
	{
		cached = 0;
		return false;
	}

	cached = 1;
	return true;
#else
	return false;
#endif
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
	out_points.clear();
	out_colors.clear();

	if(points.empty() || voxel_size <= 0.f)
		return false;

#ifndef __CUDACC__
	return CPUFallbackVoxelGrid(points, colors, voxel_size, out_points, out_colors, out_keys, out_min_bound, stats);
#else
	if(!IsCudaVoxelGridAvailable() || !ShouldUseCuda(points.size()))
	{
		return CPUFallbackVoxelGrid(points, colors, voxel_size, out_points, out_colors, out_keys, out_min_bound, stats);
	}

	Eigen::Vector3f min_bound = points.front();
	for(const auto& p : points)
		min_bound = min_bound.cwiseMin(p);
	if(out_min_bound)
		*out_min_bound = min_bound;

	const float inv_voxel = 1.0f / voxel_size;
	const float3 min_bound_f = make_float3(min_bound.x(), min_bound.y(), min_bound.z());

	const size_t count = points.size();
	thrust::host_vector<float3> h_points(count);
	thrust::host_vector<uchar3> h_colors(count);
	for(size_t i = 0; i < count; ++i)
	{
		const Eigen::Vector3f& p = points[i];
		h_points[i] = make_float3(p.x(), p.y(), p.z());
		const cv::Vec3b color = (i < colors.size()) ? colors[i] : cv::Vec3b(255,255,255);
		h_colors[i] = make_uchar3(color[0], color[1], color[2]);
	}

	try
	{
		thrust::device_vector<float3> d_points = h_points;
		thrust::device_vector<uchar3> d_colors = h_colors;
		thrust::device_vector<uint64_t> d_keys(count);

		const dim3 block(256);
		const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
		ComputeVoxelIndices<<<grid, block>>>(thrust::raw_pointer_cast(d_points.data()),
											 count,
											 min_bound_f,
											 inv_voxel,
											 thrust::raw_pointer_cast(d_keys.data()));
		if(!CheckCuda(cudaGetLastError(), "ComputeVoxelIndices"))
			return CPUFallbackVoxelGrid(points, colors, voxel_size, out_points, out_colors, out_keys, out_min_bound, stats);

		auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(d_points.begin(), d_colors.begin()));
		auto zipped_end = thrust::make_zip_iterator(thrust::make_tuple(d_points.end(), d_colors.end()));
		thrust::sort_by_key(d_keys.begin(), d_keys.end(), zipped_begin);

		thrust::device_vector<uint64_t> d_unique_keys(count);
		thrust::device_vector<int> d_counts(count);

		auto reduce_end = thrust::reduce_by_key(d_keys.begin(), d_keys.end(),
												thrust::make_constant_iterator<int>(1),
												d_unique_keys.begin(),
												d_counts.begin());

		const size_t voxel_count = reduce_end.first - d_unique_keys.begin();
		if(voxel_count == 0)
			return false;

		d_unique_keys.resize(voxel_count);
		d_counts.resize(voxel_count);

		thrust::device_vector<int> d_offsets(voxel_count);
		thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_offsets.begin());

		thrust::device_vector<float3> d_pos_sums(voxel_count);
		thrust::device_vector<float3> d_color_sums(voxel_count);

		const dim3 voxel_grid(static_cast<unsigned int>((voxel_count + block.x - 1) / block.x));
		AveragePointsInVoxels<<<voxel_grid, block>>>(thrust::raw_pointer_cast(d_points.data()),
													 thrust::raw_pointer_cast(d_colors.data()),
													 thrust::raw_pointer_cast(d_offsets.data()),
													 thrust::raw_pointer_cast(d_counts.data()),
													 voxel_count,
													 thrust::raw_pointer_cast(d_pos_sums.data()),
													 thrust::raw_pointer_cast(d_color_sums.data()));
		if(!CheckCuda(cudaGetLastError(), "AveragePointsInVoxels"))
			return CPUFallbackVoxelGrid(points, colors, voxel_size, out_points, out_colors, out_keys, out_min_bound, stats);

		thrust::device_vector<float3> d_avg_points(voxel_count);
		thrust::device_vector<uchar3> d_avg_colors(voxel_count);

		CompactResults<<<voxel_grid, block>>>(thrust::raw_pointer_cast(d_unique_keys.data()),
											  thrust::raw_pointer_cast(d_pos_sums.data()),
											  thrust::raw_pointer_cast(d_color_sums.data()),
											  thrust::raw_pointer_cast(d_counts.data()),
											  voxel_count,
											  thrust::raw_pointer_cast(d_avg_points.data()),
											  thrust::raw_pointer_cast(d_avg_colors.data()));
		if(!CheckCuda(cudaGetLastError(), "CompactResults"))
			return CPUFallbackVoxelGrid(points, colors, voxel_size, out_points, out_colors, out_keys, out_min_bound, stats);

		thrust::host_vector<float3> h_avg_points = d_avg_points;
		thrust::host_vector<uchar3> h_avg_colors = d_avg_colors;
		thrust::host_vector<uint64_t> h_unique_keys = d_unique_keys;

		out_points.resize(voxel_count);
		out_colors.resize(voxel_count);
		for(size_t i = 0; i < voxel_count; ++i)
		{
			out_points[i] = Eigen::Vector3f(h_avg_points[i].x,
											h_avg_points[i].y,
											h_avg_points[i].z);
			cv::Vec3b color;
			color[0] = h_avg_colors[i].x;
			color[1] = h_avg_colors[i].y;
			color[2] = h_avg_colors[i].z;
			out_colors[i] = color;
		}

		if(out_keys)
			out_keys->assign(h_unique_keys.begin(), h_unique_keys.end());

		if(stats)
		{
			stats->input_points = points.size();
			stats->output_points = out_points.size();
			stats->voxel_size = voxel_size;
			stats->used_cuda = true;
		}
		return true;
	}
	catch(const thrust::system_error& e)
	{
		std::fprintf(stderr, "[VoxelGridCUDA] Thrust error: %s\n", e.what());
		return CPUFallbackVoxelGrid(points, colors, voxel_size, out_points, out_colors, out_keys, out_min_bound, stats);
	}
#endif
}

} // namespace ORB_SLAM3
