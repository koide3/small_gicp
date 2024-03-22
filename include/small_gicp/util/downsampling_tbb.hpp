#pragma once

#include <memory>

#include <tbb/tbb.h>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/util/vector3i_hash.hpp>

namespace small_gicp {

/**
 * @brief Voxel grid downsampling using TBB.
 * @note  This TBB version brings only a minor speedup compared to the single-thread version (e.g., 32-threads -> 1.4x speedup), and is not worth using usually.
 */
template <typename InputPointCloud, typename OutputPointCloud = InputPointCloud>
std::shared_ptr<OutputPointCloud> voxelgrid_sampling_tbb(const InputPointCloud& points, double leaf_size) {
  const double inv_leaf_size = 1.0 / leaf_size;

  typedef tbb::concurrent_hash_map<Eigen::Vector3i, int, XORVector3iHash> VoxelMap;

  std::atomic_uint64_t num_voxels = 0;
  VoxelMap voxels;
  std::vector<Eigen::Vector4d> voxel_values(traits::size(points), Eigen::Vector4d::Zero());

  const int chunk_size = 8;
  tbb::parallel_for(tbb::blocked_range<size_t>(0, traits::size(points), chunk_size), [&](const tbb::blocked_range<size_t>& range) {
    std::vector<Eigen::Vector3i> coords;
    std::vector<Eigen::Vector4d> values;
    coords.reserve(range.size());
    values.reserve(range.size());

    for (size_t i = range.begin(); i < range.end(); i++) {
      const Eigen::Vector4d pt = traits::point(points, i);
      const Eigen::Vector3i coord = (pt * inv_leaf_size).array().floor().template cast<int>().template head<3>();

      auto found = std::ranges::find(coords, coord);
      if (found == coords.end()) {
        coords.emplace_back(coord);
        values.emplace_back(pt);
      } else {
        values[std::distance(coords.begin(), found)] += pt;
      }
    }

    for (size_t i = 0; i < coords.size(); i++) {
      VoxelMap::accessor a;
      if (voxels.insert(a, coords[i])) {
        a->second = num_voxels++;
        voxel_values[a->second] = values[i];
      } else {
        voxel_values[a->second] += values[i];
      }
    }
  });

  const int N = num_voxels;
  auto downsampled = std::make_shared<OutputPointCloud>();
  traits::resize(*downsampled, N);

  for (size_t i = 0; i < N; i++) {
    const Eigen::Vector4d pt = voxel_values[i];
    traits::set_point(*downsampled, i, pt / pt.w());
  }

  return downsampled;
}

}  // namespace small_gicp
