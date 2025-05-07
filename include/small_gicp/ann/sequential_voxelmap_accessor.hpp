// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <small_gicp/points/traits.hpp>
#include <small_gicp/ann/incremental_voxelmap.hpp>

namespace small_gicp {

/**
 * @brief A wrapper class to sequentially access points in a voxelmap (e.g., using points in a voxelmap as source point cloud for registration).
 * @note  If the contents of the voxelmap are changed, the accessor must be recreated.
 * @example
    small_gicp::IncrementalVoxelMap<small_gicp::FlatContainerCov>::Ptr last_voxelmap = ...;
    small_gicp::IncrementalVoxelMap<small_gicp::FlatContainerCov>::Ptr voxelmap = ...;

    // Create a sequential accessor
    const auto accessor = small_gicp::create_sequential_accessor(*voxelmap);

    // Use the voxelmap as a source point cloud
    small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> registration;
    auto result = registration.align(*last_voxelmap, accessor, *last_voxelmap, Eigen::Isometry3d::Identity());
 */
template <typename VoxelMap>
class SequentialVoxelmapAccessor {
public:
  /// @brief Constructor.
  /// @param voxelmap Voxelmap
  SequentialVoxelmapAccessor(const VoxelMap& voxelmap) : voxelmap(voxelmap), indices(traits::point_indices(voxelmap)) {}

  /// @brief Number of points in the voxelmap.
  size_t size() const { return indices.size(); }

public:
  const VoxelMap& voxelmap;           ///< Voxelmap
  const std::vector<size_t> indices;  ///< Indices of points in the voxelmap
};

/// @brief Create a sequential voxelmap accessor.
template <typename VoxelMap>
SequentialVoxelmapAccessor<VoxelMap> create_sequential_accessor(const VoxelMap& voxelmap) {
  return SequentialVoxelmapAccessor<VoxelMap>(voxelmap);
}

template <typename VoxelMap>
struct traits::Traits<SequentialVoxelmapAccessor<VoxelMap>> {
  static size_t size(const SequentialVoxelmapAccessor<VoxelMap>& accessor) { return accessor.size(); }

  static bool has_points(const SequentialVoxelmapAccessor<VoxelMap>& accessor) { traits::has_points(accessor.voxelmap); }
  static bool has_normals(const SequentialVoxelmapAccessor<VoxelMap>& accessor) { return traits::has_normals(accessor.voxelmap); }
  static bool has_covs(const SequentialVoxelmapAccessor<VoxelMap>& accessor) { return traits::has_covs(accessor.voxelmap); }

  static Eigen::Vector4d point(const SequentialVoxelmapAccessor<VoxelMap>& accessor, size_t i) { return traits::point(accessor.voxelmap, accessor.indices[i]); }
  static Eigen::Vector4d normal(const SequentialVoxelmapAccessor<VoxelMap>& accessor, size_t i) { return traits::normal(accessor.voxelmap, accessor.indices[i]); }
  static Eigen::Matrix4d cov(const SequentialVoxelmapAccessor<VoxelMap>& accessor, size_t i) { return traits::cov(accessor.voxelmap, accessor.indices[i]); }
};

}  // namespace small_gicp
