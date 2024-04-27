// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <small_gicp/ann/traits.hpp>
#include <small_gicp/ann/flat_container.hpp>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/util/fast_floor.hpp>
#include <small_gicp/util/vector3i_hash.hpp>

namespace small_gicp {

/// @brief Voxel meta information
struct VoxelInfo {
public:
  /// @brief Constructor
  /// @param coord Voxel coordinate
  /// @param lru   LRU counter for caching
  VoxelInfo(const Eigen::Vector3i& coord, size_t lru) : lru(lru), coord(coord) {}

public:
  size_t lru;             ///< Last used time
  Eigen::Vector3i coord;  ///< Voxel coordinate
};

/// @brief Incremental voxelmap.
///        This class supports incremental point cloud insertion and LRU-based voxel deletion.
/// @note  This class can be used as a point cloud as well as a neighbor search.
template <typename VoxelContents>
struct IncrementalVoxelMap {
public:
  using Ptr = std::shared_ptr<IncrementalVoxelMap>;
  using ConstPtr = std::shared_ptr<const IncrementalVoxelMap>;

  /// @brief Constructor
  /// @param leaf_size  Voxel size
  explicit IncrementalVoxelMap(double leaf_size) : inv_leaf_size(1.0 / leaf_size), lru_horizon(100), lru_clear_cycle(10), lru_counter(0) {}

  /// @brief Number of points in the voxelmap
  size_t size() const { return flat_voxels.size(); }

  /// @brief Insert points to the voxelmap
  /// @param points Point cloud
  /// @param T      Transformation matrix
  template <typename PointCloud>
  void insert(const PointCloud& points, const Eigen::Isometry3d& T = Eigen::Isometry3d::Identity()) {
    // Insert points to the voxelmap
    for (size_t i = 0; i < traits::size(points); i++) {
      const Eigen::Vector4d pt = T * traits::point(points, i);
      const Eigen::Vector3i coord = fast_floor(pt * inv_leaf_size).template head<3>();

      auto found = voxels.find(coord);
      if (found == voxels.end()) {
        auto voxel = std::make_shared<std::pair<VoxelInfo, VoxelContents>>(VoxelInfo(coord, lru_counter), VoxelContents());

        found = voxels.emplace_hint(found, coord, flat_voxels.size());
        flat_voxels.emplace_back(voxel);
      }

      auto& [info, voxel] = *flat_voxels[found->second];
      info.lru = lru_counter;
      voxel.add(voxel_setting, pt, points, i, T);
    }

    if ((++lru_counter) % lru_clear_cycle == 0) {
      // Remove least recently used voxels
      auto remove_counter = std::remove_if(flat_voxels.begin(), flat_voxels.end(), [&](const std::shared_ptr<std::pair<VoxelInfo, VoxelContents>>& voxel) {
        return voxel->first.lru + lru_horizon < lru_counter;
      });
      flat_voxels.erase(remove_counter, flat_voxels.end());

      // Rehash
      voxels.clear();
      for (size_t i = 0; i < flat_voxels.size(); i++) {
        voxels[flat_voxels[i]->first.coord] = i;
      }
    }

    // Finalize voxel means and covs
    for (auto& voxel : flat_voxels) {
      voxel->second.finalize();
    }
  }

  /// @brief Find the nearest neighbor
  /// @param pt       Query point
  /// @param index    Index of the nearest neighbor
  /// @param sq_dist  Squared distance to the nearest neighbor
  /// @return         Number of found points (0 or 1)
  size_t nearest_neighbor_search(const Eigen::Vector4d& pt, size_t* index, double* sq_dist) const {
    const Eigen::Vector3i coord = fast_floor(pt * inv_leaf_size).template head<3>();
    const auto found = voxels.find(coord);
    if (found == voxels.end()) {
      return 0;
    }

    const size_t voxel_index = found->second;
    const auto& voxel = flat_voxels[voxel_index]->second;

    size_t point_index;
    if (traits::nearest_neighbor_search(voxel, pt, &point_index, sq_dist) == 0) {
      return 0;
    }

    *index = calc_index(voxel_index, point_index);
    return 1;
  }

  /// @brief Find k nearest neighbors
  /// @param pt          Query point
  /// @param k           Number of neighbors
  /// @param k_indices   Indices of nearest neighbors
  /// @param k_sq_dists  Squared distances to nearest neighbors
  /// @return            Number of found points
  size_t knn_search(const Eigen::Vector4d& pt, size_t k, size_t* k_indices, double* k_sq_dists) const {
    const Eigen::Vector3i coord = fast_floor(pt * inv_leaf_size).template head<3>();
    const auto found = voxels.find(coord);
    if (found == voxels.end()) {
      return 0;
    }

    const size_t voxel_index = found->second;
    const auto& voxel = flat_voxels[voxel_index]->second;

    std::vector<size_t> point_indices(k);
    std::vector<double> sq_dists(k);
    const size_t num_found = traits::knn_search(voxel, pt, k, point_indices.data(), sq_dists.data());

    for (size_t i = 0; i < num_found; i++) {
      k_indices[i] = calc_index(voxel_index, point_indices[i]);
      k_sq_dists[i] = sq_dists[i];
    }
    return num_found;
  }

  inline size_t calc_index(const size_t voxel_id, const size_t point_id) const { return (voxel_id << point_id_bits) | point_id; }
  inline size_t voxel_id(const size_t i) const { return i >> point_id_bits; }                ///< Extract the point ID from an index
  inline size_t point_id(const size_t i) const { return i & ((1ull << point_id_bits) - 1); }  ///< Extract the voxel ID from an index

public:
  static_assert(sizeof(size_t) == 8, "size_t must be 64-bit");
  static constexpr int point_id_bits = 32;                  ///< Use the first 32 bits for point id
  static constexpr int voxel_id_bits = 64 - point_id_bits;  ///< Use the remaining bits for voxel id
  const double inv_leaf_size;                               ///< Inverse of the voxel size

  size_t lru_horizon;      ///< LRU horizon size
  size_t lru_clear_cycle;  ///< LRU clear cycle
  size_t lru_counter;      ///< LRU counter

  typename VoxelContents::Setting voxel_setting;                                  ///< Voxel setting
  std::vector<std::shared_ptr<std::pair<VoxelInfo, VoxelContents>>> flat_voxels;  ///< Voxel contents
  std::unordered_map<Eigen::Vector3i, size_t, XORVector3iHash> voxels;            ///< Voxel index map
};

namespace traits {

template <typename VoxelContents>
struct Traits<IncrementalVoxelMap<VoxelContents>> {
  static size_t size(const IncrementalVoxelMap<VoxelContents>& voxelmap) { return voxelmap.size(); }

  static Eigen::Vector4d point(const IncrementalVoxelMap<VoxelContents>& voxelmap, size_t i) {
    const size_t voxel_id = voxelmap.voxel_id(i);
    const size_t point_id = voxelmap.point_id(i);
    const auto& voxel = voxelmap.flat_voxels[voxel_id]->second;
    return traits::point(voxel, point_id);
  }
  static Eigen::Vector4d normal(const IncrementalVoxelMap<VoxelContents>& voxelmap, size_t i) {
    const size_t voxel_id = voxelmap.voxel_id(i);
    const size_t point_id = voxelmap.point_id(i);
    const auto& voxel = voxelmap.flat_voxels[voxel_id]->second;
    return traits::normal(voxel, point_id);
  }
  static Eigen::Matrix4d cov(const IncrementalVoxelMap<VoxelContents>& voxelmap, size_t i) {
    const size_t voxel_id = voxelmap.voxel_id(i);
    const size_t point_id = voxelmap.point_id(i);
    const auto& voxel = voxelmap.flat_voxels[voxel_id]->second;
    return traits::cov(voxel, point_id);
  }

  static size_t nearest_neighbor_search(const IncrementalVoxelMap<VoxelContents>& voxelmap, const Eigen::Vector4d& pt, size_t* k_index, double* k_sq_dist) {
    return voxelmap.nearest_neighbor_search(pt, k_index, k_sq_dist);
  }

  static size_t knn_search(const IncrementalVoxelMap<VoxelContents>& voxelmap, const Eigen::Vector4d& pt, int k, size_t* k_index, double* k_sq_dist) {
    return voxelmap.knn_search(pt, k, k_index, k_sq_dist);
  }
};

template <typename VoxelContents>
std::vector<size_t> point_indices(const IncrementalVoxelMap<VoxelContents>& voxelmap) {
  std::vector<size_t> indices;
  indices.reserve(voxelmap.size() * 5);

  for (size_t voxel_id = 0; voxel_id < voxelmap.flat_voxels.size(); voxel_id++) {
    const auto& voxel = voxelmap.flat_voxels[voxel_id];
    for (size_t point_id = 0; point_id < traits::size(voxel->second); point_id++) {
      indices.emplace_back(voxelmap.calc_index(voxel_id, point_id));
    }
  }

  return indices;
}

template <typename VoxelContents>
std::vector<Eigen::Vector4d> voxel_points(const IncrementalVoxelMap<VoxelContents>& voxelmap) {
  std::vector<Eigen::Vector4d> points;
  points.reserve(voxelmap.size() * 5);

  for (const auto& voxel : voxelmap.flat_voxels) {
    for (size_t i = 0; i < traits::size(voxel->second); i++) {
      points.push_back(traits::point(voxel->second, i));
    }
  }
  return points;
}

template <typename VoxelContents>
std::vector<Eigen::Vector4d> voxel_normals(const IncrementalVoxelMap<VoxelContents>& voxelmap) {
  std::vector<Eigen::Vector4d> normals;
  normals.reserve(voxelmap.size() * 5);

  for (const auto& voxel : voxelmap.flat_voxels) {
    for (size_t i = 0; i < traits::size(voxel->second); i++) {
      normals.push_back(traits::normal(voxel->second, i));
    }
  }
  return normals;
}

template <typename VoxelContents>
std::vector<Eigen::Matrix4d> voxel_covs(const IncrementalVoxelMap<VoxelContents>& voxelmap) {
  std::vector<Eigen::Matrix4d> covs;
  covs.reserve(voxelmap.size() * 5);

  for (const auto& voxel : voxelmap.flat_voxels) {
    for (size_t i = 0; i < traits::size(voxel->second); i++) {
      covs.push_back(traits::cov(voxel->second, i));
    }
  }
  return covs;
}

}  // namespace traits

}  // namespace small_gicp
