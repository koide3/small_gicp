// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <tbb/tbb.h>

#include <memory>
#include <iostream>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/util/fast_floor.hpp>
#include <small_gicp/util/vector3i_hash.hpp>

namespace small_gicp {

struct FlatVoxelInfo {
public:
  FlatVoxelInfo() : coord(0, 0, 0), num_indices(-1), index_loc(0) {}

  Eigen::Vector3i coord;
  int num_indices;
  size_t index_loc;
};

struct IndexDistance {
  bool operator<(const IndexDistance& rhs) const { return distance < rhs.distance; }

  size_t index;
  double distance;
};

template <typename PointCloud>
struct FlatVoxelMap {
public:
  using Ptr = std::shared_ptr<FlatVoxelMap>;
  using ConstPtr = std::shared_ptr<const FlatVoxelMap>;

  FlatVoxelMap(const std::shared_ptr<const PointCloud>& points, double leaf_size) : inv_leaf_size(1.0 / leaf_size), seek_count(2), points(points) {
    set_offset_pattern(7);
    create_table(*points);
  }
  ~FlatVoxelMap() {}

  void set_offset_pattern(int num_offsets) {
    offsets.clear();
    switch (num_offsets) {
      default:
        std::cerr << "warning: num_offsets must be 1, 7, or 9 (num_offsets=" << num_offsets << ")" << std::endl;
      case 1:
        offsets = {Eigen::Vector3i(0, 0, 0)};
        return;
      case 7:
        offsets = {
          Eigen::Vector3i(0, 0, 0),
          Eigen::Vector3i(-1, 0, 0),
          Eigen::Vector3i(0, -1, 0),
          Eigen::Vector3i(0, 0, -1),
          Eigen::Vector3i(1, 0, 0),
          Eigen::Vector3i(0, 1, 0),
          Eigen::Vector3i(0, 0, 1)};
        return;
      case 27:
        for (int i = -1; i <= 1; i++) {
          for (int j = -1; j <= 1; j++) {
            for (int k = -1; k <= 1; k++) {
              offsets.emplace_back(i, j, k);
            }
          }
        }
        return;
    }
  }

  size_t knn_search(const Eigen::Vector4d& pt, size_t k, size_t* k_indices, double* k_sq_dists) const {
    const auto find_voxel = [&](const Eigen::Vector3i& coord) -> const FlatVoxelInfo* {
      const size_t hash = XORVector3iHash::hash(coord);
      for (size_t bucket_index = hash; bucket_index < hash + seek_count; bucket_index++) {
        const auto& voxel = voxels[bucket_index % voxels.size()];

        if (voxel.num_indices < 0) {
          return nullptr;
        } else if (voxel.coord == coord) {
          return &voxel;
        }
      }

      return nullptr;
    };

    std::vector<IndexDistance> v;
    v.reserve(k);
    std::priority_queue<IndexDistance> queue(std::less<IndexDistance>(), std::move(v));

    const Eigen::Vector3i center_coord = fast_floor(pt * inv_leaf_size).head<3>();
    for (const auto& offset : offsets) {
      const Eigen::Vector3i coord = center_coord + offset;
      const auto voxel = find_voxel(coord);
      if (voxel == nullptr || voxel->num_indices < 0) {
        continue;
      }

      const auto index_begin = indices.data() + voxel->index_loc;
      for (auto index_itr = index_begin; index_itr != index_begin + voxel->num_indices; index_itr++) {
        const double sq_dist = (traits::point(*points, *index_itr) - pt).squaredNorm();
        if (queue.size() < k) {
          queue.push(IndexDistance{*index_itr, sq_dist});
        } else if (sq_dist < queue.top().distance) {
          queue.pop();
          queue.push(IndexDistance{*index_itr, sq_dist});
        }
      }
    }

    const size_t n = queue.size();
    while (!queue.empty()) {
      const auto top = queue.top();
      queue.pop();

      k_indices[queue.size()] = top.index;
      k_sq_dists[queue.size()] = top.distance;
    }

    return n;
  }

private:
  void create_table(const PointCloud& points) {
    // Here, we assume that the data structure of std::atomic_int64_t is the same as that of std::int64_t.
    // This is a dangerous assumption. If C++20 is available, should use std::atomic_ref<std::int64_t> instead.
    static_assert(sizeof(std::atomic_int64_t) == sizeof(std::int64_t), "We assume that std::atomic_int64_t is the same as std::int64_t.");

    const double min_sq_dist_in_cell = 0.05 * 0.05;
    const int max_points_per_cell = 10;

    const size_t buckets_size = traits::size(points);
    std::vector<std::atomic_int64_t> assignment_table(max_points_per_cell * buckets_size);
    memset(assignment_table.data(), -1, sizeof(std::atomic_int64_t) * max_points_per_cell * buckets_size);

    std::vector<Eigen::Vector3i> coords(traits::size(points));
    tbb::parallel_for(static_cast<size_t>(0), static_cast<size_t>(traits::size(points)), [&](size_t i) {
      const Eigen::Vector4d pt = traits::point(points, i);
      const Eigen::Vector3i coord = fast_floor(pt * inv_leaf_size).template head<3>();
      coords[i] = coord;

      const size_t hash = XORVector3iHash::hash(coord);
      for (size_t bucket_index = hash; bucket_index < hash + seek_count; bucket_index++) {
        auto slot_begin = assignment_table.data() + (bucket_index % buckets_size) * max_points_per_cell;

        std::int64_t expected = -1;
        if (slot_begin->compare_exchange_strong(expected, static_cast<std::int64_t>(i))) {
          // Succeeded to insert the point in the first slot
          break;
        }

        if (coords[expected] != coord) {
          // If the bucket is already occupied with another voxel coord, try the next bucket
          continue;
        }

        const double sq_dist = (traits::point(points, expected) - pt).squaredNorm();
        if (sq_dist < min_sq_dist_in_cell) {
          break;
        }

        for (auto slot = slot_begin + 1; slot != slot_begin + max_points_per_cell; slot++) {
          std::int64_t expected = -1;
          if (slot->compare_exchange_strong(expected, static_cast<std::int64_t>(i))) {
            // Succeeded to insert the point
            break;
          }

          const double sq_dist = (traits::point(points, expected) - pt).squaredNorm();
          if (sq_dist < min_sq_dist_in_cell) {
            // There already exists a very close point
            break;
          }
        }
        break;
      }
    });

    indices.clear();
    indices.reserve(buckets_size * max_points_per_cell);

    voxels.clear();
    voxels.resize(buckets_size);
    for (size_t i = 0; i < buckets_size; i++) {
      const auto slot_begin = assignment_table.data() + max_points_per_cell * i;
      if (*slot_begin < 0) {
        continue;
      }

      FlatVoxelInfo v;
      v.coord = coords[*slot_begin];
      v.index_loc = indices.size();

      const auto slot_end = std::find(slot_begin, slot_begin + max_points_per_cell, -1);
      v.num_indices = std::distance(slot_begin, slot_end);
      std::copy(slot_begin, slot_end, std::back_inserter(indices));

      voxels[i] = v;
    }

    indices.shrink_to_fit();
  }

public:
  const double inv_leaf_size;
  const int seek_count;
  std::vector<Eigen::Vector3i> offsets;

  std::shared_ptr<const PointCloud> points;
  std::vector<FlatVoxelInfo> voxels;
  std::vector<size_t> indices;
};

namespace traits {

template <typename PointCloud>
struct Traits<FlatVoxelMap<PointCloud>> {
  static size_t knn_search(const FlatVoxelMap<PointCloud>& voxelmap, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
    return voxelmap.knn_search(point, k, k_indices, k_sq_dists);
  }
};

}  // namespace traits

}  // namespace small_gicp