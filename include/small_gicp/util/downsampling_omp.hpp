// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <atomic>
#include <memory>
#include <iostream>

#include <small_gicp/points/traits.hpp>
#include <small_gicp/util/fast_floor.hpp>
#include <small_gicp/util/vector3i_hash.hpp>
#include <small_gicp/util/sort_omp.hpp>

namespace small_gicp {

/// @brief Voxel grid downsampling with OpenMP backend.
/// @note  This function has minor run-by-run non-deterministic behavior due to parallel data collection that results
///        in a deviation of the number of points in the downsampling results (up to 10% increase from the single-thread version).
/// @note  Discretized voxel coords must be in 21bit range [-1048576, 1048575].
///        For example, if the downsampling resolution is 0.01 m, point coordinates must be in [-10485.76, 10485.75] m.
///        Points outside the valid range will be ignored.
/// @param points     Input points
/// @param leaf_size  Downsampling resolution
/// @return           Downsampled points
template <typename InputPointCloud, typename OutputPointCloud = InputPointCloud>
std::shared_ptr<OutputPointCloud> voxelgrid_sampling_omp(const InputPointCloud& points, double leaf_size, int num_threads = 4) {
  if (traits::size(points) == 0) {
    return std::make_shared<OutputPointCloud>();
  }

  const double inv_leaf_size = 1.0 / leaf_size;

  constexpr std::uint64_t invalid_coord = std::numeric_limits<std::uint64_t>::max();
  constexpr int coord_bit_size = 21;                       // Bits to represent each voxel coordinate (pack 21x3 = 63bits in 64bit int)
  constexpr size_t coord_bit_mask = (1 << 21) - 1;         // Bit mask
  constexpr int coord_offset = 1 << (coord_bit_size - 1);  // Coordinate offset to make values positive

  std::vector<std::pair<std::uint64_t, size_t>> coord_pt(traits::size(points));
#pragma omp parallel for num_threads(num_threads) schedule(guided, 32)
  for (std::int64_t i = 0; i < traits::size(points); i++) {
    const Eigen::Array4i coord = fast_floor(traits::point(points, i) * inv_leaf_size) + coord_offset;
    if ((coord < 0).any() || (coord > coord_bit_mask).any()) {
      std::cerr << "warning: voxel coord is out of range!!" << std::endl;
      coord_pt[i] = {invalid_coord, i};
      continue;
    }
    // Compute voxel coord bits (0|1bit, z|21bit, y|21bit, x|21bit)
    const std::uint64_t bits =                                                           //
      (static_cast<std::uint64_t>(coord[0] & coord_bit_mask) << (coord_bit_size * 0)) |  //
      (static_cast<std::uint64_t>(coord[1] & coord_bit_mask) << (coord_bit_size * 1)) |  //
      (static_cast<std::uint64_t>(coord[2] & coord_bit_mask) << (coord_bit_size * 2));
    coord_pt[i] = {bits, i};
  }

  // Sort by voxel coords
  quick_sort_omp(coord_pt.begin(), coord_pt.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; }, num_threads);

  auto downsampled = std::make_shared<OutputPointCloud>();
  traits::resize(*downsampled, traits::size(points));

  // Take block-wise sum
  const int block_size = 1024;
  std::atomic_uint64_t num_points = 0;

#pragma omp parallel for num_threads(num_threads) schedule(guided, 4)
  for (std::int64_t block_begin = 0; block_begin < traits::size(points); block_begin += block_size) {
    std::vector<Eigen::Vector4d> sub_points;
    sub_points.reserve(block_size);

    const size_t block_end = std::min<size_t>(traits::size(points), block_begin + block_size);

    Eigen::Vector4d sum_pt = traits::point(points, coord_pt[block_begin].second);
    for (size_t i = block_begin + 1; i != block_end; i++) {
      if (coord_pt[i].first == invalid_coord) {
        continue;
      }

      if (coord_pt[i - 1].first != coord_pt[i].first) {
        sub_points.emplace_back(sum_pt / sum_pt.w());
        sum_pt.setZero();
      }
      sum_pt += traits::point(points, coord_pt[i].second);
    }
    sub_points.emplace_back(sum_pt / sum_pt.w());

    const size_t point_index_begin = num_points.fetch_add(sub_points.size());
    for (size_t i = 0; i < sub_points.size(); i++) {
      traits::set_point(*downsampled, point_index_begin + i, sub_points[i]);
    }
  }

  traits::resize(*downsampled, num_points);

  return downsampled;
}

}  // namespace small_gicp
