// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <random>
#include <iostream>
#include <unordered_map>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/util/fast_floor.hpp>
#include <small_gicp/util/vector3i_hash.hpp>

namespace small_gicp {

/// @brief Voxelgrid downsampling. This function computes exact average of points in each voxel, and each voxel can contain arbitrary number of points.
/// @note  Discretized voxel coords must be in 21bit range [-1048576, 1048575].
///        For example, if the downsampling resolution is 0.01 m, point coordinates must be in [-10485.76, 10485.75] m.
///        Points outside the valid range will be ignored.
/// @param points     Input points
/// @param leaf_size  Downsampling resolution
/// @return           Downsampled points
template <typename InputPointCloud, typename OutputPointCloud = InputPointCloud>
std::shared_ptr<OutputPointCloud> voxelgrid_sampling(const InputPointCloud& points, double leaf_size) {
  if (traits::size(points) == 0) {
    return std::make_shared<OutputPointCloud>();
  }

  const double inv_leaf_size = 1.0 / leaf_size;

  constexpr std::uint64_t invalid_coord = std::numeric_limits<std::uint64_t>::max();
  constexpr int coord_bit_size = 21;                       // Bits to represent each voxel coordinate (pack 21x3=63bits in 64bit int)
  constexpr size_t coord_bit_mask = (1 << 21) - 1;         // Bit mask
  constexpr int coord_offset = 1 << (coord_bit_size - 1);  // Coordinate offset to make values positive

  std::vector<std::pair<std::uint64_t, size_t>> coord_pt(traits::size(points));
  for (size_t i = 0; i < traits::size(points); i++) {
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

  // Sort by voxel coord
  const auto compare = [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; };
  std::sort(coord_pt.begin(), coord_pt.end(), compare);

  auto downsampled = std::make_shared<OutputPointCloud>();
  traits::resize(*downsampled, traits::size(points));

  size_t num_points = 0;
  Eigen::Vector4d sum_pt = traits::point(points, coord_pt.front().second);
  for (size_t i = 1; i < traits::size(points); i++) {
    if (coord_pt[i].first == invalid_coord) {
      continue;
    }

    if (coord_pt[i - 1].first != coord_pt[i].first) {
      traits::set_point(*downsampled, num_points++, sum_pt / sum_pt.w());
      sum_pt.setZero();
    }

    sum_pt += traits::point(points, coord_pt[i].second);
  }

  traits::set_point(*downsampled, num_points++, sum_pt / sum_pt.w());
  traits::resize(*downsampled, num_points);

  return downsampled;
}

/// @brief Random downsampling.
/// @param points      Input points
/// @param num_samples Number of samples to be drawn
/// @return            Downsampled points
template <typename InputPointCloud, typename OutputPointCloud = InputPointCloud, typename RNG = std::mt19937>
std::shared_ptr<OutputPointCloud> random_sampling(const InputPointCloud& points, size_t num_samples, RNG& rng) {
  if (traits::size(points) == 0) {
    std::cerr << "warning: empty input points!!" << std::endl;
    return std::make_shared<OutputPointCloud>();
  }

  std::vector<size_t> indices(traits::size(points));
  std::iota(indices.begin(), indices.end(), 0);

  if (num_samples >= indices.size()) {
    std::cerr << "warning: num_samples >= points.size()!! (" << num_samples << " vs " << traits::size(points) << ")" << std::endl;
    num_samples = indices.size();
  }

  std::vector<size_t> samples(num_samples);
  std::sample(indices.begin(), indices.end(), samples.begin(), num_samples, rng);

  auto downsampled = std::make_shared<OutputPointCloud>();
  traits::resize(*downsampled, num_samples);

  for (size_t i = 0; i < num_samples; i++) {
    traits::set_point(*downsampled, i, traits::point(points, samples[i]));
  }

  return downsampled;
}

}  // namespace small_gicp
