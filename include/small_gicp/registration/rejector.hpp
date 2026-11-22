// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace small_gicp {

/// @brief Null correspondence rejector. This class accepts all input correspondences.
struct NullRejector {
  template <typename TargetPointCloud, typename SourcePointCloud>
  bool operator()(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T, size_t target_index, size_t source_index, double sq_dist) const {
    return false;
  }
};

/// @brief Rejecting correspondences with large distances.
struct DistanceRejector {
  DistanceRejector() : max_dist_sq(1.0) {}

  template <typename TargetPointCloud, typename SourcePointCloud>
  bool operator()(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T, size_t target_index, size_t source_index, double sq_dist) const {
    return sq_dist > max_dist_sq;
  }

  /// @brief Maximum squared distance between corresponding points.
  /// @note  This value is also passed to the nearest neighbor search as a search radius, which allows the
  ///        search structure to skip subtrees that cannot contain a valid correspondence.
  double max_dist_sq;
};

}  // namespace small_gicp
