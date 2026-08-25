// SPDX-FileCopyrightText: Copyright 2026 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <small_gicp/util/color_gradient.hpp>

namespace small_gicp {

/// @brief Estimate point color gradients with OpenMP
/// @note  Point normals and colors must be estimated (or set) in advance. If a sufficient number of neighbors is not found,
///        a zero vector is set to the point.
/// @param cloud          [in/out] Point cloud
/// @param kdtree         Nearest neighbor search
/// @param num_neighbors  Number of neighbors used for gradient estimation
/// @param num_threads    Number of threads
/// @param max_radius     Maximum neighbor distance (neighbors farther than this are ignored)
template <typename PointCloud, typename KdTree>
void estimate_color_gradients_omp(PointCloud& cloud, KdTree& kdtree, int num_neighbors = 30, int num_threads = 1, double max_radius = std::numeric_limits<double>::max()) {
  traits::resize(cloud, traits::size(cloud));
#pragma omp parallel for num_threads(num_threads)
  for (std::int64_t i = 0; i < traits::size(cloud); i++) {
    estimate_color_gradient(cloud, kdtree, num_neighbors, max_radius, i);
  }
}

/// @brief Estimate point color gradients with OpenMP
/// @note  Point normals and colors must be estimated (or set) in advance. If a sufficient number of neighbors is not found,
///        a zero vector is set to the point.
/// @param cloud          [in/out] Point cloud
/// @param num_neighbors  Number of neighbors used for gradient estimation
/// @param num_threads    Number of threads
/// @param max_radius     Maximum neighbor distance (neighbors farther than this are ignored)
template <typename PointCloud>
void estimate_color_gradients_omp(PointCloud& cloud, int num_neighbors = 30, int num_threads = 1, double max_radius = std::numeric_limits<double>::max()) {
  traits::resize(cloud, traits::size(cloud));
  UnsafeKdTree<PointCloud> kdtree(cloud);
#pragma omp parallel for num_threads(num_threads)
  for (std::int64_t i = 0; i < traits::size(cloud); i++) {
    estimate_color_gradient(cloud, kdtree, num_neighbors, max_radius, i);
  }
}

}  // namespace small_gicp
