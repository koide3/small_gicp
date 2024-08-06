// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <tbb/tbb.h>
#include <small_gicp/util/normal_estimation.hpp>

namespace small_gicp {

template <typename Setter, typename PointCloud, typename KdTree>
void estimate_local_features_tbb(PointCloud& cloud, KdTree& kdtree, int num_neighbors) {
  traits::resize(cloud, traits::size(cloud));
  tbb::parallel_for(tbb::blocked_range<size_t>(0, traits::size(cloud)), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i < range.end(); i++) {
      estimate_local_features<Setter>(cloud, kdtree, num_neighbors, i);
    }
  });
}

template <typename Setter, typename PointCloud>
void estimate_local_features_tbb(PointCloud& cloud, int num_neighbors) {
  traits::resize(cloud, traits::size(cloud));
  UnsafeKdTree<PointCloud> kdtree(cloud);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, traits::size(cloud)), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i < range.end(); i++) {
      estimate_local_features<Setter>(cloud, kdtree, num_neighbors, i);
    }
  });
}

/// @brief Estimate point normals with TBB
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param num_neighbors  Number of neighbors used for attribute estimation
template <typename PointCloud>
void estimate_normals_tbb(PointCloud& cloud, int num_neighbors = 20) {
  estimate_local_features_tbb<NormalSetter<PointCloud>>(cloud, num_neighbors);
}

/// @brief Estimate point normals with TBB
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param kdtree         Nearest neighbor search
/// @param num_neighbors  Number of neighbors used for attribute estimation
template <typename PointCloud, typename KdTree>
void estimate_normals_tbb(PointCloud& cloud, KdTree& kdtree, int num_neighbors = 20) {
  estimate_local_features_tbb<NormalSetter<PointCloud>>(cloud, kdtree, num_neighbors);
}

/// @brief Estimate point covariances with TBB
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param num_neighbors  Number of neighbors used for attribute estimation
template <typename PointCloud>
void estimate_covariances_tbb(PointCloud& cloud, int num_neighbors = 20) {
  estimate_local_features_tbb<CovarianceSetter<PointCloud>>(cloud, num_neighbors);
}

/// @brief Estimate point covariances with TBB
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param kdtree         Nearest neighbor search
/// @param num_neighbors  Number of neighbors used for attribute estimation
template <typename PointCloud, typename KdTree>
void estimate_covariances_tbb(PointCloud& cloud, KdTree& kdtree, int num_neighbors = 20) {
  estimate_local_features_tbb<CovarianceSetter<PointCloud>>(cloud, kdtree, num_neighbors);
}

/// @brief Estimate point normals and covariances with TBB
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param num_neighbors  Number of neighbors used for attribute estimation
template <typename PointCloud>
void estimate_normals_covariances_tbb(PointCloud& cloud, int num_neighbors = 20) {
  estimate_local_features_tbb<NormalCovarianceSetter<PointCloud>>(cloud, num_neighbors);
}

/// @brief Estimate point normals and covariances with TBB
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param kdtree         Nearest neighbor search
/// @param num_neighbors  Number of neighbors used for attribute estimation
template <typename PointCloud, typename KdTree>
void estimate_normals_covariances_tbb(PointCloud& cloud, KdTree& kdtree, int num_neighbors = 20) {
  estimate_local_features_tbb<NormalCovarianceSetter<PointCloud>>(cloud, kdtree, num_neighbors);
}

}  // namespace small_gicp
