// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <small_gicp/util/normal_estimation.hpp>

namespace small_gicp {

template <typename Setter, typename PointCloud>
void estimate_local_features_omp(PointCloud& cloud, int num_neighbors, int num_threads) {
  traits::resize(cloud, traits::size(cloud));
  UnsafeKdTree<PointCloud> kdtree(cloud);
#pragma omp parallel for num_threads(num_threads)
  for (size_t i = 0; i < traits::size(cloud); i++) {
    estimate_local_features<Setter>(cloud, kdtree, num_neighbors, i);
  }
}

template <typename Setter, typename PointCloud, typename KdTree>
void estimate_local_features_omp(PointCloud& cloud, KdTree& kdtree, int num_neighbors, int num_threads) {
  traits::resize(cloud, traits::size(cloud));
#pragma omp parallel for num_threads(num_threads)
  for (std::int64_t i = 0; i < traits::size(cloud); i++) {
    estimate_local_features<Setter>(cloud, kdtree, num_neighbors, i);
  }
}

/// @brief Estimate point normals with OpenMP
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param num_neighbors  Number of neighbors used for attribute estimation
/// @param num_threads    Number of threads
template <typename PointCloud>
void estimate_normals_omp(PointCloud& cloud, int num_neighbors = 20, int num_threads = 4) {
  estimate_local_features_omp<NormalSetter<PointCloud>>(cloud, num_neighbors, num_threads);
}

/// @brief Estimate point normals with OpenMP
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param kdtree         Nearest neighbor search
/// @param num_neighbors  Number of neighbors used for attribute estimation
/// @param num_threads    Number of threads
template <typename PointCloud, typename KdTree>
void estimate_normals_omp(PointCloud& cloud, KdTree& kdtree, int num_neighbors = 20, int num_threads = 4) {
  estimate_local_features_omp<NormalSetter<PointCloud>>(cloud, kdtree, num_neighbors, num_threads);
}

/// @brief Estimate point covariances with OpenMP
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param num_neighbors  Number of neighbors used for attribute estimation
/// @param num_threads    Number of threads
template <typename PointCloud>
void estimate_covariances_omp(PointCloud& cloud, int num_neighbors = 20, int num_threads = 4) {
  estimate_local_features_omp<CovarianceSetter<PointCloud>>(cloud, num_neighbors, num_threads);
}

/// @brief Estimate point covariances with OpenMP
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param kdtree         Nearest neighbor search
/// @param num_neighbors  Number of neighbors used for attribute estimation
/// @param num_threads    Number of threads
template <typename PointCloud, typename KdTree>
void estimate_covariances_omp(PointCloud& cloud, KdTree& kdtree, int num_neighbors = 20, int num_threads = 4) {
  estimate_local_features_omp<CovarianceSetter<PointCloud>>(cloud, kdtree, num_neighbors, num_threads);
}

/// @brief Estimate point normals and covariances with OpenMP
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param num_neighbors  Number of neighbors used for attribute estimation
/// @param num_threads    Number of threads
template <typename PointCloud>
void estimate_normals_covariances_omp(PointCloud& cloud, int num_neighbors = 20, int num_threads = 4) {
  estimate_local_features_omp<NormalCovarianceSetter<PointCloud>>(cloud, num_neighbors, num_threads);
}

/// @brief Estimate point normals and covariances with OpenMP
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param kdtree         Nearest neighbor search
/// @param num_neighbors  Number of neighbors used for attribute estimation
/// @param num_threads    Number of threads
template <typename PointCloud, typename KdTree>
void estimate_normals_covariances_omp(PointCloud& cloud, KdTree& kdtree, int num_neighbors = 20, int num_threads = 4) {
  estimate_local_features_omp<NormalCovarianceSetter<PointCloud>>(cloud, kdtree, num_neighbors, num_threads);
}

}  // namespace small_gicp
