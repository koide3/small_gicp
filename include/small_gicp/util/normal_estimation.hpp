// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Eigen>
#include <small_gicp/ann/kdtree.hpp>

namespace small_gicp {

/// @brief Computes point normals from eigenvectors and sets them to the point cloud.
/// @note  If a sufficient number of neighbor points is not found, a zero vector is set to the point.
template <typename PointCloud>
struct NormalSetter {
  /// @brief Handle invalid case (too few points).
  static void set_invalid(PointCloud& cloud, size_t i) { traits::set_normal(cloud, i, Eigen::Vector4d::Zero()); }

  /// @brief Compute and set the normal to the point cloud.
  static void set(PointCloud& cloud, size_t i, const Eigen::Matrix3d& eigenvectors) {
    const Eigen::Vector4d normal = (Eigen::Vector4d() << eigenvectors.col(0).normalized(), 0.0).finished();
    if (traits::point(cloud, i).dot(normal) > 0) {
      traits::set_normal(cloud, i, -normal);
    } else {
      traits::set_normal(cloud, i, normal);
    }
  }
};

/// @brief Computes point covariances from eigenvectors and sets them to the point cloud.
/// @note  If a sufficient number of neighbor points is not found, an identity matrix is set to the point.
template <typename PointCloud>
struct CovarianceSetter {
  /// @brief Handle invalid case (too few points).
  static void set_invalid(PointCloud& cloud, size_t i) {
    Eigen::Matrix4d cov = Eigen::Matrix4d::Identity();
    cov(3, 3) = 0.0;
    traits::set_cov(cloud, i, cov);
  }

  /// @brief Compute and set the covariance to the point cloud.
  static void set(PointCloud& cloud, size_t i, const Eigen::Matrix3d& eigenvectors) {
    const Eigen::Vector3d values(1e-3, 1.0, 1.0);
    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
    cov.block<3, 3>(0, 0) = eigenvectors * values.asDiagonal() * eigenvectors.transpose();
    traits::set_cov(cloud, i, cov);
  }
};

/// @brief Computes point normals and covariances from eigenvectors and sets them to the point cloud.
/// @note  If a sufficient number of neighbor points is not found, a zero vector and an identity matrix are set to the point.
template <typename PointCloud>
struct NormalCovarianceSetter {
  /// @brief Handle invalid case (too few points).
  static void set_invalid(PointCloud& cloud, size_t i) {
    NormalSetter<PointCloud>::set_invalid(cloud, i);
    CovarianceSetter<PointCloud>::set_invalid(cloud, i);
  }

  /// @brief Compute and set the normal and covariance to the point cloud.
  static void set(PointCloud& cloud, size_t i, const Eigen::Matrix3d& eigenvectors) {
    NormalSetter<PointCloud>::set(cloud, i, eigenvectors);
    CovarianceSetter<PointCloud>::set(cloud, i, eigenvectors);
  }
};

template <typename Setter, typename PointCloud, typename Tree>
void estimate_local_features(PointCloud& cloud, Tree& kdtree, int num_neighbors, size_t point_index) {
  std::vector<size_t> k_indices(num_neighbors);
  std::vector<double> k_sq_dists(num_neighbors);
  const size_t n = kdtree.knn_search(traits::point(cloud, point_index), num_neighbors, k_indices.data(), k_sq_dists.data());

  if (n < 5) {
    // Insufficient number of neighbors
    Setter::set_invalid(cloud, point_index);
    return;
  }

  Eigen::Vector4d sum_points = Eigen::Vector4d::Zero();
  Eigen::Matrix4d sum_cross = Eigen::Matrix4d::Zero();
  for (size_t i = 0; i < n; i++) {
    const auto& pt = traits::point(cloud, k_indices[i]);
    sum_points += pt;
    sum_cross += pt * pt.transpose();
  }

  const Eigen::Vector4d mean = sum_points / n;
  const Eigen::Matrix4d cov = (sum_cross - mean * sum_points.transpose()) / n;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
  eig.computeDirect(cov.block<3, 3>(0, 0));

  Setter::set(cloud, point_index, eig.eigenvectors());
}

template <typename Setter, typename PointCloud, typename KdTree>
void estimate_local_features(PointCloud& cloud, const KdTree& kdtree, int num_neighbors) {
  traits::resize(cloud, traits::size(cloud));
  for (size_t i = 0; i < traits::size(cloud); i++) {
    estimate_local_features<Setter>(cloud, kdtree, num_neighbors, i);
  }
}

template <typename Setter, typename PointCloud>
void estimate_local_features(PointCloud& cloud, int num_neighbors) {
  traits::resize(cloud, traits::size(cloud));

  UnsafeKdTree<PointCloud> kdtree(cloud);
  for (size_t i = 0; i < traits::size(cloud); i++) {
    estimate_local_features<Setter>(cloud, kdtree, num_neighbors, i);
  }
}

/// @brief Estimate point normals
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param num_neighbors  Number of neighbors used for attribute estimation
template <typename PointCloud>
void estimate_normals(PointCloud& cloud, int num_neighbors = 20) {
  estimate_local_features<NormalSetter<PointCloud>>(cloud, num_neighbors);
}

/// @brief Estimate point normals
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param kdtree         Nearest neighbor search
/// @param num_neighbors  Number of neighbors used for attribute estimation
template <typename PointCloud, typename KdTree>
void estimate_normals(PointCloud& cloud, KdTree& kdtree, int num_neighbors = 20) {
  estimate_local_features<NormalSetter<PointCloud>>(cloud, kdtree, num_neighbors);
}

/// @brief Estimate point covariances
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param num_neighbors  Number of neighbors used for attribute estimation
template <typename PointCloud>
void estimate_covariances(PointCloud& cloud, int num_neighbors = 20) {
  estimate_local_features<CovarianceSetter<PointCloud>>(cloud, num_neighbors);
}

/// @brief Estimate point covariances
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param kdtree         Nearest neighbor search
/// @param num_neighbors  Number of neighbors used for attribute estimation
template <typename PointCloud, typename KdTree>
void estimate_covariances(PointCloud& cloud, KdTree& kdtree, int num_neighbors = 20) {
  estimate_local_features<CovarianceSetter<PointCloud>>(cloud, kdtree, num_neighbors);
}

/// @brief Estimate point normals and covariances
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param num_neighbors  Number of neighbors used for attribute estimation
template <typename PointCloud>
void estimate_normals_covariances(PointCloud& cloud, int num_neighbors = 20) {
  estimate_local_features<NormalCovarianceSetter<PointCloud>>(cloud, num_neighbors);
}

/// @brief Estimate point normals and covariances
/// @note  If a sufficient number of neighbor points for normal/covariance estimation (5 points) is not found,
///        an invalid normal/covariance is set to the point (normal=zero vector, covariance=identity matrix).
/// @param cloud          [in/out] Point cloud
/// @param kdtree         Nearest neighbor search
/// @param num_neighbors  Number of neighbors used for attribute estimation
template <typename PointCloud, typename KdTree>
void estimate_normals_covariances(PointCloud& cloud, KdTree& kdtree, int num_neighbors = 20) {
  estimate_local_features<NormalCovarianceSetter<PointCloud>>(cloud, kdtree, num_neighbors);
}

}  // namespace small_gicp
