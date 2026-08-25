// SPDX-FileCopyrightText: Copyright 2026 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <limits>
#include <Eigen/Eigen>
#include <small_gicp/ann/kdtree.hpp>

namespace small_gicp {

/**
 * @brief Estimate the color gradient of a single point.
 *        The intensity of a point is defined as the mean of its color channels, and the gradient of the intensity field
 *        over the local tangent plane is estimated by a least-squares fit with the neighborhood points, being equivalent
 *        to Open3D's ColoredICP color gradient estimation (J. Park et al., "Colored Point Cloud Registration Revisited", ICCV 2017).
 * @note  Point normals and colors must be estimated (or set) in advance. If a sufficient number of neighbors is not found,
 *        a zero vector is set to the point.
 * @param cloud        [in/out] Point cloud
 * @param kdtree       Nearest neighbor search
 * @param num_neighbors  Number of neighbors used for gradient estimation
 * @param max_radius   Maximum neighbor distance (neighbors farther than this are ignored; hybrid knn + radius search)
 * @param point_index  Target point index
 */
template <typename PointCloud, typename KdTree>
void estimate_color_gradient(PointCloud& cloud, const KdTree& kdtree, int num_neighbors, double max_radius, size_t point_index) {
  std::vector<size_t> k_indices(num_neighbors);
  std::vector<double> k_sq_dists(num_neighbors);
  const size_t n = kdtree.knn_search(traits::point(cloud, point_index), num_neighbors, k_indices.data(), k_sq_dists.data());

  const double max_sq_dist = max_radius * max_radius;
  const Eigen::Vector3d normal = traits::normal(cloud, point_index).template head<3>();
  const auto& color_i = traits::color(cloud, point_index);
  const double intensity_i = (color_i[0] + color_i[1] + color_i[2]) / 3.0;

  // k_indices[0] is the query point itself. Accumulate the normal equations of the intensity field over the neighbors within max_radius
  Eigen::Matrix3d ATA = Eigen::Matrix3d::Zero();
  Eigen::Vector3d ATb = Eigen::Vector3d::Zero();
  int num_valid_neighbors = 0;
  for (size_t j = 1; j < n; j++) {
    if (k_sq_dists[j] > max_sq_dist) {
      continue;
    }

    const Eigen::Vector3d d = (traits::point(cloud, k_indices[j]) - traits::point(cloud, point_index)).template head<3>();
    const auto& color_j = traits::color(cloud, k_indices[j]);
    const double delta_intensity = (color_j[0] + color_j[1] + color_j[2]) / 3.0 - intensity_i;

    ATA += d * d.transpose();
    ATb += d * delta_intensity;
    num_valid_neighbors++;
  }

  if (num_valid_neighbors < 3) {
    // Insufficient number of neighbors
    traits::set_color_grad(cloud, point_index, Eigen::Vector4d::Zero());
    return;
  }

  // Regularize the normal equation so that the gradient is constrained on the tangent plane
  ATA += num_valid_neighbors * num_valid_neighbors * normal * normal.transpose();

  const Eigen::Vector3d gradient = ATA.ldlt().solve(ATb);
  if (!gradient.allFinite()) {
    traits::set_color_grad(cloud, point_index, Eigen::Vector4d::Zero());
    return;
  }

  traits::set_color_grad(cloud, point_index, (Eigen::Vector4d() << gradient, 0.0).finished());
}

/**
 * @brief Estimate point color gradients.
 * @note  Point normals and colors must be estimated (or set) in advance. If a sufficient number of neighbors is not found,
 *        a zero vector is set to the point.
 * @note  To reproduce Open3D's ColoredICP behavior, set max_radius to about twice the maximum correspondence distance
 *        (Open3D internally uses a hybrid search with radius = 2 * max_correspondence_distance and max_nn = 30).
 * @param cloud          [in/out] Point cloud
 * @param kdtree         Nearest neighbor search
 * @param num_neighbors  Number of neighbors used for gradient estimation (default=30 following Open3D's ColoredICP)
 * @param max_radius     Maximum neighbor distance (neighbors farther than this are ignored)
 */
template <typename PointCloud, typename KdTree>
void estimate_color_gradients(PointCloud& cloud, const KdTree& kdtree, int num_neighbors = 30, double max_radius = std::numeric_limits<double>::max()) {
  traits::resize(cloud, traits::size(cloud));
  for (size_t i = 0; i < traits::size(cloud); i++) {
    estimate_color_gradient(cloud, kdtree, num_neighbors, max_radius, i);
  }
}

/**
 * @brief Estimate point color gradients.
 * @note  Point normals and colors must be estimated (or set) in advance. If a sufficient number of neighbors is not found,
 *        a zero vector is set to the point.
 * @param cloud          [in/out] Point cloud
 * @param num_neighbors  Number of neighbors used for gradient estimation (default=30 following Open3D's ColoredICP)
 * @param max_radius     Maximum neighbor distance (neighbors farther than this are ignored)
 */
template <typename PointCloud>
void estimate_color_gradients(PointCloud& cloud, int num_neighbors = 30, double max_radius = std::numeric_limits<double>::max()) {
  traits::resize(cloud, traits::size(cloud));

  UnsafeKdTree<PointCloud> kdtree(cloud);
  for (size_t i = 0; i < traits::size(cloud); i++) {
    estimate_color_gradient(cloud, kdtree, num_neighbors, max_radius, i);
  }
}

}  // namespace small_gicp
