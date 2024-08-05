// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/registration_result.hpp>

namespace small_gicp {

/// @brief Preprocess point cloud (downsampling, kdtree creation, and normal and covariance estimation).
/// @note  When num_threads >= 2, this function has minor run-by-run non-determinism due to the parallel downsampling.
/// @see   small_gicp::voxelgrid_sampling_omp, small_gicp::estimate_normals_covariances_omp
/// @param points                Input points
/// @param downsampling_resolution Downsample resolution
/// @param num_neighbors         Number of neighbors for normal/covariance estimation
/// @param num_threads           Number of threads
std::pair<PointCloud::Ptr, std::shared_ptr<KdTree<PointCloud>>>
preprocess_points(const PointCloud& points, double downsampling_resolution, int num_neighbors = 10, int num_threads = 4);

/// @brief Preprocess point cloud (downsampling, kdtree creation, and normal and covariance estimation)
/// @note  This function only accepts Eigen::Vector(3|4)(f|d) as input
/// @note  When num_threads >= 2, this function has minor run-by-run non-determinism due to the parallel downsampling.
/// @see   small_gicp::voxelgrid_sampling_omp, small_gicp::estimate_normals_covariances_omp
template <typename T, int D>
std::pair<PointCloud::Ptr, std::shared_ptr<KdTree<PointCloud>>>
preprocess_points(const std::vector<Eigen::Matrix<T, D, 1>>& points, double downsampling_resolution, int num_neighbors = 10, int num_threads = 4);

/// @brief Create an incremental Gaussian voxel map.
/// @see   small_gicp::IncrementalVoxelMap
/// @param points            Input points
/// @param voxel_resolution  Voxel resolution
GaussianVoxelMap::Ptr create_gaussian_voxelmap(const PointCloud& points, double voxel_resolution);

/// @brief Registration setting
struct RegistrationSetting {
  enum RegistrationType { ICP, PLANE_ICP, GICP, VGICP };

  RegistrationType type = GICP;              ///< Registration type
  double voxel_resolution = 1.0;             ///< Voxel resolution for VGICP
  double downsampling_resolution = 0.25;     ///< Downsample resolution (this will be used only in the Eigen-based interface)
  double max_correspondence_distance = 1.0;  ///< Maximum correspondence distance
  double rotation_eps = 0.1 * M_PI / 180.0;  ///< Rotation tolerance for convergence check [rad]
  double translation_eps = 1e-3;             ///< Translation tolerance for convergence check
  int num_threads = 4;                       ///< Number of threads
  int max_iterations = 20;                   ///< Maximum number of iterations
  bool verbose = false;                      ///< Verbose mode
};

/// @brief Align point clouds
/// @note This function only accepts Eigen::Vector(3|4)(f|d) as input
/// @see  small_gicp::voxelgrid_sampling_omp, small_gicp::estimate_normals_covariances_omp
/// @param target     Target points
/// @param source     Source points
/// @param init_T     Initial guess of T_target_source
/// @param setting    Registration setting
/// @return           Registration result
template <typename T, int D>
RegistrationResult align(
  const std::vector<Eigen::Matrix<T, D, 1>>& target,
  const std::vector<Eigen::Matrix<T, D, 1>>& source,
  const Eigen::Isometry3d& init_T = Eigen::Isometry3d::Identity(),
  const RegistrationSetting& setting = RegistrationSetting());

/// @brief Align preprocessed point clouds
/// @param target       Target point cloud
/// @param source       Source point cloud
/// @param target_tree  Nearest neighbor search for the target point cloud
/// @param init_T       Initial guess of T_target_source
/// @param setting      Registration setting
/// @return             Registration result
RegistrationResult align(
  const PointCloud& target,
  const PointCloud& source,
  const KdTree<PointCloud>& target_tree,
  const Eigen::Isometry3d& init_T = Eigen::Isometry3d::Identity(),
  const RegistrationSetting& setting = RegistrationSetting());

/// @brief Align preprocessed point clouds with VGICP
/// @param target       Target Gaussian voxelmap
/// @param source       Source point cloud
/// @param init_T       Initial guess of T_target_source
/// @param setting      Registration setting
/// @return             Registration result
RegistrationResult align(
  const GaussianVoxelMap& target,
  const PointCloud& source,
  const Eigen::Isometry3d& init_T = Eigen::Isometry3d::Identity(),
  const RegistrationSetting& setting = RegistrationSetting());

}  // namespace small_gicp
