// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <small_gicp/registration/registration_helper.hpp>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/factors/plane_icp_factor.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>

namespace small_gicp {

// Preprocess points
std::pair<PointCloud::Ptr, std::shared_ptr<KdTree<PointCloud>>> preprocess_points(const PointCloud& points, double downsampling_resolution, int num_neighbors, int num_threads) {
  if (num_threads == 1) {
    auto downsampled = voxelgrid_sampling(points, downsampling_resolution);
    auto kdtree = std::make_shared<KdTree<PointCloud>>(downsampled);
    estimate_normals_covariances(*downsampled, *kdtree, num_neighbors);
    return {downsampled, kdtree};
  } else {
    auto downsampled = voxelgrid_sampling_omp(points, downsampling_resolution);
    auto kdtree = std::make_shared<KdTree<PointCloud>>(downsampled);
    estimate_normals_covariances_omp(*downsampled, *kdtree, num_neighbors, num_threads);
    return {downsampled, kdtree};
  }
}

// Preprocess points with Eigen input
template <typename T, int D>
std::pair<PointCloud::Ptr, std::shared_ptr<KdTree<PointCloud>>>
preprocess_points(const std::vector<Eigen::Matrix<T, D, 1>>& points, double downsampling_resolution, int num_neighbors, int num_threads) {
  return preprocess_points(*std::make_shared<PointCloud>(points), downsampling_resolution, num_neighbors, num_threads);
}

// Explicit instantiation
template std::pair<PointCloud::Ptr, std::shared_ptr<KdTree<PointCloud>>> preprocess_points(const std::vector<Eigen::Matrix<float, 3, 1>>&, double, int, int);
template std::pair<PointCloud::Ptr, std::shared_ptr<KdTree<PointCloud>>> preprocess_points(const std::vector<Eigen::Matrix<float, 4, 1>>&, double, int, int);
template std::pair<PointCloud::Ptr, std::shared_ptr<KdTree<PointCloud>>> preprocess_points(const std::vector<Eigen::Matrix<double, 3, 1>>&, double, int, int);
template std::pair<PointCloud::Ptr, std::shared_ptr<KdTree<PointCloud>>> preprocess_points(const std::vector<Eigen::Matrix<double, 4, 1>>&, double, int, int);

// Create Gaussian voxel map
GaussianVoxelMap::Ptr create_gaussian_voxelmap(const PointCloud& points, double voxel_resolution) {
  auto voxelmap = std::make_shared<GaussianVoxelMap>(voxel_resolution);
  voxelmap->insert(points);
  return voxelmap;
}

// Align point clouds with Eigen input
template <typename T, int D>
RegistrationResult
align(const std::vector<Eigen::Matrix<T, D, 1>>& target, const std::vector<Eigen::Matrix<T, D, 1>>& source, const Eigen::Isometry3d& init_T, const RegistrationSetting& setting) {
  auto [target_points, target_tree] = preprocess_points(*std::make_shared<PointCloud>(target), setting.downsampling_resolution, 10, setting.num_threads);
  auto [source_points, source_tree] = preprocess_points(*std::make_shared<PointCloud>(source), setting.downsampling_resolution, 10, setting.num_threads);

  if (setting.type == RegistrationSetting::VGICP) {
    auto target_voxelmap = create_gaussian_voxelmap(*target_points, setting.voxel_resolution);
    return align(*target_voxelmap, *source_points, init_T, setting);
  } else {
    return align(*target_points, *source_points, *target_tree, init_T, setting);
  }
}

template RegistrationResult
align(const std::vector<Eigen::Matrix<float, 3, 1>>&, const std::vector<Eigen::Matrix<float, 3, 1>>&, const Eigen::Isometry3d&, const RegistrationSetting&);
template RegistrationResult
align(const std::vector<Eigen::Matrix<float, 4, 1>>&, const std::vector<Eigen::Matrix<float, 4, 1>>&, const Eigen::Isometry3d&, const RegistrationSetting&);
template RegistrationResult
align(const std::vector<Eigen::Matrix<double, 3, 1>>&, const std::vector<Eigen::Matrix<double, 3, 1>>&, const Eigen::Isometry3d&, const RegistrationSetting&);
template RegistrationResult
align(const std::vector<Eigen::Matrix<double, 4, 1>>&, const std::vector<Eigen::Matrix<double, 4, 1>>&, const Eigen::Isometry3d&, const RegistrationSetting&);

// Align point clouds
RegistrationResult
align(const PointCloud& target, const PointCloud& source, const KdTree<PointCloud>& target_tree, const Eigen::Isometry3d& init_T, const RegistrationSetting& setting) {
  switch (setting.type) {
    default:
      std::cerr << "invalid registration type" << std::endl;
      abort();
    case RegistrationSetting::ICP: {
      Registration<ICPFactor, ParallelReductionOMP> registration;
      registration.reduction.num_threads = setting.num_threads;
      registration.rejector.max_dist_sq = setting.max_correspondence_distance * setting.max_correspondence_distance;
      registration.criteria.rotation_eps = setting.rotation_eps;
      registration.criteria.translation_eps = setting.translation_eps;
      return registration.align(target, source, target_tree, init_T);
    }
    case RegistrationSetting::PLANE_ICP: {
      Registration<PointToPlaneICPFactor, ParallelReductionOMP> registration;
      registration.reduction.num_threads = setting.num_threads;
      registration.rejector.max_dist_sq = setting.max_correspondence_distance * setting.max_correspondence_distance;
      registration.criteria.rotation_eps = setting.rotation_eps;
      registration.criteria.translation_eps = setting.translation_eps;
      return registration.align(target, source, target_tree, init_T);
    }
    case RegistrationSetting::GICP: {
      Registration<GICPFactor, ParallelReductionOMP> registration;
      registration.reduction.num_threads = setting.num_threads;
      registration.rejector.max_dist_sq = setting.max_correspondence_distance * setting.max_correspondence_distance;
      registration.criteria.rotation_eps = setting.rotation_eps;
      registration.criteria.translation_eps = setting.translation_eps;
      return registration.align(target, source, target_tree, init_T);
    }
    case RegistrationSetting::VGICP: {
      std::cerr << "error: use align(const GaussianVoxelMap&, const GaussianVoxelMap&, const Eigen::Isometry3d&, const RegistrationSetting&) for VGICP" << std::endl;
      return RegistrationResult(Eigen::Isometry3d::Identity());
    }
  }
}

// Align point clouds with VGICP
RegistrationResult align(const GaussianVoxelMap& target, const PointCloud& source, const Eigen::Isometry3d& init_T, const RegistrationSetting& setting) {
  if (setting.type != RegistrationSetting::VGICP) {
    std::cerr << "invalid registration type for GaussianVoxelMap" << std::endl;
  }

  Registration<GICPFactor, ParallelReductionOMP> registration;
  registration.reduction.num_threads = setting.num_threads;
  registration.criteria.rotation_eps = setting.rotation_eps;
  registration.criteria.translation_eps = setting.translation_eps;
  return registration.align(target, source, target, init_T);
}

}  // namespace small_gicp
