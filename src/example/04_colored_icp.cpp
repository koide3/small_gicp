// SPDX-FileCopyrightText: Copyright 2026 Kenji Koide
// SPDX-License-Identifier: MIT

/// @brief Colored ICP registration example (Park et al., ICCV 2017)
#include <iostream>
#include <small_gicp/benchmark/read_points.hpp>

#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/factors/colored_icp_factor.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/util/color_gradient_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>

using namespace small_gicp;

/// @brief Assign linear color fields so that the example datasets, which have no color information, can be used with colored ICP.
/// @note  Colors are defined as a function of the position in the target frame so that the two clouds share a consistent color field under the ground truth alignment.
static void synthesize_colors(PointCloud& cloud, const Eigen::Isometry3d& T = Eigen::Isometry3d::Identity()) {
  for (size_t i = 0; i < cloud.size(); i++) {
    const Eigen::Vector3d pt = (T * cloud.point(i)).head<3>();
    cloud.color(i) = Eigen::Vector4d(0.5 + 0.3 * pt.x(), 0.2 * pt.y() + 0.1, 0.5 + 0.25 * pt.z(), 0.0);
  }
}

int main() {
  const int num_threads = 4;
  const double downsampling_resolution = 0.3;
  const double max_correspondence_distance = 1.0;

  // Load points and ground truth transformation
  const std::vector<Eigen::Vector4f> target_points = read_ply("data/target.ply");
  const std::vector<Eigen::Vector4f> source_points = read_ply("data/source.ply");

  Eigen::Isometry3d gt_T_target_source = Eigen::Isometry3d::Identity();
  {
    std::ifstream ifs("data/T_target_source.txt");
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        ifs >> gt_T_target_source.matrix()(i, j);
      }
    }
  }

  // Convert to small_gicp::PointCloud with colors
  auto target = std::make_shared<PointCloud>(target_points);
  auto source = std::make_shared<PointCloud>(source_points);
  synthesize_colors(*target);
  synthesize_colors(*source, gt_T_target_source);

  // Downsampling (colors are averaged in each voxel)
  target = voxelgrid_sampling_omp(*target, downsampling_resolution, num_threads);
  source = voxelgrid_sampling_omp(*source, downsampling_resolution, num_threads);

  // Normals and color gradients of the target point cloud
  auto target_tree = std::make_shared<KdTree<PointCloud>>(target, KdTreeBuilderOMP(num_threads));
  estimate_normals_omp(*target, *target_tree, 30, num_threads);
  estimate_color_gradients_omp(*target, *target_tree, 30, num_threads, 2.0 * downsampling_resolution);

  // Colored ICP
  Registration<ColoredICPFactor, ParallelReductionOMP> registration;
  registration.point_factor.lambda_geometric = 0.968;
  registration.reduction.num_threads = num_threads;
  registration.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;

  const auto result = registration.align(*target, *source, *target_tree, Eigen::Isometry3d::Identity());

  // Report the result
  const Eigen::Isometry3d error = gt_T_target_source.inverse() * result.T_target_source;
  std::cout << "converged: " << result.converged << std::endl;
  std::cout << "iterations: " << result.iterations << std::endl;
  std::endl(std::cout);
  std::cout << "T_target_source:\n" << result.T_target_source.matrix() << std::endl;
  std::cout << "error against ground truth (rot [deg], trans [m]): " << Eigen::AngleAxisd(error.linear()).angle() * 180.0 / M_PI << ", " << error.translation().norm() << std::endl;

  return 0;
}
