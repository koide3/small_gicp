// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT

/// @brief Basic point cloud registration example with PCL interfaces
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/benchmark/read_points.hpp>

using namespace small_gicp;

/// @brief Example of using RegistrationPCL that can be used as a drop-in replacement for pcl::GeneralizedIterativeClosestPoint.
void example1(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& raw_target, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& raw_source) {
  // small_gicp::voxelgrid_downsampling can directly operate on pcl::PointCloud.
  pcl::PointCloud<pcl::PointXYZ>::Ptr target = voxelgrid_sampling_omp(*raw_target, 0.25);
  pcl::PointCloud<pcl::PointXYZ>::Ptr source = voxelgrid_sampling_omp(*raw_source, 0.25);

  // RegistrationPCL is derived from pcl::Registration and has mostly the same interface as pcl::GeneralizedIterativeClosestPoint.
  RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ> reg;
  reg.setNumThreads(4);
  reg.setCorrespondenceRandomness(20);
  reg.setMaxCorrespondenceDistance(1.0);
  reg.setVoxelResolution(1.0);
  reg.setRegistrationType("VGICP");  // or "GICP" (default = "GICP")

  // Set input point clouds.
  reg.setInputTarget(target);
  reg.setInputSource(source);

  // Align point clouds.
  auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  reg.align(*aligned);

  std::cout << "--- T_target_source ---" << std::endl << reg.getFinalTransformation() << std::endl;
  std::cout << "--- H ---" << std::endl << reg.getFinalHessian() << std::endl;

  // Swap source and target and align again.
  // This is useful when you want to re-use preprocessed point clouds for successive registrations (e.g., odometry estimation).
  reg.swapSourceAndTarget();
  reg.align(*aligned);

  std::cout << "--- T_target_source ---" << std::endl << reg.getFinalTransformation().inverse() << std::endl;
}

/// @brief Example to directly feed pcl::PointCloud<pcl::PointCovariance> to small_gicp::Registration.
void example2(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& raw_target, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& raw_source) {
  // Downsample points and convert them into pcl::PointCloud<pcl::PointCovariance>.
  pcl::PointCloud<pcl::PointCovariance>::Ptr target = voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZ>, pcl::PointCloud<pcl::PointCovariance>>(*raw_target, 0.25);
  pcl::PointCloud<pcl::PointCovariance>::Ptr source = voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZ>, pcl::PointCloud<pcl::PointCovariance>>(*raw_source, 0.25);

  // Estimate covariances of points.
  const int num_threads = 4;
  const int num_neighbors = 20;
  estimate_covariances_omp(*target, num_neighbors, num_threads);
  estimate_covariances_omp(*source, num_neighbors, num_threads);

  // Create KdTree for target and source.
  auto target_tree = std::make_shared<KdTree<pcl::PointCloud<pcl::PointCovariance>>>(target, KdTreeBuilderOMP(num_threads));
  auto source_tree = std::make_shared<KdTree<pcl::PointCloud<pcl::PointCovariance>>>(source, KdTreeBuilderOMP(num_threads));

  Registration<GICPFactor, ParallelReductionOMP> registration;
  registration.reduction.num_threads = num_threads;
  registration.rejector.max_dist_sq = 1.0;

  // Align point clouds. Note that the input point clouds are pcl::PointCloud<pcl::PointCovariance>.
  auto result = registration.align(*target, *source, *target_tree, Eigen::Isometry3d::Identity());

  std::cout << "--- T_target_source ---" << std::endl << result.T_target_source.matrix() << std::endl;
  std::cout << "converged:" << result.converged << std::endl;
  std::cout << "error:" << result.error << std::endl;
  std::cout << "iterations:" << result.iterations << std::endl;
  std::cout << "num_inliers:" << result.num_inliers << std::endl;
  std::cout << "--- H ---" << std::endl << result.H << std::endl;
  std::cout << "--- b ---" << std::endl << result.b.transpose() << std::endl;

  // Because this usage exposes all preprocessed data, you can easily re-use them to obtain the best efficiency.
  auto result2 = registration.align(*source, *target, *source_tree, Eigen::Isometry3d::Identity());

  std::cout << "--- T_target_source ---" << std::endl << result2.T_target_source.inverse().matrix() << std::endl;
}

int main(int argc, char** argv) {
  std::vector<Eigen::Vector4f> target_points = read_ply("data/target.ply");
  std::vector<Eigen::Vector4f> source_points = read_ply("data/source.ply");
  if (target_points.empty() || source_points.empty()) {
    std::cerr << "error: failed to read points from data/(target|source).ply" << std::endl;
    return 1;
  }

  const auto convert_to_pcl = [](const std::vector<Eigen::Vector4f>& raw_points) {
    auto points = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    points->resize(raw_points.size());
    for (size_t i = 0; i < raw_points.size(); i++) {
      points->at(i).getVector4fMap() = raw_points[i];
    }
    return points;
  };

  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_target = convert_to_pcl(target_points);
  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_source = convert_to_pcl(source_points);

  example1(raw_target, raw_source);
  example2(raw_target, raw_source);

  return 0;
}