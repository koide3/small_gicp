// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT

/// @brief Basic point cloud registration example with small_gicp::align()
#include <iostream>
#include <small_gicp/benchmark/read_points.hpp>
#include <small_gicp/registration/registration_helper.hpp>

using namespace small_gicp;

/// @brief Most basic registration example.
void example1(const std::vector<Eigen::Vector4f>& target_points, const std::vector<Eigen::Vector4f>& source_points) {
  RegistrationSetting setting;
  setting.num_threads = 4;                    // Number of threads to be used
  setting.downsampling_resolution = 0.25;     // Downsampling resolution
  setting.max_correspondence_distance = 1.0;  // Maximum correspondence distance between points (e.g., triming threshold)

  Eigen::Isometry3d init_T_target_source = Eigen::Isometry3d::Identity();
  RegistrationResult result = align(target_points, source_points, init_T_target_source, setting);

  std::cout << "--- T_target_source ---" << std::endl << result.T_target_source.matrix() << std::endl;
  std::cout << "converged:" << result.converged << std::endl;
  std::cout << "error:" << result.error << std::endl;
  std::cout << "iterations:" << result.iterations << std::endl;
  std::cout << "num_inliers:" << result.num_inliers << std::endl;
  std::cout << "--- H ---" << std::endl << result.H << std::endl;
  std::cout << "--- b ---" << std::endl << result.b.transpose() << std::endl;
}

/// @brief Example to perform preprocessing and registration separately.
void example2(const std::vector<Eigen::Vector4f>& target_points, const std::vector<Eigen::Vector4f>& source_points) {
  int num_threads = 4;                    // Number of threads to be used
  double downsampling_resolution = 0.25;  // Downsampling resolution
  int num_neighbors = 10;                 // Number of neighbor points used for normal and covariance estimation

  // std::pair<PointCloud::Ptr, KdTree<PointCloud>::Ptr>
  auto [target, target_tree] = preprocess_points(target_points, downsampling_resolution, num_neighbors, num_threads);
  auto [source, source_tree] = preprocess_points(source_points, downsampling_resolution, num_neighbors, num_threads);

  RegistrationSetting setting;
  setting.num_threads = num_threads;
  setting.max_correspondence_distance = 1.0;  // Maximum correspondence distance between points (e.g., triming threshold)

  Eigen::Isometry3d init_T_target_source = Eigen::Isometry3d::Identity();
  RegistrationResult result = align(*target, *source, *target_tree, init_T_target_source, setting);

  std::cout << "--- T_target_source ---" << std::endl << result.T_target_source.matrix() << std::endl;
  std::cout << "converged:" << result.converged << std::endl;
  std::cout << "error:" << result.error << std::endl;
  std::cout << "iterations:" << result.iterations << std::endl;
  std::cout << "num_inliers:" << result.num_inliers << std::endl;
  std::cout << "--- H ---" << std::endl << result.H << std::endl;
  std::cout << "--- b ---" << std::endl << result.b.transpose() << std::endl;

  // Preprocessed points and trees can be reused for the next registration for efficiency
  RegistrationResult result2 = align(*source, *target, *source_tree, Eigen::Isometry3d::Identity(), setting);
}

int main(int argc, char** argv) {
  std::vector<Eigen::Vector4f> target_points = read_ply("data/target.ply");
  std::vector<Eigen::Vector4f> source_points = read_ply("data/source.ply");
  if (target_points.empty() || source_points.empty()) {
    std::cerr << "error: failed to read points from data/(target|source).ply" << std::endl;
    return 1;
  }

  example1(target_points, source_points);
  example2(target_points, source_points);

  return 0;
}