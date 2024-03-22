#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace small_gicp {

struct RegistrationResult {
  RegistrationResult(const Eigen::Isometry3d& T)
  : T_target_source(T),
    converged(false),
    iterations(0),
    num_inliers(0),
    H(Eigen::Matrix<double, 6, 6>::Zero()),
    b(Eigen::Matrix<double, 6, 1>::Zero()),
    error(0.0) {}

  Eigen::Isometry3d T_target_source;

  bool converged;
  size_t iterations;
  size_t num_inliers;

  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double error;
};

}  // namespace small_gicp
