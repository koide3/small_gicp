// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstddef>
#include <vector>

namespace small_gicp {

/// @brief One LM trial, including rejected attempts. Candidate error is the
/// reduction.error value; comparison_reference_error is the value against
/// which the optimizer actually compared it (including general factor updates).
struct OptimizationTrialRecord {
  size_t trial = 0;
  double lambda = 0.0;
  double candidate_error = 0.0;
  double comparison_reference_error = 0.0;
  Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix4d T_candidate = Eigen::Matrix4d::Identity();
  bool accepted = false;
};

/// @brief State of one outer optimization iteration.
/// H, b, and linearized_error are computed at T_before, not at T_after.
struct IterationRecord {
  size_t iteration = 0;
  Eigen::Matrix4d T_before = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_after = Eigen::Matrix4d::Identity();
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
  double linearized_error = 0.0;
  double reported_error = 0.0;
  double lambda_before = 0.0;
  double lambda_after = 0.0;
  size_t num_inliers = 0;
  bool accepted = false;
  std::vector<OptimizationTrialRecord> trials;
};

/// @brief Registration result
struct RegistrationResult {
  RegistrationResult(const Eigen::Isometry3d& T = Eigen::Isometry3d::Identity())
  : T_target_source(T),
    converged(false),
    iterations(0),
    num_inliers(0),
    H(Eigen::Matrix<double, 6, 6>::Zero()),
    b(Eigen::Matrix<double, 6, 1>::Zero()),
    error(0.0) {}

  Eigen::Isometry3d T_target_source;  ///<  Estimated transformation

  bool converged;      ///< If the optimization converged
  size_t iterations;   ///< Number of optimization iterations
  size_t num_inliers;  ///< Number of inliear points

  Eigen::Matrix<double, 6, 6> H;  ///< Final information matrix
  Eigen::Matrix<double, 6, 1> b;  ///< Final information vector
  double error;                   ///< Final error
  std::vector<IterationRecord> history;  ///< Optional per-iteration trace
};

}  // namespace small_gicp
