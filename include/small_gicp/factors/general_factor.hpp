// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace small_gicp {

/// @brief Null factor that gives no constraints.
struct NullFactor {
  NullFactor() = default;

  /// @brief Update linearized system consisting of linearized per-point factors.
  /// @param target       Target point cloud
  /// @param source       Source point cloud
  /// @param target_tree  Nearest neighbor search for the target point cloud
  /// @param T            Linearization point
  /// @param H            [in/out] Linearized information matrix.
  /// @param b            [in/out] Linearized information vector.
  /// @param e            [in/out] Error at the linearization point.
  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree>
  void update_linearized_system(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const Eigen::Isometry3d& T,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* e) const {}

  /// @brief Update error consisting of per-point factors.
  /// @param target   Target point cloud
  /// @param source   Source point cloud
  /// @param T        Evaluation point
  /// @param e        [in/out] Error at the evaluation point.
  template <typename TargetPointCloud, typename SourcePointCloud>
  void update_error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T, double* e) const {}
};

/// @brief Factor to restrict the degrees of freedom of optimization (e.g., fixing roll, pitch rotation).
/// @note  This factor only enables *soft* constraints. The source point cloud can move to the restricted directions slightly).
struct RestrictDoFFactor {
  /// @brief Constructor.
  RestrictDoFFactor() {
    lambda = 1e9;
    mask.setOnes();
  }

  /// @brief Set rotation mask. (1.0 = active, 0.0 = inactive)
  void set_rotation_mask(const Eigen::Array3d& rot_mask) { mask.head<3>() = rot_mask; }

  /// @brief  Set translation mask. (1.0 = active, 0.0 = inactive)
  void set_translation_mask(const Eigen::Array3d& trans_mask) { mask.tail<3>() = trans_mask; }

  /// @brief Update linearized system consisting of linearized per-point factors.
  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree>
  void update_linearized_system(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const Eigen::Isometry3d& T,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* e) const {
    *H += lambda * (mask - 1.0).abs().matrix().asDiagonal();
  }

  /// @brief Update error consisting of per-point factors.
  template <typename TargetPointCloud, typename SourcePointCloud>
  void update_error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T, double* e) const {}

  double lambda;                    ///< Regularization parameter (Increasing this makes the constraint stronger)
  Eigen::Array<double, 6, 1> mask;  ///< Mask for restricting DoF (rx, ry, rz, tx, ty, tz)
};

}  // namespace small_gicp
