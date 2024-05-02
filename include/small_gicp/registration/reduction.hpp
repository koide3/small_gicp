// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <Eigen/Core>

namespace small_gicp {

/// @brief Single-thread reduction
struct SerialReduction {
  /// @brief Sum up linearized systems
  /// @param target       Target point cloud
  /// @param source       Source point cloud
  /// @param target_tree  Nearest neighbor search for the target point cloud
  /// @param rejector     Correspondence rejector
  /// @param T            Linearization point
  /// @param factors      Factors to be linearized
  /// @return             Sum of the linearized systems (information matrix, information vector, and error)
  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree, typename CorrespondenceRejector, typename Factor>
  std::tuple<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>, double> linearize(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const CorrespondenceRejector& rejector,
    const Eigen::Isometry3d& T,
    std::vector<Factor>& factors) const {
    Eigen::Matrix<double, 6, 6> sum_H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> sum_b = Eigen::Matrix<double, 6, 1>::Zero();
    double sum_e = 0.0;

    for (size_t i = 0; i < factors.size(); i++) {
      Eigen::Matrix<double, 6, 6> H;
      Eigen::Matrix<double, 6, 1> b;
      double e;

      if (!factors[i].linearize(target, source, target_tree, T, i, rejector, &H, &b, &e)) {
        continue;
      }

      sum_H += H;
      sum_b += b;
      sum_e += e;
    }

    return {sum_H, sum_b, sum_e};
  }

  /// @brief Sum up evaluated errors
  /// @param target       Target point cloud
  /// @param source       Source point cloud
  /// @param T            Evaluation point
  /// @param factors      Factors to be evaluated
  /// @return Sum of the evaluated errors
  template <typename TargetPointCloud, typename SourcePointCloud, typename Factor>
  double error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T, std::vector<Factor>& factors) const {
    double sum_e = 0.0;
    for (size_t i = 0; i < factors.size(); i++) {
      sum_e += factors[i].error(target, source, T);
    }
    return sum_e;
  }
};

}  // namespace small_gicp
