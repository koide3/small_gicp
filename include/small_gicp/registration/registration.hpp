// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <small_gicp/ann/traits.hpp>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/factors/general_factor.hpp>
#include <small_gicp/registration/rejector.hpp>
#include <small_gicp/registration/reduction.hpp>
#include <small_gicp/registration/optimizer.hpp>
#include <small_gicp/registration/termination_criteria.hpp>
#include <small_gicp/registration/registration_result.hpp>

namespace small_gicp {

/// @brief Point cloud registration.
template <
  typename PointFactor,
  typename Reduction,
  typename GeneralFactor = NullFactor,
  typename CorrespondenceRejector = DistanceRejector,
  typename Optimizer = LevenbergMarquardtOptimizer>
struct Registration {
public:
  /// @brief Align point clouds.
  /// @param target       Target point cloud
  /// @param source       Source point cloud
  /// @param target_tree  Nearest neighbor search for the target point cloud
  /// @param init_T       Initial guess
  /// @return             Registration result
  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree>
  RegistrationResult
  align(const TargetPointCloud& target, const SourcePointCloud& source, const TargetTree& target_tree, const Eigen::Isometry3d& init_T = Eigen::Isometry3d::Identity()) const {
    if (traits::size(target) <= 10) {
      std::cerr << "warning: target point cloud is too small. |target|=" << traits::size(target) << std::endl;
    }
    if (traits::size(source) <= 10) {
      std::cerr << "warning: source point cloud is too small. |source|=" << traits::size(source) << std::endl;
    }

    std::vector<PointFactor> factors(traits::size(source), PointFactor(point_factor));
    return optimizer.optimize(target, source, target_tree, rejector, criteria, reduction, init_T, factors, general_factor);
  }

public:
  using PointFactorSetting = typename PointFactor::Setting;

  TerminationCriteria criteria;     ///< Termination criteria
  CorrespondenceRejector rejector;  ///< Correspondence rejector
  PointFactorSetting point_factor;  ///< Factor setting
  GeneralFactor general_factor;     ///< General factor
  Reduction reduction;              ///< Reduction
  Optimizer optimizer;              ///< Optimizer
};

}  // namespace small_gicp
