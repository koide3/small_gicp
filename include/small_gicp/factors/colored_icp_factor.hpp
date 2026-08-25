// SPDX-FileCopyrightText: Copyright 2026 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <small_gicp/util/lie.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/points/traits.hpp>

namespace small_gicp {

/// @brief Colored-ICP per-point error factor (J. Park et al., "Colored Point Cloud Registration Revisited", ICCV 2017).
/// @note  The target point cloud must have colors, normals, and color gradients, and the source point cloud must have colors.
struct ColoredICPFactor {
  /// @brief Factor setting
  struct Setting {
    Setting() : lambda_geometric(0.968) {}

    double lambda_geometric;  ///< Weight of the geometric (point-to-plane) term (the photometric term weights 1.0 - lambda_geometric)
  };

  /// @brief Constructor
  ColoredICPFactor(const Setting& setting = Setting())
  : lambda_geometric(setting.lambda_geometric),
    target_index(std::numeric_limits<size_t>::max()),
    source_index(std::numeric_limits<size_t>::max()),
    p_target(Eigen::Vector3d::Zero()),
    normal(Eigen::Vector3d::Zero()),
    grad(Eigen::Vector3d::Zero()),
    intensity_target(0.0),
    intensity_source(0.0) {}

  /// @brief Linearize the factor
  /// @param target       Target point cloud
  /// @param source       Source point cloud
  /// @param target_tree  Nearest neighbor search for the target point cloud
  /// @param T            Linearization point
  /// @param source_index Source point index
  /// @param rejector     Correspondence rejector
  /// @param H            Linearized information matrix
  /// @param b            Linearized information vector
  /// @param e            Error at the linearization point
  /// @return             True if the point is inlier
  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree, typename CorrespondenceRejector>
  bool linearize(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const Eigen::Isometry3d& T,
    size_t source_index,
    const CorrespondenceRejector& rejector,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* e) {
    //
    this->source_index = source_index;
    this->target_index = std::numeric_limits<size_t>::max();

    const double sqrt_lambda_geometric = std::sqrt(lambda_geometric);
    const double sqrt_lambda_photometric = std::sqrt(1.0 - lambda_geometric);

    const Eigen::Vector4d transed_source_pt = T * traits::point(source, source_index);

    size_t k_index;
    double k_sq_dist;
    if (!traits::nearest_neighbor_search(target_tree, transed_source_pt, &k_index, &k_sq_dist) || rejector(target, source, T, k_index, source_index, k_sq_dist)) {
      return false;
    }

    target_index = k_index;

    const Eigen::Vector3d p_target = traits::point(target, target_index).template head<3>();
    const Eigen::Vector3d n = traits::normal(target, target_index).template head<3>();
    const Eigen::Vector3d color_grad = traits::color_grad(target, target_index).template head<3>();

    const auto color_t = traits::color(target, target_index);
    const auto color_s = traits::color(source, source_index);
    const double intensity_t = (color_t[0] + color_t[1] + color_t[2]) / 3.0;
    const double intensity_s = (color_s[0] + color_s[1] + color_s[2]) / 3.0;

    // Project the color gradient onto the tangent plane
    const Eigen::Vector3d g = color_grad - color_grad.dot(n) * n;

    this->p_target = p_target;
    this->normal = n;
    this->grad = g;
    this->intensity_target = intensity_t;
    this->intensity_source = intensity_s;

    const Eigen::Matrix3d R = T.linear();
    const Eigen::Matrix3d Rp_s = R * skew(traits::point(source, source_index).template head<3>());

    Eigen::Matrix<double, 2, 6> J = Eigen::Matrix<double, 2, 6>::Zero();
    Eigen::Matrix<double, 2, 1> r;

    // Geometric term (point-to-plane residual decreases with the transformed source point)
    r(0) = sqrt_lambda_geometric * n.dot(p_target - transed_source_pt.template head<3>());
    J.row(0).head<3>() = (sqrt_lambda_geometric * n).transpose() * Rp_s;
    J.row(0).tail<3>() = (sqrt_lambda_geometric * n).transpose() * (-R);

    // Photometric term (color consistency residual increases with the transformed source point, hence the flipped Jacobian signs)
    r(1) = sqrt_lambda_photometric * (intensity_t + g.dot(transed_source_pt.template head<3>() - p_target) - intensity_s);
    J.row(1).head<3>() = (sqrt_lambda_photometric * g).transpose() * (-Rp_s);
    J.row(1).tail<3>() = (sqrt_lambda_photometric * g).transpose() * R;

    *H = J.transpose() * J;
    *b = J.transpose() * r;
    *e = 0.5 * r.squaredNorm();

    return true;
  }

  /// @brief Evaluate error
  /// @param target   Target point cloud
  /// @param source   Source point cloud
  /// @param T        Evaluation point
  /// @return Error
  template <typename TargetPointCloud, typename SourcePointCloud>
  double error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T) const {
    if (target_index == std::numeric_limits<size_t>::max()) {
      return 0.0;
    }

    const double sqrt_lambda_geometric = std::sqrt(lambda_geometric);
    const double sqrt_lambda_photometric = std::sqrt(1.0 - lambda_geometric);

    const Eigen::Vector3d transed_source_pt = (T * traits::point(source, source_index)).template head<3>();
    const double r0 = sqrt_lambda_geometric * normal.dot(p_target - transed_source_pt);
    const double r1 = sqrt_lambda_photometric * (intensity_target + grad.dot(transed_source_pt - p_target) - intensity_source);
    return 0.5 * (r0 * r0 + r1 * r1);
  }

  /// @brief Returns true if this factor is not rejected as an outlier
  bool inlier() const { return target_index != std::numeric_limits<size_t>::max(); }

  double lambda_geometric;  ///< Weight of the geometric (point-to-plane) term

  size_t target_index;       ///< Target point index
  size_t source_index;       ///< Source point index
  Eigen::Vector3d p_target;  ///< Target point (cached for error evaluation)
  Eigen::Vector3d normal;    ///< Target normal (cached for error evaluation)
  Eigen::Vector3d grad;      ///< Tangent-projected color gradient (cached for error evaluation)
  double intensity_target;   ///< Target intensity (cached for error evaluation)
  double intensity_source;   ///< Source intensity (cached for error evaluation)
};
}  // namespace small_gicp
