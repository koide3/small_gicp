// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <Eigen/Core>
#include <Eigen/Eigen>

namespace small_gicp {

/// @brief Parameters to control the projection axis search.
struct ProjectionSetting {
  int max_scan_count = 128;  ///< Maximum number of points to use for the axis search.
};

/// @brief Conventional axis-aligned projection (i.e., selecting any of XYZ axes with the largest variance).
/// @note  Up to max_scan_count samples are used to estimate the variance.
struct AxisAlignedProjection {
public:
  /// @brief Project the point to the selected axis.
  /// @param pt  Point to project
  /// @return    Projected value
  double operator()(const Eigen::Vector4d& pt) const { return pt[axis]; }

  /// @brief Find the axis with the largest variance.
  /// @param points     Point cloud
  /// @param first      First point index iterator
  /// @param last       Last point index iterator
  /// @param setting    Search setting
  /// @return           Projection with the largest variance axis
  template <typename PointCloud, typename IndexConstIterator>
  static AxisAlignedProjection find_axis(const PointCloud& points, IndexConstIterator first, IndexConstIterator last, const ProjectionSetting& setting) {
    const size_t N = std::distance(first, last);
    Eigen::Vector4d sum_pt = Eigen::Vector4d::Zero();
    Eigen::Vector4d sum_sq = Eigen::Vector4d::Zero();

    const size_t step = N < setting.max_scan_count ? 1 : N / setting.max_scan_count;
    const size_t num_steps = N / step;
    for (int i = 0; i < num_steps; i++) {
      const auto itr = first + step * i;
      const Eigen::Vector4d pt = traits::point(points, *itr);
      sum_pt += pt;
      sum_sq += pt.cwiseProduct(pt);
    }

    const Eigen::Vector4d mean = sum_pt / sum_pt.w();
    const Eigen::Vector4d var = (sum_sq - mean.cwiseProduct(sum_pt));

    return AxisAlignedProjection{var[0] > var[1] ? (var[0] > var[2] ? 0 : 2) : (var[1] > var[2] ? 1 : 2)};
  }

public:
  int axis;  ///< Axis index (0: X, 1: Y, 2: Z)
};

/// @brief Normal projection (i.e., selecting the 3D direction with the largest variance of the points).
/// @note  Up to max_scan_count samples are used to estimate the variance along the axis.
struct NormalProjection {
public:
  /// @brief Project the point to the normal direction.
  /// @param pt   Point to project
  /// @return     Projected value
  double operator()(const Eigen::Vector4d& pt) const { return Eigen::Map<const Eigen::Vector3d>(normal.data()).dot(pt.head<3>()); }

  /// @brief  Find the direction with the largest variance.
  /// @param points   Point cloud
  /// @param first    First point index iterator
  /// @param last     Last point index iterator
  /// @param setting  Search setting
  /// @return         Projection with the largest variance direction
  template <typename PointCloud, typename IndexConstIterator>
  static NormalProjection find_axis(const PointCloud& points, IndexConstIterator first, IndexConstIterator last, const ProjectionSetting& setting) {
    const size_t N = std::distance(first, last);
    Eigen::Vector4d sum_pt = Eigen::Vector4d::Zero();
    Eigen::Matrix4d sum_sq = Eigen::Matrix4d::Zero();

    const size_t step = N < setting.max_scan_count ? 1 : N / setting.max_scan_count;
    const size_t num_steps = N / step;
    for (int i = 0; i < num_steps; i++) {
      const auto itr = first + step * i;
      const Eigen::Vector4d pt = traits::point(points, *itr);
      sum_pt += pt;
      sum_sq += pt * pt.transpose();
    }

    const Eigen::Vector4d mean = sum_pt / sum_pt.w();
    const Eigen::Matrix4d cov = (sum_sq - mean * sum_pt.transpose()) / sum_pt.w();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
    eig.computeDirect(cov.block<3, 3>(0, 0));

    NormalProjection p;
    Eigen::Map<Eigen::Vector3d>(p.normal.data()) = eig.eigenvectors().col(2);
    return p;
  }

public:
  std::array<double, 3> normal;  ///< Projection direction
};

}  // namespace small_gicp
