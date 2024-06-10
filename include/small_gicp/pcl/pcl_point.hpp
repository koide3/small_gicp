// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <pcl/point_types.h>

namespace pcl {

using Matrix4fMap = Eigen::Map<Eigen::Matrix4f, Eigen::Aligned>;
using Matrix4fMapConst = const Eigen::Map<const Eigen::Matrix4f, Eigen::Aligned>;

/// @brief  Point with covariance for PCL
struct PointCovariance {
  PCL_ADD_POINT4D;        ///< Point coordinates
  Eigen::Matrix4f cov;    ///< 4x4 covariance matrix

  /// @brief Get covariance matrix as Matrix4fMap
  Matrix4fMap getCovariance4fMap() { return Matrix4fMap(cov.data()); }

  /// @brief Get covariance matrix as Matrix4fMapConst
  Matrix4fMapConst getCovariance4fMap() const { return Matrix4fMapConst(cov.data()); }

  PCL_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief  Point with normal and covariance for PCL
struct PointNormalCovariance {
  PCL_ADD_POINT4D;      ///< Point coordinates
  PCL_ADD_NORMAL4D      ///< Point normal
  Eigen::Matrix4f cov;  ///< 4x4 covariance matrix

  /// @brief Get covariance matrix as Matrix4fMap
  Matrix4fMap getCovariance4fMap() { return Matrix4fMap(cov.data()); }

  /// @brief Get covariance matrix as Matrix4fMapConst
  Matrix4fMapConst getCovariance4fMap() const { return Matrix4fMapConst(cov.data()); }

  PCL_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace pcl
