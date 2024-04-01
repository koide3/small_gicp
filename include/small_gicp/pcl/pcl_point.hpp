// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <pcl/point_types.h>

namespace pcl {

using Matrix4fMap = Eigen::Map<Eigen::Matrix4f, Eigen::Aligned>;
using Matrix4fMapConst = const Eigen::Map<const Eigen::Matrix4f, Eigen::Aligned>;

/// @brief  Point with covariance
struct PointCovariance {
  PCL_ADD_POINT4D;
  Eigen::Matrix4f cov;

  Matrix4fMap getCovariance4fMap() { return Matrix4fMap(cov.data()); }
  Matrix4fMapConst getCovariance4fMap() const { return Matrix4fMapConst(cov.data()); }

  PCL_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief  Point with normal and covariance
struct PointNormalCovariance {
  PCL_ADD_POINT4D;
  PCL_ADD_NORMAL4D
  Eigen::Matrix4f cov;

  Matrix4fMap getCovariance4fMap() { return Matrix4fMap(cov.data()); }
  Matrix4fMapConst getCovariance4fMap() const { return Matrix4fMapConst(cov.data()); }

  PCL_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace pcl
