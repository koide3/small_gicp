// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <Eigen/Core>
#include <small_gicp/points/traits.hpp>

namespace small_gicp {
namespace traits {

template <>
struct Traits<Eigen::MatrixXd> {
  static size_t size(const Eigen::MatrixXd& points) { return points.rows(); }
  static bool has_points(const Eigen::MatrixXd& points) { return points.rows(); }
  static Eigen::Vector4d point(const Eigen::MatrixXd& points, size_t i) { return Eigen::Vector4d(points(i, 0), points(i, 1), points(i, 2), 1.0); }
};

template <typename Scalar, int D>
struct Traits<std::vector<Eigen::Matrix<Scalar, D, 1>>> {
  static_assert(D == 3 || D == 4, "Only 3D and 4D (homogeneous) points are supported");
  static_assert(std::is_floating_point<Scalar>::value, "Only floating point types are supported");

  using Point = Eigen::Matrix<Scalar, D, 1>;
  static size_t size(const std::vector<Point>& points) { return points.size(); }
  static bool has_points(const std::vector<Point>& points) { return points.size(); }
  static Eigen::Vector4d point(const std::vector<Point>& points, size_t i) {
    if constexpr (std::is_same_v<Scalar, double>) {
      if constexpr (D == 3) {
        return points[i].homogeneous();
      } else {
        return points[i];
      }
    } else {
      if constexpr (D == 3) {
        return points[i].homogeneous().template cast<double>();
      } else {
        return points[i].template cast<double>();
      }
    }
  }
};

}  // namespace traits
}  // namespace small_gicp