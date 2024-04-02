// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <small_gicp/points/traits.hpp>

namespace small_gicp {

namespace traits {

template <typename PointType>
struct Traits<pcl::PointCloud<PointType>> {
  static_assert(std::is_same_v<pcl::shared_ptr<void>, std::shared_ptr<void>>, "Old PCL version detected. Please update PCL to 1.11 or later.");

  using Points = pcl::PointCloud<PointType>;

  static size_t size(const Points& points) { return points.size(); }
  static void resize(Points& points, size_t n) { points.resize(n); }

  static bool has_points(const Points& points) { return !points.empty(); }
  static bool has_normals(const Points& points) { return !points.empty(); }
  static bool has_covs(const Points& points) { return !points.empty(); }

  static void set_point(Points& points, size_t i, const Eigen::Vector4d& pt) { points.at(i).getVector4fMap() = pt.template cast<float>(); }
  static void set_normal(Points& points, size_t i, const Eigen::Vector4d& pt) { points.at(i).getNormalVector4fMap() = pt.template cast<float>(); }
  static void set_cov(Points& points, size_t i, const Eigen::Matrix4d& cov) { points.at(i).getCovariance4fMap() = cov.template cast<float>(); }

  static Eigen::Vector4d point(const Points& points, size_t i) { return points.at(i).getVector4fMap().template cast<double>(); }
  static Eigen::Vector4d normal(const Points& points, size_t i) { return points.at(i).getNormalVector4fMap().template cast<double>(); }
  static Eigen::Matrix4d cov(const Points& points, size_t i) { return points.at(i).getCovariance4fMap().template cast<double>(); }
};

}  // namespace traits

}  // namespace small_gicp
