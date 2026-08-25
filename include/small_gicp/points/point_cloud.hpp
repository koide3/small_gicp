// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <small_gicp/points/traits.hpp>

namespace small_gicp {

/**
 * @brief Point cloud
 */
struct PointCloud {
public:
  using Ptr = std::shared_ptr<PointCloud>;
  using ConstPtr = std::shared_ptr<const PointCloud>;

  /// @brief Constructor
  PointCloud() {}

  /// @brief Constructor
  /// @param points  Points to initialize the point cloud
  template <typename T, int D, typename Allocator>
  explicit PointCloud(const std::vector<Eigen::Matrix<T, D, 1>, Allocator>& points) {
    this->resize(points.size());
    for (size_t i = 0; i < points.size(); i++) {
      this->point(i) << points[i].template cast<double>().template head<3>(), 1.0;
    }
  }

  /// @brief Destructor
  ~PointCloud() {}

  /// @brief Number of points.
  size_t size() const { return points.size(); }

  /// @brief Check if the point cloud is empty.
  bool empty() const { return points.empty(); }

  /// @brief Resize point/normal/cov/color buffers.
  /// @param n  Number of points
  void resize(size_t n) {
    points.resize(n);
    normals.resize(n);
    covs.resize(n);
    colors.resize(n);
    color_grads.resize(n);
  }

  /// @brief Get i-th point.
  Eigen::Vector4d& point(size_t i) { return points[i]; }

  /// @brief Get i-th normal.
  Eigen::Vector4d& normal(size_t i) { return normals[i]; }

  /// @brief Get i-th covariance.
  Eigen::Matrix4d& cov(size_t i) { return covs[i]; }

  /// @brief Get i-th color.
  Eigen::Vector4d& color(size_t i) { return colors[i]; }

  /// @brief Get i-th color gradient.
  Eigen::Vector4d& color_grad(size_t i) { return color_grads[i]; }

  /// @brief Get i-th point (const).
  const Eigen::Vector4d& point(size_t i) const { return points[i]; }

  /// @brief Get i-th normal (const).
  const Eigen::Vector4d& normal(size_t i) const { return normals[i]; }

  /// @brief Get i-th covariance (const).
  const Eigen::Matrix4d& cov(size_t i) const { return covs[i]; }

  /// @brief Get i-th color (const).
  const Eigen::Vector4d& color(size_t i) const { return colors[i]; }

  /// @brief Get i-th color gradient (const).
  const Eigen::Vector4d& color_grad(size_t i) const { return color_grads[i]; }

public:
  std::vector<Eigen::Vector4d> points;       ///< Point coordinates (x, y, z, 1)
  std::vector<Eigen::Vector4d> normals;      ///< Point normals (nx, ny, nz, 0)
  std::vector<Eigen::Matrix4d> covs;         ///< Point covariances (3x3 matrix) + zero padding
  std::vector<Eigen::Vector4d> colors;       ///< Point colors (r, g, b, 0)
  std::vector<Eigen::Vector4d> color_grads;  ///< Point color gradients (dI/dx, dI/dy, dI/dz, 0)
};

namespace traits {

template <>
struct Traits<PointCloud> {
  using Points = PointCloud;

  static size_t size(const Points& points) { return points.size(); }

  static bool has_points(const Points& points) { return !points.points.empty(); }
  static bool has_normals(const Points& points) { return !points.normals.empty(); }
  static bool has_covs(const Points& points) { return !points.covs.empty(); }
  static bool has_colors(const Points& points) { return !points.colors.empty(); }
  static bool has_color_grads(const Points& points) { return !points.color_grads.empty(); }

  static const Eigen::Vector4d& point(const Points& points, size_t i) { return points.point(i); }
  static const Eigen::Vector4d& normal(const Points& points, size_t i) { return points.normal(i); }
  static const Eigen::Matrix4d& cov(const Points& points, size_t i) { return points.cov(i); }
  static const Eigen::Vector4d& color(const Points& points, size_t i) { return points.color(i); }
  static const Eigen::Vector4d& color_grad(const Points& points, size_t i) { return points.color_grad(i); }

  static void resize(Points& points, size_t n) { points.resize(n); }
  static void set_point(Points& points, size_t i, const Eigen::Vector4d& pt) { points.point(i) = pt; }
  static void set_normal(Points& points, size_t i, const Eigen::Vector4d& n) { points.normal(i) = n; }
  static void set_cov(Points& points, size_t i, const Eigen::Matrix4d& cov) { points.cov(i) = cov; }
  static void set_color(Points& points, size_t i, const Eigen::Vector4d& color) { points.color(i) = color; }
  static void set_color_grad(Points& points, size_t i, const Eigen::Vector4d& grad) { points.color_grad(i) = grad; }
};

}  // namespace traits

}  // namespace small_gicp
