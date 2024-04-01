// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>

namespace small_gicp {

namespace traits {

template <typename T>
struct Traits;

/// @brief  Get the number of points.
template <typename T>
size_t size(const T& points) {
  return Traits<T>::size(points);
}

/// @brief Check if the point cloud has points.
template <typename T>
bool has_points(const T& points) {
  return Traits<T>::has_points(points);
}

/// @brief Check if the point cloud has normals.
template <typename T>
bool has_normals(const T& points) {
  return Traits<T>::has_normals(points);
}

/// @brief Check if the point cloud has covariances.
template <typename T>
bool has_covs(const T& points) {
  return Traits<T>::has_covs(points);
}

/// @brief Get i-th point. 4D vector is used to take advantage of SIMD intrinsics. The last element must be filled by one (x, y, z, 1).
template <typename T>
auto point(const T& points, size_t i) {
  return Traits<T>::point(points, i);
}

/// @brief Get i-th normal. 4D vector is used to take advantage of SIMD intrinsics. The last element must be filled by zero (nx, ny, nz, 0).
template <typename T>
auto normal(const T& points, size_t i) {
  return Traits<T>::normal(points, i);
}

/// @brief Get i-th covariance. Only the top-left 3x3 matrix is filled, and the bottom row and the right col must be filled by zero.
template <typename T>
auto cov(const T& points, size_t i) {
  return Traits<T>::cov(points, i);
}

/// @brief Resize the point cloud (this function should resize all attributes)
template <typename T>
void resize(T& points, size_t n) {
  Traits<T>::resize(points, n);
}

/// @brief Set i-th point. (x, y, z, 1)
template <typename T>
void set_point(T& points, size_t i, const Eigen::Vector4d& pt) {
  Traits<T>::set_point(points, i, pt);
}

/// @brief Set i-th normal. (nx, nz, nz, 0)
template <typename T>
void set_normal(T& points, size_t i, const Eigen::Vector4d& pt) {
  Traits<T>::set_normal(points, i, pt);
}

/// @brief Set i-th covariance. Only the top-left 3x3 matrix should be filled.
template <typename T>
void set_cov(T& points, size_t i, const Eigen::Matrix4d& cov) {
  Traits<T>::set_cov(points, i, cov);
}

}  // namespace traits
}  // namespace small_gicp
