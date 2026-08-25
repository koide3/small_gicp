// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>

namespace small_gicp {

namespace traits {

template <typename T>
struct Traits;

/// @brief Check if Traits<T> has color accessor (i.e., the point cloud supports color attributes).
template <typename T>
struct has_color {
  template <typename U, int = (&Traits<U>::color, 0)>
  static std::true_type test(U*);
  static std::false_type test(...);

  static constexpr bool value = decltype(test((T*)nullptr))::value;
};

/// @brief Check if Traits<T> has color gradient accessor.
template <typename T>
struct has_color_grad {
  template <typename U, int = (&Traits<U>::color_grad, 0)>
  static std::true_type test(U*);
  static std::false_type test(...);

  static constexpr bool value = decltype(test((T*)nullptr))::value;
};

/// @brief Check if Traits<T> has set_color (i.e., color attributes can be written to the point cloud).
template <typename T>
struct has_set_color {
  template <typename U, int = (&Traits<U>::set_color, 0)>
  static std::true_type test(U*);
  static std::false_type test(...);

  static constexpr bool value = decltype(test((T*)nullptr))::value;
};

/// @brief Check if Traits<T> has set_color_grad.
template <typename T>
struct has_set_color_grad {
  template <typename U, int = (&Traits<U>::set_color_grad, 0)>
  static std::true_type test(U*);
  static std::false_type test(...);

  static constexpr bool value = decltype(test((T*)nullptr))::value;
};

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

/// @brief Check if the point cloud has colors.
/// @note  As with normals and covariances, color values of a point cloud that never had colors set are unspecified
///        (small_gicp::PointCloud only guarantees that the buffers are sized), so a true return value does not mean
///        that meaningful colors were assigned.
template <typename T>
bool has_colors(const T& points) {
  return Traits<T>::has_colors(points);
}

/// @brief Check if the point cloud has color gradients.
template <typename T>
bool has_color_grads(const T& points) {
  return Traits<T>::has_color_grads(points);
}

/// @brief Get i-th color. The last element must be filled by zero (r, g, b, 0).
template <typename T>
auto color(const T& points, size_t i) {
  return Traits<T>::color(points, i);
}

/// @brief Get i-th color gradient. The last element must be filled by zero (dI/dx, dI/dy, dI/dz, 0).
template <typename T>
auto color_grad(const T& points, size_t i) {
  return Traits<T>::color_grad(points, i);
}

/// @brief Set i-th color. (r, g, b, 0)
template <typename T>
void set_color(T& points, size_t i, const Eigen::Vector4d& color) {
  Traits<T>::set_color(points, i, color);
}

/// @brief Set i-th color gradient. (dI/dx, dI/dy, dI/dz, 0)
template <typename T>
void set_color_grad(T& points, size_t i, const Eigen::Vector4d& grad) {
  Traits<T>::set_color_grad(points, i, grad);
}

}  // namespace traits
}  // namespace small_gicp
