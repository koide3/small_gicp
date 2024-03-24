#pragma once

#include <Eigen/Core>

namespace small_gicp {

namespace traits {

template <typename T>
struct Traits;

/// @brief  Get the number of points
template <typename T>
size_t size(const T& points) {
  return Traits<T>::size(points);
}

/// @brief Check if the point cloud has points
template <typename T>
bool has_points(const T& points) {
  return Traits<T>::has_points(points);
}

/// @brief Check if the point cloud has normals
template <typename T>
bool has_normals(const T& points) {
  return Traits<T>::has_normals(points);
}

/// @brief Check if the point cloud has covariances
template <typename T>
bool has_covs(const T& points) {
  return Traits<T>::has_covs(points);
}

/// @brief Get i-th point
template <typename T>
auto point(const T& points, size_t i) {
  return Traits<T>::point(points, i);
}

/// @brief Get i-th normal
template <typename T>
auto normal(const T& points, size_t i) {
  return Traits<T>::normal(points, i);
}

/// @brief Get i-th covariance
template <typename T>
auto cov(const T& points, size_t i) {
  return Traits<T>::cov(points, i);
}

/// @brief Resize the point cloud (this function should resize all attributes)
template <typename T>
void resize(T& points, size_t n) {
  Traits<T>::resize(points, n);
}

/// @brief Set i-th point
template <typename T>
void set_point(T& points, size_t i, const Eigen::Vector4d& pt) {
  Traits<T>::set_point(points, i, pt);
}

/// @brief Set i-th normal
template <typename T>
void set_normal(T& points, size_t i, const Eigen::Vector4d& pt) {
  Traits<T>::set_normal(points, i, pt);
}

/// @brief Set i-th covariance
template <typename T>
void set_cov(T& points, size_t i, const Eigen::Matrix4d& cov) {
  Traits<T>::set_cov(points, i, cov);
}

}  // namespace traits
}  // namespace small_gicp
