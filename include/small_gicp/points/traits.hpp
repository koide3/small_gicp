#pragma once

#include <Eigen/Core>

namespace small_gicp {

namespace traits {

template <typename T>
struct Traits;

template <typename T>
size_t size(const T& points) {
  return Traits<T>::size(points);
}

template <typename T>
bool has_points(const T& points) {
  return Traits<T>::has_points(points);
}

template <typename T>
bool has_normals(const T& points) {
  return Traits<T>::has_normals(points);
}

template <typename T>
bool has_covs(const T& points) {
  return Traits<T>::has_covs(points);
}

template <typename T>
auto point(const T& points, size_t i) {
  return Traits<T>::point(points, i);
}

template <typename T>
auto normal(const T& points, size_t i) {
  return Traits<T>::normal(points, i);
}

template <typename T>
auto cov(const T& points, size_t i) {
  return Traits<T>::cov(points, i);
}

template <typename T>
void resize(T& points, size_t n) {
  Traits<T>::resize(points, n);
}

template <typename T>
void set_point(T& points, size_t i, const Eigen::Vector4d& pt) {
  Traits<T>::set_point(points, i, pt);
}

template <typename T>
void set_normal(T& points, size_t i, const Eigen::Vector4d& pt) {
  Traits<T>::set_normal(points, i, pt);
}

template <typename T>
void set_cov(T& points, size_t i, const Eigen::Matrix4d& cov) {
  Traits<T>::set_cov(points, i, cov);
}

}  // namespace traits
}  // namespace small_gicp
