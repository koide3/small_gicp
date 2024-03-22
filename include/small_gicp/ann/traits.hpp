#pragma once

#include <Eigen/Core>

namespace small_gicp {

namespace traits {

template <typename T>
struct Traits;

template <typename T>
size_t knn_search(const T& tree, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
  return Traits<T>::knn_search(tree, point, k, k_indices, k_sq_dists);
}

}  // namespace traits

}  // namespace small_gicp
