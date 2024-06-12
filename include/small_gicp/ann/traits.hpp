// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <type_traits>
#include <Eigen/Core>

namespace small_gicp {

namespace traits {

template <typename T>
struct Traits;

/// @brief Find k-nearest neighbors.
/// @param tree       Nearest neighbor search (e.g., KdTree)
/// @param point      Query point
/// @param k          Number of neighbors
/// @param k_indices  [out] Indices of k-nearest neighbors
/// @param k_sq_dists [out] Squared distances to k-nearest neighbors
/// @return Number of found neighbors
template <typename T>
size_t knn_search(const T& tree, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
  return Traits<T>::knn_search(tree, point, k, k_indices, k_sq_dists);
}

/// @brief Check if T has nearest_neighbor_search method.
template <typename T>
struct has_nearest_neighbor_search {
  template <typename U, int = (&Traits<U>::nearest_neighbor_search, 0)>
  static std::true_type test(U*);
  static std::false_type test(...);

  static constexpr bool value = decltype(test((T*)nullptr))::value;
};

/// @brief Find the nearest neighbor. If Traits<T>::nearest_neighbor_search is not defined, fallback to knn_search with k=1.
/// @param tree       Nearest neighbor search (e.g., KdTree)
/// @param point      Query point
/// @param k_index    [out] Index of the nearest neighbor
/// @param k_sq_dist  [out] Squared distance to the nearest neighbor
/// @return 1 if a neighbor is found else 0
template <typename T, std::enable_if_t<has_nearest_neighbor_search<T>::value, bool> = true>
size_t nearest_neighbor_search(const T& tree, const Eigen::Vector4d& point, size_t* k_index, double* k_sq_dist) {
  return Traits<T>::nearest_neighbor_search(tree, point, k_index, k_sq_dist);
}

/// @brief Find the nearest neighbor. If Traits<T>::nearest_neighbor_search is not defined, fallback to knn_search with k=1.
/// @param tree       Nearest neighbor search (e.g., KdTree)
/// @param point      Query point
/// @param k_index    [out] Index of the nearest neighbor
/// @param k_sq_dist  [out] Squared distance to the nearest neighbor
/// @return 1 if a neighbor is found else 0
template <typename T, std::enable_if_t<!has_nearest_neighbor_search<T>::value, bool> = true>
size_t nearest_neighbor_search(const T& tree, const Eigen::Vector4d& point, size_t* k_index, double* k_sq_dist) {
  return Traits<T>::knn_search(tree, point, 1, k_index, k_sq_dist);
}

}  // namespace traits

}  // namespace small_gicp
