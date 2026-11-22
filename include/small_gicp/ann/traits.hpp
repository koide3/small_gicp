// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <type_traits>
#include <utility>
#include <Eigen/Core>

#include <small_gicp/ann/knn_result.hpp>

namespace small_gicp {

namespace traits {

template <typename T>
struct Traits;

namespace detail {

/// @brief Check if T exposes a maximum squared distance to a valid correspondence (e.g., small_gicp::DistanceRejector).
template <typename T, typename = void>
struct has_max_dist_sq : std::false_type {};

template <typename T>
struct has_max_dist_sq<T, std::void_t<decltype(std::declval<const T&>().max_dist_sq)>> : std::true_type {};

/// @brief Check if Traits<T> implements the nearest neighbor search with a search setting.
template <typename T, typename = void>
struct has_knn_setting_nearest_neighbor_search : std::false_type {};

template <typename T>
struct has_knn_setting_nearest_neighbor_search<
  T,
  std::void_t<decltype(Traits<T>::nearest_neighbor_search(
    std::declval<const T&>(),
    std::declval<const Eigen::Vector4d&>(),
    std::declval<size_t*>(),
    std::declval<double*>(),
    std::declval<const KnnSetting&>()))>> : std::true_type {};

}  // namespace detail

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

/// @brief Find the nearest neighbor, bounding the search by the maximum distance of the given rejector.
/// @note  If the rejector has no distance bound (e.g., NullRejector) or the search structure does not support
///        a bounded search (e.g., GaussianVoxelMap), this falls back to the unbounded search. The caller must
///        reject the found correspondence by itself in that case, and even in the bounded case for consistency
///        of the acceptance test (a neighbor at exactly the bound may be returned).
/// @param tree       Nearest neighbor search (e.g., KdTree)
/// @param point      Query point
/// @param k_index    [out] Index of the nearest neighbor
/// @param k_sq_dist  [out] Squared distance to the nearest neighbor
/// @param rejector   Correspondence rejector (e.g., small_gicp::DistanceRejector)
/// @return 1 if a neighbor within the bound is found, else 0
template <typename T, typename Rejector>
size_t nearest_neighbor_search(const T& tree, const Eigen::Vector4d& point, size_t* k_index, double* k_sq_dist, const Rejector& rejector) {
  KnnSetting setting;
  if constexpr (detail::has_max_dist_sq<Rejector>::value) {
    setting.max_dist_sq = static_cast<double>(rejector.max_dist_sq);
  }

  if constexpr (detail::has_knn_setting_nearest_neighbor_search<T>::value) {
    return Traits<T>::nearest_neighbor_search(tree, point, k_index, k_sq_dist, setting);
  } else {
    (void)setting;
    return nearest_neighbor_search(tree, point, k_index, k_sq_dist);
  }
}

}  // namespace traits

}  // namespace small_gicp
