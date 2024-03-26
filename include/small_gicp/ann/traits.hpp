#pragma once

#include <Eigen/Core>

namespace small_gicp {

namespace traits {

template <typename T>
struct Traits;

/// @brief Find k-nearest neighbors
/// @param tree       Nearest neighbor search (e.g., KdTree)
/// @param point      Query point
/// @param k          Number of neighbors
/// @param k_index    [out] Index of the nearest neighbor
/// @param k_sq_dist  [out] Squared distance to the nearest neighbor
/// @return Number of found neighbors
template <typename T>
size_t knn_search(const T& tree, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
  return Traits<T>::knn_search(tree, point, k, k_indices, k_sq_dists);
}

template <typename T>
concept HasNearestNeighborSearch = requires(const T& tree) { Traits<T>::nearest_neighbor_search(tree, Eigen::Vector4d(), nullptr, nullptr); };

/// @brief Find the nearest neighbor
/// @param tree       Nearest neighbor search (e.g., KdTree)
/// @param point      Query point
/// @param k_index    [out] Index of the nearest neighbor
/// @param k_sq_dist  [out] Squared distance to the nearest neighbor
/// @return 1 if a neighbor is found else 0
template <HasNearestNeighborSearch T>
size_t nearest_neighbor_search(const T& tree, const Eigen::Vector4d& point, size_t* k_index, double* k_sq_dist) {
  return Traits<T>::nearest_neighbor_search(tree, point, k_index, k_sq_dist);
}

template <typename T>
size_t nearest_neighbor_search(const T& tree, const Eigen::Vector4d& point, size_t* k_index, double* k_sq_dist) {
  return Traits<T>::knn_search(tree, point, 1, k_index, k_sq_dist);
}

}  // namespace traits

}  // namespace small_gicp
