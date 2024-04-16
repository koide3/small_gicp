#pragma once

#include <queue>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/points/traits.hpp>

namespace small_gicp {

/// @brief Flat point container
/// @note  IncrementalVoxelMap combined with FlastContainer is mostly the same as iVox.
///        Bai et al., "Faster-LIO: Lightweight Tightly Coupled Lidar-Inertial Odometry Using Parallel Sparse Incremental Voxels", IEEE RA-L, 2022
/// @tparam HasNormals  If true, normals are stored
/// @tparam HasCovs     If true, covariances are stored
template <bool HasNormals = false, bool HasCovs = false>
struct FlatContainer {
public:
  struct Setting {
    double min_sq_dist_in_cell = 0.1 * 0.1;  ///< Minimum squared distance between points in a cell
    size_t max_num_points_in_cell = 10;      ///< Maximum number of points in a cell
  };

  /// @brief Constructor
  FlatContainer() { points.reserve(5); }

  /// @brief Number of points
  size_t size() const { return points.size(); }

  /// @brief Add a point to the container
  template <typename PointCloud>
  void add(const Setting& setting, const Eigen::Vector4d& transformed_pt, const PointCloud& points, size_t i, const Eigen::Isometry3d& T) {
    if (
      this->points.size() >= setting.max_num_points_in_cell ||                                                                                                  //
      std::any_of(this->points.begin(), this->points.end(), [&](const auto& pt) { return (pt - transformed_pt).squaredNorm() < setting.min_sq_dist_in_cell; })  //
    ) {
      return;
    }

    this->points.emplace_back(transformed_pt);
    if constexpr (HasNormals) {
      this->normals.emplace_back(T.matrix() * traits::normal(points, i));
    }
    if constexpr (HasCovs) {
      this->covs.emplace_back(T.matrix() * traits::cov(points, i) * T.matrix().transpose());
    }
  }

  /// @brief Finalize the container (Nothing to do for FlatContainer)
  void finalize() {}

  /// @brief Find the nearest neighbor
  /// @param pt           Query point
  /// @param k_index      Index of the nearest neighbor
  /// @param k_sq_dist    Squared distance to the nearest neighbor
  /// @return             Number of found points (0 or 1)
  size_t nearest_neighbor_search(const Eigen::Vector4d& pt, size_t* k_index, double* k_sq_dist) const {
    if (points.empty()) {
      return 0;
    }

    size_t min_index = -1;
    double min_sq_dist = std::numeric_limits<double>::max();

    for (size_t i = 0; i < points.size(); i++) {
      const double sq_dist = (points[i] - pt).squaredNorm();
      if (sq_dist < min_sq_dist) {
        min_index = i;
        min_sq_dist = sq_dist;
      }
    }

    *k_index = min_index;
    *k_sq_dist = min_sq_dist;

    return 1;
  }

  /// @brief Find k nearest neighbors
  /// @param pt           Query point
  /// @param k            Number of neighbors
  /// @param k_index      Indices of nearest neighbors
  /// @param k_sq_dist    Squared distances to nearest neighbors
  /// @return             Number of found points
  size_t knn_search(const Eigen::Vector4d& pt, int k, size_t* k_index, double* k_sq_dist) const {
    if (points.empty()) {
      return 0;
    }

    std::priority_queue<std::pair<size_t, double>> queue;
    for (size_t i = 0; i < points.size(); i++) {
      const double sq_dist = (points[i] - pt).squaredNorm();
      queue.push({i, sq_dist});
      if (queue.size() > k) {
        queue.pop();
      }
    }

    const size_t n = queue.size();
    while (!queue.empty()) {
      k_index[queue.size() - 1] = queue.top().first;
      k_sq_dist[queue.size() - 1] = queue.top().second;
      queue.pop();
    }

    return n;
  }

public:
  struct Empty {};

  std::vector<Eigen::Vector4d> points;
  std::conditional_t<HasNormals, std::vector<Eigen::Vector4d>, Empty> normals;
  std::conditional_t<HasCovs, std::vector<Eigen::Matrix4d>, Empty> covs;
};

using FlatContainerPoints = FlatContainer<false, false>;
using FlatContainerNormal = FlatContainer<true, false>;
using FlatContainerCov = FlatContainer<false, true>;
using FlatContainerNormalCov = FlatContainer<true, true>;

namespace traits {

template <bool HasNormals, bool HasCovs>
struct Traits<FlatContainer<HasNormals, HasCovs>> {
  static size_t size(const FlatContainer<HasNormals, HasCovs>& container) { return container.size(); }
  static bool has_points(const FlatContainer<HasNormals, HasCovs>& container) { return container.size(); }
  static bool has_normals(const FlatContainer<HasNormals, HasCovs>& container) { return HasNormals && container.size(); }
  static bool has_covs(const FlatContainer<HasNormals, HasCovs>& container) { return HasCovs && container.size(); }

  static const Eigen::Vector4d& point(const FlatContainer<HasNormals, HasCovs>& container, size_t i) { return container.points[i]; }
  static const Eigen::Vector4d& normal(const FlatContainer<HasNormals, HasCovs>& container, size_t i) { return container.normals[i]; }
  static const Eigen::Matrix4d& cov(const FlatContainer<HasNormals, HasCovs>& container, size_t i) { return container.covs[i]; }

  static size_t nearest_neighbor_search(const FlatContainer<HasNormals, HasCovs>& container, const Eigen::Vector4d& pt, size_t* k_index, double* k_sq_dist) {
    return container.nearest_neighbor_search(pt, k_index, k_sq_dist);
  }

  static size_t knn_search(const FlatContainer<HasNormals, HasCovs>& container, const Eigen::Vector4d& pt, size_t k, size_t* k_index, double* k_sq_dist) {
    return container.knn_search(pt, k, k_index, k_sq_dist);
  }
};

}  // namespace traits

}  // namespace small_gicp
