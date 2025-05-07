// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <Eigen/Core>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/ann/knn_result.hpp>

namespace small_gicp {

/// @brief Equirectangular projection.
struct EquirectangularProjection {
public:
  /// @brief Project the point into the normalized image coordinates. (u, v) in ([0, 1], [0, 1])
  Eigen::Vector2d operator()(const Eigen::Vector3d& pt_3d) const {
    if (pt_3d.squaredNorm() < 1e-3) {
      return Eigen::Vector2d(0.5, 0.5);
    }

    const Eigen::Vector3d bearing = pt_3d.normalized();
    const double lat = -std::asin(bearing[1]);
    const double lon = std::atan2(bearing[0], bearing[2]);

    return Eigen::Vector2d(lon / (2.0 * M_PI) + 0.5, lat / M_PI + 0.5);
  };
};

/// @brief Border clamp mode. Points out of the border are discarded.
struct BorderClamp {
public:
  int operator()(int x, int width) const { return x; }
};

/// @brief Border repeat mode. Points out of the border are wrapped around.
struct BorderRepeat {
public:
  int operator()(int x, int width) const { return x < 0 ? x + width : (x >= width ? x - width : x); }
};

/// @brief "Unsafe" projective search. This class does not hold the ownership of the target point cloud.
template <typename PointCloud, typename Projection = EquirectangularProjection, typename BorderModeH = BorderRepeat, typename BorderModeV = BorderClamp>
struct UnsafeProjectiveSearch {
public:
  /// @brief Constructor.
  /// @param width    Index map width
  /// @param height   Index map height
  /// @param points   Target point cloud
  UnsafeProjectiveSearch(int width, int height, const PointCloud& points) : points(points), index_map(height, width), search_window_h(10), search_window_v(5) {
    index_map.setConstant(invalid_index);

    Projection project;
    for (size_t i = 0; i < traits::size(points); ++i) {
      const Eigen::Vector4d pt = traits::point(points, i);
      const Eigen::Vector2d uv = project(pt.head<3>());
      const int u = uv[0] * index_map.cols();
      const int v = uv[1] * index_map.rows();

      if (u < 0 || u >= index_map.cols() || v < 0 || v >= index_map.rows()) {
        continue;
      }
      index_map(v, u) = i;
    }
  }

  /// @brief Find the nearest neighbor.
  /// @param query        Query point
  /// @param k_indices    Index of the nearest neighbor (uninitialized if not found)
  /// @param k_sq_dists   Squared distance to the nearest neighbor (uninitialized if not found)
  /// @param setting      KNN search setting
  /// @return             Number of found neighbors (0 or 1)
  size_t nearest_neighbor_search(const Eigen::Vector4d& query, size_t* k_indices, double* k_sq_dists, const KnnSetting& setting = KnnSetting()) const {
    return knn_search<1>(query, k_indices, k_sq_dists, setting);
  }

  /// @brief  Find k-nearest neighbors. This method uses dynamic memory allocation.
  /// @param  query       Query point
  /// @param  k           Number of neighbors
  /// @param  k_indices   Indices of neighbors
  /// @param  k_sq_dists  Squared distances to neighbors (sorted in ascending order)
  /// @param  setting     KNN search setting
  /// @return             Number of found neighbors
  size_t knn_search(const Eigen::Vector4d& query, int k, size_t* k_indices, double* k_sq_dists, const KnnSetting& setting = KnnSetting()) const {
    KnnResult<-1> result(k_indices, k_sq_dists, k);
    knn_search(query, result, setting);
    return result.num_found();
  }

  /// @brief Find k-nearest neighbors. This method uses fixed and static memory allocation. Might be faster for small k.
  /// @param query       Query point
  /// @param k_indices   Indices of neighbors
  /// @param k_sq_dists  Squared distances to neighbors (sorted in ascending order)
  /// @param setting     KNN search setting
  /// @return            Number of found neighbors
  template <int N>
  size_t knn_search(const Eigen::Vector4d& query, size_t* k_indices, double* k_sq_dists, const KnnSetting& setting = KnnSetting()) const {
    KnnResult<N> result(k_indices, k_sq_dists);
    knn_search(query, result, setting);
    return result.num_found();
  }

private:
  template <typename Result>
  void knn_search(const Eigen::Vector4d& query, Result& result, const KnnSetting& setting) const {
    BorderModeH border_h;
    BorderModeV border_v;

    Projection project;
    const Eigen::Vector2d uv = project(query.head<3>());
    const int u = uv[0] * index_map.cols();
    const int v = uv[1] * index_map.rows();

    for (int du = -search_window_h; du <= search_window_h; du++) {
      const int u_clamped = border_h(u + du, index_map.cols());
      if (u_clamped < 0 || u_clamped >= index_map.cols()) {
        continue;
      }

      for (int dv = -search_window_v; dv <= search_window_v; dv++) {
        const int v_clamped = border_v(v + dv, index_map.rows());
        if (v_clamped < 0 || v_clamped >= index_map.rows()) {
          continue;
        }

        const auto index = index_map(v_clamped, u_clamped);
        if (index == invalid_index) {
          continue;
        }

        const double sq_dist = (traits::point(points, index) - query).squaredNorm();
        result.push(index, sq_dist);

        if (setting.fulfilled(result)) {
          return;
        }
      }
    }
  }

public:
  static constexpr std::uint32_t invalid_index = std::numeric_limits<std::uint32_t>::max();

  const PointCloud& points;
  Eigen::Matrix<std::uint32_t, -1, -1> index_map;

  int search_window_h;
  int search_window_v;
};

/// @brief "Safe" projective search. This class holds the ownership of the target point cloud.
template <typename PointCloud, typename Projection = EquirectangularProjection, typename BorderModeH = BorderRepeat, typename BorderModeV = BorderClamp>
struct ProjectiveSearch {
public:
  using Ptr = std::shared_ptr<ProjectiveSearch<PointCloud, Projection>>;
  using ConstPtr = std::shared_ptr<const ProjectiveSearch<PointCloud, Projection>>;

  explicit ProjectiveSearch(int width, int height, std::shared_ptr<const PointCloud> points) : points(points), search(width, height, *points) {}

  /// @brief  Find k-nearest neighbors. This method uses dynamic memory allocation.
  /// @param  query       Query point
  /// @param  k           Number of neighbors
  /// @param  k_indices   Indices of neighbors
  /// @param  k_sq_dists  Squared distances to neighbors (sorted in ascending order)
  /// @param  setting     KNN search setting
  /// @return             Number of found neighbors
  size_t nearest_neighbor_search(const Eigen::Vector4d& query, size_t* k_indices, double* k_sq_dists, const KnnSetting& setting = KnnSetting()) const {
    return search.nearest_neighbor_search(query, k_indices, k_sq_dists, setting);
  }

  /// @brief  Find k-nearest neighbors. This method uses dynamic memory allocation.
  /// @param  query       Query point
  /// @param  k           Number of neighbors
  /// @param  k_indices   Indices of neighbors
  /// @param  k_sq_dists  Squared distances to neighbors (sorted in ascending order)
  /// @param  setting     KNN search setting
  /// @return             Number of found neighbors
  size_t knn_search(const Eigen::Vector4d& query, size_t k, size_t* k_indices, double* k_sq_dists, const KnnSetting& setting = KnnSetting()) const {
    return search.knn_search(query, k, k_indices, k_sq_dists, setting);
  }

public:
  const std::shared_ptr<const PointCloud> points;                                         ///< Points
  const UnsafeProjectiveSearch<PointCloud, Projection, BorderModeH, BorderModeV> search;  ///< Search
};

namespace traits {

template <typename PointCloud, typename Projection, typename BorderModeH, typename BorderModeV>
struct Traits<UnsafeProjectiveSearch<PointCloud, Projection, BorderModeH, BorderModeV>> {
  static size_t nearest_neighbor_search(
    const UnsafeProjectiveSearch<PointCloud, Projection, BorderModeH, BorderModeV>& tree,
    const Eigen::Vector4d& point,
    size_t* k_indices,
    double* k_sq_dists) {
    return tree.nearest_neighbor_search(point, k_indices, k_sq_dists);
  }

  static size_t
  knn_search(const UnsafeProjectiveSearch<PointCloud, Projection, BorderModeH, BorderModeV>& tree, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
    return tree.knn_search(point, k, k_indices, k_sq_dists);
  }
};

template <typename PointCloud, typename Projection, typename BorderModeH, typename BorderModeV>
struct Traits<ProjectiveSearch<PointCloud, Projection, BorderModeH, BorderModeV>> {
  static size_t
  nearest_neighbor_search(const ProjectiveSearch<PointCloud, Projection, BorderModeH, BorderModeV>& tree, const Eigen::Vector4d& point, size_t* k_indices, double* k_sq_dists) {
    return tree.nearest_neighbor_search(point, k_indices, k_sq_dists);
  }

  static size_t
  knn_search(const ProjectiveSearch<PointCloud, Projection, BorderModeH, BorderModeV>& tree, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
    return tree.knn_search(point, k, k_indices, k_sq_dists);
  }
};

}  // namespace traits

}  // namespace small_gicp
