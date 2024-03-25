#pragma once

#include <Eigen/Core>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/ann/nanoflann.hpp>
#include <small_gicp/ann/nanoflann_mt.hpp>

namespace small_gicp {

/// @brief Unsafe KdTree with multi-thread tree construction.
/// @note  This class only parallelizes the tree construction. The search is still single-threaded as in the normal KdTree.
template <typename PointCloud>
class UnsafeKdTreeMT {
public:
  using Ptr = std::shared_ptr<UnsafeKdTreeMT>;
  using ConstPtr = std::shared_ptr<const UnsafeKdTreeMT>;
  using Index = nanoflann::KDTreeSingleIndexAdaptorMT<nanoflann::L2_Simple_Adaptor<double, UnsafeKdTreeMT<PointCloud>, double>, UnsafeKdTreeMT<PointCloud>, 3, size_t>;

  /// @brief Constructor
  /// @param points  Input points
  UnsafeKdTreeMT(const PointCloud& points, int num_threads = 4) : points(points), index(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10)) { index.buildIndex(num_threads); }
  ~UnsafeKdTreeMT() {}

  // Interfaces for nanoflann
  size_t kdtree_get_point_count() const { return traits::size(points); }
  double kdtree_get_pt(const size_t idx, const size_t dim) const { return traits::point(points, idx)[dim]; }

  template <class BBox>
  bool kdtree_get_bbox(BBox&) const {
    return false;
  }

  /// @brief Find k-nearest neighbors
  size_t knn_search(const Eigen::Vector4d& pt, size_t k, size_t* k_indices, double* k_sq_dists) const { return index.knnSearch(pt.data(), k, k_indices, k_sq_dists); }

private:
  const PointCloud& points;  ///< Input points
  Index index;               ///< KdTree index
};

/// @brief KdTree
template <typename PointCloud>
class KdTreeMT {
public:
  using Ptr = std::shared_ptr<KdTreeMT>;
  using ConstPtr = std::shared_ptr<const KdTreeMT>;

  /// @brief Constructor
  /// @param points  Input points
  KdTreeMT(const std::shared_ptr<const PointCloud>& points, int num_threads = 4) : points(points), tree(*points, num_threads) {}
  ~KdTreeMT() {}

  /// @brief Find k-nearest neighbors
  size_t knn_search(const Eigen::Vector4d& pt, size_t k, size_t* k_indices, double* k_sq_dists) const { return tree.knn_search(pt, k, k_indices, k_sq_dists); }

private:
  const std::shared_ptr<const PointCloud> points;  ///< Input points
  const UnsafeKdTreeMT<PointCloud> tree;           ///< KdTree
};

namespace traits {

template <typename PointCloud>
struct Traits<UnsafeKdTreeMT<PointCloud>> {
  static size_t knn_search(const UnsafeKdTreeMT<PointCloud>& tree, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
    return tree.knn_search(point, k, k_indices, k_sq_dists);
  }
};

template <typename PointCloud>
struct Traits<KdTreeMT<PointCloud>> {
  static size_t knn_search(const KdTreeMT<PointCloud>& tree, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
    return tree.knn_search(point, k, k_indices, k_sq_dists);
  }
};

}  // namespace traits

}  // namespace small_gicp
