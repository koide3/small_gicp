#pragma once

#include <Eigen/Core>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/ann/nanoflann.hpp>

namespace small_gicp {

template <typename PointCloud>
class UnsafeKdTree {
public:
  using Ptr = std::shared_ptr<UnsafeKdTree>;
  using ConstPtr = std::shared_ptr<const UnsafeKdTree>;
  using Index = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, UnsafeKdTree<PointCloud>, double>, UnsafeKdTree<PointCloud>, 3, size_t>;

  UnsafeKdTree(const PointCloud& points) : points(points), index(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10)) { index.buildIndex(); }
  ~UnsafeKdTree() {}

  size_t kdtree_get_point_count() const { return traits::size(points); }
  double kdtree_get_pt(const size_t idx, const size_t dim) const { return traits::point(points, idx)[dim]; }

  template <class BBox>
  bool kdtree_get_bbox(BBox&) const {
    return false;
  }

  size_t knn_search(const Eigen::Vector4d& pt, size_t k, size_t* k_indices, double* k_sq_dists) const { return index.knnSearch(pt.data(), k, k_indices, k_sq_dists); }

private:
  const PointCloud& points;
  Index index;
};

template <typename PointCloud>
class KdTree {
public:
  using Ptr = std::shared_ptr<KdTree>;
  using ConstPtr = std::shared_ptr<const KdTree>;

  KdTree(const std::shared_ptr<const PointCloud>& points) : points(points), tree(*points) {}
  ~KdTree() {}

  size_t knn_search(const Eigen::Vector4d& pt, size_t k, size_t* k_indices, double* k_sq_dists) const { return tree.knn_search(pt, k, k_indices, k_sq_dists); }

private:
  const std::shared_ptr<const PointCloud> points;
  const UnsafeKdTree<PointCloud> tree;
};

namespace traits {

template <typename PointCloud>
struct Traits<UnsafeKdTree<PointCloud>> {
  static size_t knn_search(const UnsafeKdTree<PointCloud>& tree, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
    return tree.knn_search(point, k, k_indices, k_sq_dists);
  }
};

template <typename PointCloud>
struct Traits<KdTree<PointCloud>> {
  static size_t knn_search(const KdTree<PointCloud>& tree, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
    return tree.knn_search(point, k, k_indices, k_sq_dists);
  }
};

}  // namespace traits

}  // namespace small_gicp
