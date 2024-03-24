#pragma once

#include <atomic>
#include <tbb/tbb.h>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/util/fast_floor.hpp>
#include <small_gicp/util/vector3i_hash.hpp>

namespace small_gicp {

/// @brief KdTree with voxel-based caching.
/// @note  This class is usually useless.
template <typename PointCloud>
class CachedKdTree {
public:
  using Ptr = std::shared_ptr<CachedKdTree>;
  using ConstPtr = std::shared_ptr<const CachedKdTree>;

  /// @brief Constructor
  /// @param points     Input points
  /// @param leaf_size  Cache voxel resolution
  CachedKdTree(const PointCloud& points, double leaf_size) : inv_leaf_size(1.0 / leaf_size), kdtree(points) {}

  size_t knn_search(const Eigen::Vector4d& pt, size_t k, size_t* k_indices, double* k_sq_dists) const {
    const Eigen::Vector3i coord = fast_floor(pt * inv_leaf_size).head<3>();

    CacheTable::const_accessor ca;
    if (cache.find(ca, coord)) {
      std::ranges::copy(ca->second.first, k_indices);
      std::ranges::copy(ca->second.second, k_sq_dists);
      return ca->second.first.size();
    }

    const size_t n = kdtree.knn_search(pt, k, k_indices, k_sq_dists);
    std::vector<size_t> indices(k_indices, k_indices + n);
    std::vector<double> sq_dists(k_sq_dists, k_sq_dists + n);

    CacheTable::accessor a;
    if (cache.insert(a, coord)) {
      a->second.first = std::move(indices);
      a->second.second = std::move(sq_dists);
    }

    return n;
  }

public:
  const double inv_leaf_size;

  using KnnResult = std::pair<std::vector<size_t>, std::vector<double>>;
  using CacheTable = tbb::concurrent_hash_map<Eigen::Vector3i, KnnResult, XORVector3iHash>;
  mutable CacheTable cache;

  UnsafeKdTree<PointCloud> kdtree;
};

namespace traits {

template <typename PointCloud>
struct Traits<CachedKdTree<PointCloud>> {
  static size_t knn_search(const CachedKdTree<PointCloud>& tree, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
    return tree.knn_search(point, k, k_indices, k_sq_dists);
  }
};

}  // namespace traits

}  // namespace small_gicp
