// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT

// While the following KdTree code is written from scratch, it is heavily inspired by the nanoflann library.
// Thus, the following original license of nanoflann is included to be sure.

// https://github.com/jlblancoc/nanoflann/blob/master/include/nanoflann.hpp
/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 * Copyright 2011-2024  Jose Luis Blanco (joseluisblancoc@gmail.com).
 *   All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/
#pragma once

#include <memory>
#include <numeric>
#include <Eigen/Core>

#include <small_gicp/points/traits.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/ann/projection.hpp>
#include <small_gicp/ann/knn_result.hpp>

namespace small_gicp {

using NodeIndexType = std::uint32_t;
static constexpr NodeIndexType INVALID_NODE = std::numeric_limits<NodeIndexType>::max();

/// @brief KdTree node.
template <typename Projection>
struct KdTreeNode {
  union {
    struct Leaf {
      NodeIndexType first;  ///< First point index in the leaf node.
      NodeIndexType last;   ///< Last point index in the leaf node.
    } lr;                   ///< Leaf node.
    struct NonLeaf {
      Projection proj;  ///< Projection axis.
      double thresh;    ///< Threshold value.
    } sub;              ///< Non-leaf node.
  } node_type;

  NodeIndexType left = INVALID_NODE;   ///< Left child node index.
  NodeIndexType right = INVALID_NODE;  ///< Right child node index.
};

/// @brief Single thread Kd-tree builder.
struct KdTreeBuilder {
public:
  /// @brief Build KdTree
  /// @param kdtree         Kd-tree to build
  /// @param points         Point cloud
  template <typename KdTree, typename PointCloud>
  void build_tree(KdTree& kdtree, const PointCloud& points) const {
    kdtree.indices.resize(traits::size(points));
    std::iota(kdtree.indices.begin(), kdtree.indices.end(), 0);

    size_t node_count = 0;
    kdtree.nodes.resize(traits::size(points));
    kdtree.root = create_node(kdtree, node_count, points, kdtree.indices.begin(), kdtree.indices.begin(), kdtree.indices.end());
    kdtree.nodes.resize(node_count);
  }

  /// @brief Create a Kd-tree node from the given point indices.
  /// @param global_first     Global first point index iterator (i.e., this->indices.begin()).
  /// @param first            First point index iterator to be scanned.
  /// @param last             Last point index iterator to be scanned.
  /// @return                 Index of the created node.
  template <typename PointCloud, typename KdTree, typename IndexConstIterator>
  NodeIndexType create_node(KdTree& kdtree, size_t& node_count, const PointCloud& points, IndexConstIterator global_first, IndexConstIterator first, IndexConstIterator last)
    const {
    const size_t N = std::distance(first, last);
    const NodeIndexType node_index = node_count++;
    auto& node = kdtree.nodes[node_index];

    // Create a leaf node.
    if (N <= max_leaf_size) {
      // std::sort(first, last);
      node.node_type.lr.first = std::distance(global_first, first);
      node.node_type.lr.last = std::distance(global_first, last);

      return node_index;
    }

    // Find the best axis to split the input points.
    using Projection = typename KdTree::Projection;
    const auto proj = Projection::find_axis(points, first, last, projection_setting);
    const auto median_itr = first + N / 2;
    std::nth_element(first, median_itr, last, [&](size_t i, size_t j) { return proj(traits::point(points, i)) < proj(traits::point(points, j)); });

    // Create a non-leaf node.
    node.node_type.sub.proj = proj;
    node.node_type.sub.thresh = proj(traits::point(points, *median_itr));

    // Create left and right child nodes.
    node.left = create_node(kdtree, node_count, points, global_first, first, median_itr);
    node.right = create_node(kdtree, node_count, points, global_first, median_itr, last);

    return node_index;
  }

public:
  int max_leaf_size = 20;                ///< Maximum number of points in a leaf node.
  ProjectionSetting projection_setting;  ///< Projection setting.
};

/// @brief "Unsafe" KdTree.
/// @note  This class does not hold the ownership of the input points.
///        You must keep the input points along with this class.
template <typename PointCloud, typename Projection_ = AxisAlignedProjection>
struct UnsafeKdTree {
public:
  using Projection = Projection_;
  using Node = KdTreeNode<Projection>;

  /// @brief Constructor
  /// @param points   Point cloud
  /// @param builder  Kd-tree builder
  template <typename Builder = KdTreeBuilder>
  explicit UnsafeKdTree(const PointCloud& points, const Builder& builder = KdTreeBuilder()) : points(points) {
    if (traits::size(points) == 0) {
      std::cerr << "warning: Empty point cloud" << std::endl;
      return;
    }

    builder.build_tree(*this, points);
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
    knn_search(query, root, result, setting);
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
    knn_search(query, root, result, setting);
    return result.num_found();
  }

private:
  /// @brief Find k-nearest neighbors.
  template <typename Result>
  bool knn_search(const Eigen::Vector4d& query, NodeIndexType node_index, Result& result, const KnnSetting& setting) const {
    const auto& node = nodes[node_index];

    // Check if it's a leaf node.
    if (node.left == INVALID_NODE) {
      // Compare the query point with all points in the leaf node.
      for (size_t i = node.node_type.lr.first; i < node.node_type.lr.last; i++) {
        const double sq_dist = (traits::point(points, indices[i]) - query).squaredNorm();
        result.push(indices[i], sq_dist);
      }
      return !setting.fulfilled(result);
    }

    const double val = node.node_type.sub.proj(query);
    const double diff = val - node.node_type.sub.thresh;
    const double cut_sq_dist = diff * diff;

    NodeIndexType best_child;
    NodeIndexType other_child;

    if (diff < 0.0) {
      best_child = node.left;
      other_child = node.right;
    } else {
      best_child = node.right;
      other_child = node.left;
    }

    // Check the best child node first.
    if (!knn_search(query, best_child, result, setting)) {
      return false;
    }

    // Check if the other child node needs to be tested.
    if (result.worst_distance() > cut_sq_dist) {
      return knn_search(query, other_child, result, setting);
    }

    return true;
  }

public:
  const PointCloud& points;     ///< Input points
  std::vector<size_t> indices;  ///< Point indices refered by nodes

  NodeIndexType root;       ///< Root node index (should be zero)
  std::vector<Node> nodes;  ///< Kd-tree nodes
};

/// @brief "Safe" KdTree that holds the ownership of the input points.
template <typename PointCloud, typename Projection = AxisAlignedProjection>
struct KdTree {
public:
  using Ptr = std::shared_ptr<KdTree<PointCloud, Projection>>;
  using ConstPtr = std::shared_ptr<const KdTree<PointCloud, Projection>>;

  template <typename Builder = KdTreeBuilder>
  explicit KdTree(std::shared_ptr<const PointCloud> points, const Builder& builder = Builder()) : points(points),
                                                                                                  kdtree(*points, builder) {}

  /// @brief  Find k-nearest neighbors. This method uses dynamic memory allocation.
  /// @param  query       Query point
  /// @param  k           Number of neighbors
  /// @param  k_indices   Indices of neighbors
  /// @param  k_sq_dists  Squared distances to neighbors (sorted in ascending order)
  /// @param  setting     KNN search setting
  /// @return             Number of found neighbors
  size_t nearest_neighbor_search(const Eigen::Vector4d& query, size_t* k_indices, double* k_sq_dists, const KnnSetting& setting = KnnSetting()) const {
    return kdtree.nearest_neighbor_search(query, k_indices, k_sq_dists, setting);
  }

  /// @brief  Find k-nearest neighbors. This method uses dynamic memory allocation.
  /// @param  query       Query point
  /// @param  k           Number of neighbors
  /// @param  k_indices   Indices of neighbors
  /// @param  k_sq_dists  Squared distances to neighbors (sorted in ascending order)
  /// @param  setting     KNN search setting
  /// @return             Number of found neighbors
  size_t knn_search(const Eigen::Vector4d& query, size_t k, size_t* k_indices, double* k_sq_dists, const KnnSetting& setting = KnnSetting()) const {
    return kdtree.knn_search(query, k, k_indices, k_sq_dists, setting);
  }

public:
  const std::shared_ptr<const PointCloud> points;     ///< Points
  const UnsafeKdTree<PointCloud, Projection> kdtree;  ///< KdTree
};

namespace traits {

template <typename PointCloud, typename Projection>
struct Traits<UnsafeKdTree<PointCloud, Projection>> {
  static size_t nearest_neighbor_search(const UnsafeKdTree<PointCloud, Projection>& tree, const Eigen::Vector4d& point, size_t* k_indices, double* k_sq_dists) {
    return tree.nearest_neighbor_search(point, k_indices, k_sq_dists);
  }

  static size_t knn_search(const UnsafeKdTree<PointCloud, Projection>& tree, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
    return tree.knn_search(point, k, k_indices, k_sq_dists);
  }
};

template <typename PointCloud, typename Projection>
struct Traits<KdTree<PointCloud, Projection>> {
  static size_t nearest_neighbor_search(const KdTree<PointCloud, Projection>& tree, const Eigen::Vector4d& point, size_t* k_indices, double* k_sq_dists) {
    return tree.nearest_neighbor_search(point, k_indices, k_sq_dists);
  }

  static size_t knn_search(const KdTree<PointCloud, Projection>& tree, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
    return tree.knn_search(point, k, k_indices, k_sq_dists);
  }
};

}  // namespace traits

}  // namespace small_gicp
