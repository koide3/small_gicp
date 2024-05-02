// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <atomic>
#include <tbb/tbb.h>
#include <small_gicp/ann/kdtree.hpp>

namespace small_gicp {

/// @brief Kd-tree builder with TBB.
struct KdTreeBuilderTBB {
public:
  /// @brief Build KdTree
  template <typename KdTree, typename PointCloud>
  void build_tree(KdTree& kdtree, const PointCloud& points) const {
    kdtree.indices.resize(traits::size(points));
    std::iota(kdtree.indices.begin(), kdtree.indices.end(), 0);

    std::atomic_uint64_t node_count = 0;
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
  NodeIndexType create_node(
    KdTree& kdtree,
    std::atomic_uint64_t& node_count,
    const PointCloud& points,
    IndexConstIterator global_first,
    IndexConstIterator first,
    IndexConstIterator last) const {
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
    if (N > 512) {
      tbb::parallel_invoke(
        [&] { node.left = create_node(kdtree, node_count, points, global_first, first, median_itr); },
        [&] { node.right = create_node(kdtree, node_count, points, global_first, median_itr, last); });
    } else {
      node.left = create_node(kdtree, node_count, points, global_first, first, median_itr);
      node.right = create_node(kdtree, node_count, points, global_first, median_itr, last);
    }

    return node_index;
  }

public:
  int max_leaf_size = 20;                ///< Maximum number of points in a leaf node.
  ProjectionSetting projection_setting;  ///< Projection setting.
};

}  // namespace small_gicp
