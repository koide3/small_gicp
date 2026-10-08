// SPDX-FileCopyrightText: Copyright 2026 Yong Ling
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/ann/kdtree_tbb.hpp>
#include <small_gicp/points/point_cloud.hpp>

template <typename Builder>
void check_empty_search(const Builder& builder) {
  const small_gicp::PointCloud points;
  const small_gicp::UnsafeKdTree<small_gicp::PointCloud> tree(points, builder);
  const Eigen::Vector4d query(0.0, 0.0, 0.0, 1.0);
  size_t indices[3];
  double distances[3];
  EXPECT_EQ(tree.nearest_neighbor_search(query, indices, distances), 0);
  EXPECT_EQ(tree.knn_search(query, 3, indices, distances), 0);
  EXPECT_EQ(tree.knn_search<3>(query, indices, distances), 0);
}

TEST(EmptyKdTree, SerialBuilder) {
  check_empty_search(small_gicp::KdTreeBuilder());
}
TEST(EmptyKdTree, OpenMPBuilder) {
  check_empty_search(small_gicp::KdTreeBuilderOMP());
}
TEST(EmptyKdTree, TBBBuilder) {
  check_empty_search(small_gicp::KdTreeBuilderTBB());
}
