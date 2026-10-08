// SPDX-FileCopyrightText: Copyright 2026 Yong Ling
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <small_gicp/ann/incremental_voxelmap.hpp>
#include <small_gicp/points/point_cloud.hpp>

TEST(VoxelSearchOffsets, ReplacingPatternDoesNotDuplicateNeighbors) {
  small_gicp::IncrementalVoxelMap<small_gicp::FlatContainerPoints> map(1.0);
  const small_gicp::PointCloud points(std::vector<Eigen::Vector4d>{Eigen::Vector4d(0.1, 0.2, 0.3, 1.0)});
  map.insert(points);
  for (int pattern : {27, 27, 7, 27, 1, 27}) {
    map.set_search_offsets(pattern);
    EXPECT_EQ(map.search_offsets.size(), pattern);
    size_t indices[3];
    double distances[3];
    EXPECT_EQ(map.knn_search(Eigen::Vector4d(0.1, 0.2, 0.3, 1.0), 3, indices, distances), 1);
    EXPECT_DOUBLE_EQ(distances[0], 0.0);
  }
}
