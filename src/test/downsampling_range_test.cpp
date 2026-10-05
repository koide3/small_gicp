// SPDX-FileCopyrightText: Copyright 2026 Jinhang Dong
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/downsampling_tbb.hpp>

using namespace small_gicp;

class DownsamplingRangeTest : public testing::TestWithParam<std::string> {
public:
  PointCloud::Ptr downsample(const PointCloud& points) {
    const auto method = GetParam();
    if (method == "SMALL") {
      return voxelgrid_sampling(points, 1.0);
    }
    if (method == "OMP1" || method == "OMP4") {
      return voxelgrid_sampling_omp(points, 1.0, method == "OMP1" ? 1 : 4);
    }
    if (method == "TBB") {
      return voxelgrid_sampling_tbb(points, 1.0);
    }
    throw std::runtime_error("Invalid method: " + method);
  }
};

INSTANTIATE_TEST_SUITE_P(Backends, DownsamplingRangeTest, testing::Values("SMALL", "OMP1", "OMP4", "TBB"), [](const auto& info) { return info.param; });

TEST_P(DownsamplingRangeTest, EmptyInput) {
  const auto result = downsample(PointCloud());
  ASSERT_TRUE(result);
  EXPECT_TRUE(result->empty());
}

TEST_P(DownsamplingRangeTest, AllPointsOutOfRange) {
  PointCloud points;
  points.resize(2);
  points.point(0) << 1048576.0, 0.0, 0.0, 1.0;
  points.point(1) << -1048577.0, 0.0, 0.0, 1.0;
  points.color(0).setZero();
  points.color(1).setZero();

  const auto result = downsample(points);
  ASSERT_TRUE(result);
  EXPECT_TRUE(result->empty());
}

TEST_P(DownsamplingRangeTest, InvalidOnlyBlocksAreIgnored) {
  PointCloud points;
  points.resize(4098);
  for (size_t i = 0; i < points.size(); i++) {
    points.point(i) << 1048576.0, 0.0, 0.0, 1.0;
    points.color(i) << 0.0, 0.0, 1.0, 0.0;
  }
  points.point(0) << 0.25, 0.25, 0.25, 1.0;
  points.point(1) << 0.75, 0.75, 0.75, 1.0;
  points.color(0) << 1.0, 0.0, 0.0, 0.0;
  points.color(1) << 0.0, 1.0, 0.0, 0.0;

  const auto result = downsample(points);
  ASSERT_TRUE(result);
  ASSERT_EQ(result->size(), 1);
  EXPECT_TRUE(result->point(0).isApprox(Eigen::Vector4d(0.5, 0.5, 0.5, 1.0)));
  EXPECT_TRUE(result->color(0).isApprox(Eigen::Vector4d(0.5, 0.5, 0.0, 0.0)));
}

TEST_P(DownsamplingRangeTest, ValidCoordinateLimitsAreKept) {
  PointCloud points;
  points.resize(2);
  points.point(0) << -1048576.0, 0.0, 0.0, 1.0;
  points.point(1) << 1048575.0, 0.0, 0.0, 1.0;
  points.color(0).setZero();
  points.color(1).setZero();

  const auto result = downsample(points);
  ASSERT_TRUE(result);
  ASSERT_EQ(result->size(), 2);
  for (const auto& expected : points.points) {
    EXPECT_TRUE(std::any_of(result->points.begin(), result->points.end(), [&](const auto& point) { return point.isApprox(expected); }));
  }
}
