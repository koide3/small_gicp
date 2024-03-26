#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/points/pcl_point_traits.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/downsampling_tbb.hpp>

using namespace small_gicp;

TEST(DownsamplingTest, NullTest) {
  auto points = std::make_shared<small_gicp::PointCloud>();
  EXPECT_EQ(small_gicp::voxelgrid_sampling(*points, 0.1)->size(), 0);
  EXPECT_EQ(small_gicp::voxelgrid_sampling_omp(*points, 0.1)->size(), 0);
  EXPECT_EQ(small_gicp::voxelgrid_sampling_tbb(*points, 0.1)->size(), 0);

  auto points_pcl = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  EXPECT_EQ(small_gicp::voxelgrid_sampling(*points_pcl, 0.1)->size(), 0);
  EXPECT_EQ(small_gicp::voxelgrid_sampling_omp(*points_pcl, 0.1)->size(), 0);
  EXPECT_EQ(small_gicp::voxelgrid_sampling_tbb(*points_pcl, 0.1)->size(), 0);
}
