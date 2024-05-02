#include <unordered_set>

#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/downsampling_tbb.hpp>
#include <small_gicp/benchmark/read_points.hpp>

using namespace small_gicp;

class DownsamplingTest : public testing::Test, public testing::WithParamInterface<std::string> {
public:
  void SetUp() override {
    // Load points
    auto points_4f = read_ply("data/target.ply");
    points = std::make_shared<PointCloud>(points_4f);
    points_pcl = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    points_pcl->resize(points_4f.size());
    for (size_t i = 0; i < points_4f.size(); i++) {
      points_pcl->at(i).getVector4fMap() = points_4f[i];
    }

    // Downsample points using pcl::VoxelGrid for reference
    resolutions = {0.1, 0.5, 1.0};
    for (double resolution : resolutions) {
      pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
      voxelgrid.setLeafSize(resolution, resolution, resolution);
      voxelgrid.setInputCloud(points_pcl);

      auto downsampled = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      voxelgrid.filter(*downsampled);
      downsampled_pcl.push_back(downsampled);
    }
  }

  // Apply downsampling
  template <typename PointCloud>
  std::shared_ptr<PointCloud> downsample(const PointCloud& points, double resolution) {
    const std::string method = GetParam();
    if (method == "SMALL") {
      return voxelgrid_sampling(points, resolution);
    } else if (method == "TBB") {
      return voxelgrid_sampling_tbb(points, resolution);
    } else if (method == "OMP") {
      return voxelgrid_sampling_omp(points, resolution);
    } else {
      throw std::runtime_error("Invalid method: " + method);
    }
  }

protected:
  std::vector<double> resolutions;                                   ///< Downsampling resolutions
  PointCloud::Ptr points;                                            ///< Input points
  pcl::PointCloud<pcl::PointXYZ>::Ptr points_pcl;                    ///< Input points (pcl)
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> downsampled_pcl;  ///< Reference downsampling results (pcl)
};

// Load check
TEST_F(DownsamplingTest, LoadCheck) {
  EXPECT_FALSE(points->empty());
  EXPECT_FALSE(points_pcl->empty());
  for (auto downsampled : downsampled_pcl) {
    EXPECT_FALSE(downsampled->empty());
  }
}

INSTANTIATE_TEST_SUITE_P(DownsamplingTest, DownsamplingTest, testing::Values("SMALL", "TBB", "OMP"), [](const auto& info) { return info.param; });

// Check if downsampling works correctly for empty points
TEST_P(DownsamplingTest, EmptyTest) {
  auto empty_points = std::make_shared<PointCloud>();
  auto empty_downsampled = downsample(*empty_points, 0.1);
  EXPECT_TRUE(empty_downsampled);
  EXPECT_EQ(empty_downsampled->size(), 0) << "Empty test small: " + GetParam();

  auto empty_points_pcl = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  auto empty_downsampled_pcl = downsample(*empty_points_pcl, 0.1);
  EXPECT_TRUE(empty_downsampled_pcl);
  EXPECT_EQ(empty_downsampled_pcl->size(), 0) << "Empty test pcl: " + GetParam();
}

// Check if downsampling results are mostly identical to those of pcl::VoxelGrid
TEST_P(DownsamplingTest, DownsampleTest) {
  for (size_t i = 0; i < resolutions.size(); i++) {
    auto result = downsample(*points, resolutions[i]);
    EXPECT_LT(std::abs(1.0 - static_cast<double>(result->size()) / downsampled_pcl[i]->size()), 0.9) << "Downsampled size check (small): " + GetParam();
    auto result_pcl = downsample(*points_pcl, resolutions[i]);
    EXPECT_LT(std::abs(1.0 - static_cast<double>(result_pcl->size()) / downsampled_pcl[i]->size()), 0.9) << "Downsampled size check (pcl): " + GetParam();
    EXPECT_EQ(result->size(), result_pcl->size()) << "Size check (small vs pcl): " + GetParam();
  }
}

// Check if random sampling works correctly for empty points
TEST_P(DownsamplingTest, EmptyRandamSamplingTest) {
  std::mt19937 mt;

  auto empty_points = std::make_shared<PointCloud>();
  auto empty_downsampled = random_sampling(*empty_points, 1000, mt);
  EXPECT_TRUE(empty_downsampled);
  EXPECT_EQ(empty_downsampled->size(), 0) << "Empty test small: " + GetParam();

  auto empty_points_pcl = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  auto empty_downsampled_pcl = random_sampling(*empty_points_pcl, 1000, mt);
  EXPECT_TRUE(empty_downsampled_pcl);
  EXPECT_EQ(empty_downsampled_pcl->size(), 0) << "Empty test pcl: " + GetParam();
}

// Test random sampling
TEST_P(DownsamplingTest, RandamSamplingTest) {
  std::mt19937 mt;

  auto downsampled = voxelgrid_sampling(*points, 0.1);

  const std::vector<size_t> num_points = {0, 100, 1000};
  for (size_t N : num_points) {
    auto result = random_sampling(*downsampled, N, mt);
    EXPECT_TRUE(result);
    EXPECT_EQ(result->size(), N) << "Size check (small): " + GetParam();

    std::unordered_set<size_t> indices;
    for (size_t i = 0; i < N; i++) {
      const auto found = std::find_if(downsampled->points.begin(), downsampled->points.end(), [&](const auto& p) { return (p - result->points[i]).norm() < 1.0e-6; });
      EXPECT_NE(found, downsampled->points.end()) << "Existence check (small): " + GetParam();

      const size_t index = std::distance(downsampled->points.begin(), found);
      EXPECT_EQ(indices.count(index), 0) << "Uniqueness check (small): " + GetParam();
      indices.insert(index);
    }
  }
}
