#include <gtest/gtest.h>

#include <random>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>

using namespace small_gicp;

template <typename PointCloud>
void test_points(const std::vector<Eigen::Vector4d>& src_points, PointCloud& points, std::mt19937& mt) {
  EXPECT_EQ(traits::size(points), src_points.size());
  EXPECT_TRUE(traits::has_points(points));
  EXPECT_TRUE(traits::has_normals(points));
  EXPECT_TRUE(traits::has_covs(points));

  std::uniform_real_distribution<> dist(-100.0, 100.0);
  std::vector<Eigen::Vector4d> normals(src_points.size());
  std::vector<Eigen::Matrix4d> covs(src_points.size());
  for (size_t i = 0; i < traits::size(points); i++) {
    normals[i] = Eigen::Vector4d(dist(mt), dist(mt), dist(mt), dist(mt));
    covs[i] = normals[i] * normals[i].transpose();
  }

  for (size_t i = 0; i < traits::size(points); i++) {
    traits::set_normal(points, i, normals[i]);
    traits::set_cov(points, i, covs[i]);

    EXPECT_NEAR((traits::point(points, i) - src_points[i]).norm(), 0.0, 1e-3);
    EXPECT_NEAR((traits::normal(points, i) - normals[i]).norm(), 0.0, 1e-3);
    EXPECT_NEAR((traits::cov(points, i) - covs[i]).norm(), 0.0, 1e-3);
  }

  traits::resize(points, src_points.size() / 2);
  EXPECT_EQ(traits::size(points), src_points.size() / 2);

  for (size_t i = 0; i < traits::size(points); i++) {
    EXPECT_NEAR((traits::point(points, i) - src_points[i]).norm(), 0.0, 1e-3);
    EXPECT_NEAR((traits::normal(points, i) - normals[i]).norm(), 0.0, 1e-3);
    EXPECT_NEAR((traits::cov(points, i) - covs[i]).norm(), 0.0, 1e-3);
  }
}

TEST(PointsTest, PointsTest) {
  std::mt19937 mt;
  std::uniform_real_distribution<> dist(-100.0, 100.0);

  std::vector<Eigen::Vector4d> src_points(100);
  std::generate(src_points.begin(), src_points.end(), [&] { return Eigen::Vector4d(dist(mt), dist(mt), dist(mt), 1.0); });

  auto points = std::make_shared<PointCloud>(src_points);
  test_points(src_points, *points, mt);

  auto points_pcl = pcl::make_shared<pcl::PointCloud<pcl::PointNormalCovariance>>();
  points_pcl->resize(src_points.size());
  for (size_t i = 0; i < src_points.size(); i++) {
    points_pcl->at(i).getVector4fMap() = src_points[i].cast<float>();
  }
  test_points(src_points, *points_pcl, mt);
}