#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/reduction.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/rejector.hpp>

using namespace small_gicp;

namespace {

// Same distance-based rejection as DistanceRejector, but without the `max_dist_sq` member,
// i.e., a rejector that cannot tell the search how far to look.
struct OpaqueDistanceRejector {
  double thr_sq = 1.0;

  template <typename TargetPointCloud, typename SourcePointCloud>
  bool operator()(const TargetPointCloud&, const SourcePointCloud&, const Eigen::Isometry3d&, size_t, size_t, double sq_dist) const {
    return sq_dist > thr_sq;
  }
};

void random_clouds(size_t num_points, size_t num_queries, std::vector<Eigen::Vector4d>& points, std::vector<Eigen::Vector4d>& queries) {
  std::mt19937 mt(42);
  std::uniform_real_distribution<double> udist(-10.0, 10.0);

  points.resize(num_points);
  for (auto& pt : points) {
    pt = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0);
  }

  queries.resize(num_queries);
  for (auto& q : queries) {
    q = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0);
  }
}

}  // namespace

TEST(KnnSettingTest, DefaultSettingIsUnbounded) {
  std::vector<Eigen::Vector4d> points, queries;
  random_clouds(1000, 200, points, queries);

  PointCloud cloud(points);
  UnsafeKdTree<PointCloud> kdtree(cloud);

  for (const auto& q : queries) {
    size_t index0, index1;
    double sq_dist0, sq_dist1;
    kdtree.nearest_neighbor_search(q, &index0, &sq_dist0);
    kdtree.nearest_neighbor_search(q, &index1, &sq_dist1, KnnSetting());

    EXPECT_EQ(index0, index1);
    EXPECT_DOUBLE_EQ(sq_dist0, sq_dist1);
  }
}

TEST(KnnSettingTest, BoundedSearchDoesNotChangeNeighborsWithinBound) {
  std::vector<Eigen::Vector4d> points, queries;
  random_clouds(2000, 500, points, queries);

  PointCloud cloud(points);
  UnsafeKdTree<PointCloud> kdtree(cloud);

  const double max_dist = 0.5;
  KnnSetting setting;
  setting.max_dist_sq = max_dist * max_dist;

  size_t num_inliers = 0;
  size_t num_outliers = 0;
  for (const auto& q : queries) {
    size_t index_full;
    double sq_dist_full;
    kdtree.nearest_neighbor_search(q, &index_full, &sq_dist_full);

    size_t index_bounded;
    double sq_dist_bounded;
    const size_t num_found = kdtree.nearest_neighbor_search(q, &index_bounded, &sq_dist_bounded, setting);

    if (sq_dist_full < setting.max_dist_sq) {
      // A neighbor exists within the bound: the bounded search must return exactly the same one.
      num_inliers++;
      EXPECT_EQ(num_found, 1u);
      EXPECT_EQ(index_full, index_bounded);
      EXPECT_DOUBLE_EQ(sq_dist_full, sq_dist_bounded);
    } else if (sq_dist_full > setting.max_dist_sq) {
      // The nearest neighbor is beyond the bound: nothing must be reported.
      num_outliers++;
      EXPECT_EQ(num_found, 0u);
    }
  }

  EXPECT_GT(num_inliers, 0u);
  EXPECT_GT(num_outliers, 0u);
}

TEST(KnnSettingTest, BoundedKnnSearchRespectsBound) {
  std::vector<Eigen::Vector4d> points, queries;
  random_clouds(2000, 200, points, queries);

  PointCloud cloud(points);
  UnsafeKdTree<PointCloud> kdtree(cloud);

  constexpr int k = 10;
  const double max_dist = 0.5;

  KnnSetting setting;
  setting.max_dist_sq = max_dist * max_dist;

  size_t num_found_total = 0;
  for (const auto& q : queries) {
    size_t k_indices[k];
    double k_sq_dists[k];
    const size_t num_found = kdtree.knn_search<k>(q, k_indices, k_sq_dists, setting);

    for (size_t i = 0; i < num_found; i++) {
      EXPECT_LT(k_sq_dists[i], setting.max_dist_sq);
    }
    num_found_total += num_found;
  }

  EXPECT_GT(num_found_total, 0u);
}

TEST(KnnSettingTest, NearestNeighborSearchUsesRejectorDistanceBound) {
  std::vector<Eigen::Vector4d> points, queries;
  random_clouds(2000, 300, points, queries);

  PointCloud cloud(points);
  UnsafeKdTree<PointCloud> kdtree(cloud);

  DistanceRejector rejector;
  rejector.max_dist_sq = 0.25;

  // A rejector without a distance bound must fall back to the unbounded search.
  OpaqueDistanceRejector opaque_rejector;

  size_t num_found = 0;
  for (const auto& q : queries) {
    size_t index;
    double sq_dist;
    const size_t found = traits::nearest_neighbor_search(kdtree, q, &index, &sq_dist, rejector);
    num_found += found;

    if (found) {
      EXPECT_LE(sq_dist, rejector.max_dist_sq);
    }
  }
  EXPECT_GT(num_found, 0u);

  // Both rejector types must return the same neighbor whenever the correspondence is accepted.
  for (const auto& q : queries) {
    size_t index_a, index_b;
    double sq_dist_a, sq_dist_b;
    const size_t found_a = traits::nearest_neighbor_search(kdtree, q, &index_a, &sq_dist_a, rejector);
    const size_t found_b = traits::nearest_neighbor_search(kdtree, q, &index_b, &sq_dist_b, opaque_rejector);

    if (found_a) {
      ASSERT_EQ(found_b, 1u);
      EXPECT_EQ(index_a, index_b);
      EXPECT_DOUBLE_EQ(sq_dist_a, sq_dist_b);
    }
  }
}

TEST(KnnSettingTest, RegistrationResultIsIndependentOfSearchBound) {
  std::mt19937 mt(7);
  std::uniform_real_distribution<double> udist(-2.0, 2.0);

  std::vector<Eigen::Vector4d> target_points(500);
  for (auto& pt : target_points) {
    pt = Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0);
  }

  // The source is a rigidly transformed copy of half of the target.
  const Eigen::Isometry3d T_gt(Eigen::Translation3d(0.1, -0.05, 0.02) * Eigen::AngleAxisd(0.02, Eigen::Vector3d::UnitZ()));
  std::vector<Eigen::Vector4d> source_points(250);
  for (size_t i = 0; i < source_points.size(); i++) {
    source_points[i] = T_gt.inverse() * target_points[2 * i];
  }

  auto target = std::make_shared<PointCloud>(target_points);
  auto source = std::make_shared<PointCloud>(source_points);
  auto target_tree = std::make_shared<KdTree<PointCloud>>(target);

  DistanceRejector bounded_rejector;
  bounded_rejector.max_dist_sq = 1.0;
  OpaqueDistanceRejector unbounded_rejector;
  unbounded_rejector.thr_sq = 1.0;

  Registration<ICPFactor, SerialReduction, NullFactor, DistanceRejector> bounded_registration;
  bounded_registration.rejector = bounded_rejector;
  const auto bounded_result = bounded_registration.align(*target, *source, *target_tree, Eigen::Isometry3d::Identity());

  Registration<ICPFactor, SerialReduction, NullFactor, OpaqueDistanceRejector> unbounded_registration;
  unbounded_registration.rejector = unbounded_rejector;
  const auto unbounded_result = unbounded_registration.align(*target, *source, *target_tree, Eigen::Isometry3d::Identity());

  EXPECT_EQ(bounded_result.converged, unbounded_result.converged);
  EXPECT_EQ(bounded_result.num_inliers, unbounded_result.num_inliers);
  EXPECT_LT((bounded_result.T_target_source.matrix() - unbounded_result.T_target_source.matrix()).norm(), 1e-9);
}
