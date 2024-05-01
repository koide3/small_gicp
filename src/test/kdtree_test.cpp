#include <random>

#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/ann/kdtree_tbb.hpp>
#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/ann/incremental_voxelmap.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/normal_estimation.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/benchmark/read_points.hpp>

using namespace small_gicp;

class KdTreeTest : public testing::Test, public testing::WithParamInterface<std::string> {
public:
  void SetUp() override {
    // Load points
    auto points_4f = read_ply("data/target.ply");
    points = voxelgrid_sampling(*std::make_shared<PointCloud>(points_4f), 0.5);
    estimate_normals_covariances(*points);

    points_pcl = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    points_pcl->resize(points->size());
    for (size_t i = 0; i < points->size(); i++) {
      points_pcl->at(i).getVector4fMap() = points->point(i).cast<float>();
    }

    // Generate queries
    std::mt19937 mt;
    std::uniform_real_distribution<> udist(0.0, 1.0);
    for (size_t i = 0; i < 50; i++) {
      queries.emplace_back(points->point(mt() % points->size()));
      queries.emplace_back(points->point(mt() % points->size()) + Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 0.0));
      queries.emplace_back(Eigen::Vector4d(udist(mt) * 100.0, udist(mt) * 100.0, udist(mt) * 100.0, 1.0));
    }

    // Find k-nearest neighbors with brute force search
    struct IndexDist {
      bool operator<(const IndexDist& rhs) const { return dist < rhs.dist; }
      size_t index;
      double dist;
    };

    k = 20;
    k_indices.resize(queries.size());
    k_sq_dists.resize(queries.size());
    for (size_t i = 0; i < queries.size(); i++) {
      const auto& query = queries[i];

      std::priority_queue<IndexDist> queue;
      for (size_t pt_idx = 0; pt_idx < points->size(); pt_idx++) {
        const double sq_dist = (points->point(pt_idx) - query).squaredNorm();
        if (queue.size() < k) {
          queue.push({pt_idx, sq_dist});
        } else if (sq_dist < queue.top().dist) {
          queue.pop();
          queue.push({pt_idx, sq_dist});
        }
      }

      std::vector<size_t> indices(queue.size(), 0);
      std::vector<double> dists(queue.size(), 0.0);

      while (!queue.empty()) {
        indices[queue.size() - 1] = queue.top().index;
        dists[queue.size() - 1] = queue.top().dist;
        queue.pop();
      }

      k_indices[i] = std::move(indices);
      k_sq_dists[i] = std::move(dists);
    }
  }

  template <typename PointCloud, typename KdTree>
  void test_kdtree(const PointCloud& points, const KdTree& tree) {
    for (size_t i = 0; i < queries.size(); i++) {
      // k-nearest neighbors search
      const auto& query = queries[i];
      std::vector<size_t> indices(k);
      std::vector<double> sq_dists(k);
      const size_t num_results = traits::knn_search(tree, query, k, indices.data(), sq_dists.data());

      EXPECT_EQ(num_results, k) << "num_neighbors must be k";
      for (size_t j = 0; j < k; j++) {
        EXPECT_EQ(indices[j], k_indices[i][j]);
        EXPECT_NEAR(sq_dists[j], k_sq_dists[i][j], 1e-3);
      }

      // Nearest neighbor search
      size_t k_index;
      double k_sq_dist;
      const size_t found = traits::nearest_neighbor_search(tree, query, &k_index, &k_sq_dist);

      EXPECT_EQ(found, 1) << "num_neighbors must be 1";
      EXPECT_EQ(k_index, k_indices[i][0]);
      EXPECT_NEAR(k_sq_dist, k_sq_dists[i][0], 1e-3);
    }
  }

  template <typename PointCloud, typename VoxelMap>
  void test_voxelmap(const PointCloud& points, const VoxelMap& voxelmap) {
    size_t hit_count = 0;
    for (size_t i = 0; i < queries.size(); i++) {
      // k-nearest neighbors search
      const auto& query = queries[i];
      std::vector<size_t> indices(k);
      std::vector<double> sq_dists(k);
      const size_t num_results = traits::knn_search(voxelmap, query, k, indices.data(), sq_dists.data());

      EXPECT_LE(num_results, k) << "num_neighbors must be less than or equal to k";
      for (size_t j = 0; j < num_results; j++) {
        const Eigen::Vector4d pt = traits::point(voxelmap, indices[j]);
        const double sq_dist = (pt - query).squaredNorm();
        EXPECT_NEAR(sq_dists[j], sq_dist, 1e-3);
      }

      // Nearest neighbor search
      size_t nn_index;
      double nn_sq_dist;
      const size_t found = traits::nearest_neighbor_search(voxelmap, query, &nn_index, &nn_sq_dist);

      EXPECT_LE(found, 1) << "num_neighbors must be less than or equal to 1";
      if (found) {
        const Eigen::Vector4d pt = traits::point(voxelmap, nn_index);
        const double sq_dist = (pt - query).squaredNorm();
        EXPECT_NEAR(nn_sq_dist, sq_dist, 1e-3);
        hit_count++;
      }
    }

    const double net_tp = queries.size() * 2.0 / 3.0;
    EXPECT_GE(hit_count, net_tp * 0.5) << "Hit_count must be greater than or equal to " << net_tp;
  }

protected:
  PointCloud::Ptr points;                          ///< Input points
  pcl::PointCloud<pcl::PointXYZ>::Ptr points_pcl;  ///< Input points (pcl)

  int k;
  std::vector<Eigen::Vector4d> queries;
  std::vector<std::vector<size_t>> k_indices;
  std::vector<std::vector<double>> k_sq_dists;
};

// Load check
TEST_F(KdTreeTest, LoadCheck) {
  EXPECT_NE(points->size(), 0) << "Load check";
  EXPECT_NE(points_pcl->size(), 0) << "Load check";

  EXPECT_NE(queries.size(), 0);
  EXPECT_EQ(queries.size(), k_indices.size());
  EXPECT_EQ(queries.size(), k_sq_dists.size());
  for (size_t i = 0; i < queries.size(); i++) {
    EXPECT_EQ(k_indices[i].size(), k);
    EXPECT_EQ(k_sq_dists[i].size(), k);
    EXPECT_TRUE(std::is_sorted(k_sq_dists[i].begin(), k_sq_dists[i].end())) << "Must be sorted by distance";
  }
}

INSTANTIATE_TEST_SUITE_P(KdTreeTest, KdTreeTest, testing::Values("SMALL", "TBB", "OMP", "IVOX", "GVOX"), [](const auto& info) { return info.param; });

// Check if kdtree works correctly for empty points
TEST_P(KdTreeTest, EmptyTest) {
  auto empty_points = std::make_shared<PointCloud>();
  auto kdtree = std::make_shared<UnsafeKdTree<PointCloud>>(*empty_points);

  auto empty_points_pcl = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  auto kdtree_pcl = std::make_shared<UnsafeKdTree<pcl::PointCloud<pcl::PointXYZ>>>(*empty_points_pcl);
}

// Check if nearest neighbor search results are identical to those of brute force search
TEST_P(KdTreeTest, KnnTest) {
  const auto method = GetParam();
  if (method == "SMALL") {
    auto kdtree = std::make_shared<KdTree<PointCloud>>(points);
    test_kdtree(*points, *kdtree);

    auto kdtree_pcl = std::make_shared<KdTree<pcl::PointCloud<pcl::PointXYZ>>>(points_pcl);
    test_kdtree(*points_pcl, *kdtree_pcl);
  } else if (method == "TBB") {
    auto kdtree = std::make_shared<KdTree<PointCloud>>(points, KdTreeBuilderTBB());
    test_kdtree(*points, *kdtree);

    auto kdtree_pcl = std::make_shared<KdTree<pcl::PointCloud<pcl::PointXYZ>>>(points_pcl, KdTreeBuilderTBB());
    test_kdtree(*points_pcl, *kdtree_pcl);
  } else if (method == "OMP") {
    auto kdtree = std::make_shared<KdTree<PointCloud>>(points, KdTreeBuilderOMP(4));
    test_kdtree(*points, *kdtree);

    auto kdtree_pcl = std::make_shared<KdTree<pcl::PointCloud<pcl::PointXYZ>>>(points_pcl, KdTreeBuilderOMP(4));
    test_kdtree(*points_pcl, *kdtree_pcl);
  } else if (method == "IVOX") {
    auto voxelmap = std::make_shared<IncrementalVoxelMap<FlatContainerNormalCov>>(1.0);
    voxelmap->insert(*points);
    test_voxelmap(*points, *voxelmap);

    auto indices = traits::point_indices(*voxelmap);
    auto voxel_points = traits::voxel_points(*voxelmap);
    auto voxel_normals = traits::voxel_normals(*voxelmap);
    auto voxel_covs = traits::voxel_covs(*voxelmap);

    EXPECT_EQ(indices.size(), voxel_points.size());
    EXPECT_EQ(indices.size(), voxel_normals.size());
    EXPECT_EQ(indices.size(), voxel_covs.size());

    for (size_t i = 0; i < indices.size(); i++) {
      EXPECT_NEAR((voxel_points[i] - traits::point(*voxelmap, indices[i])).squaredNorm(), 0.0, 1e-6);
      EXPECT_NEAR((voxel_normals[i] - traits::normal(*voxelmap, indices[i])).squaredNorm(), 0.0, 1e-6);
      EXPECT_NEAR((voxel_covs[i] - traits::cov(*voxelmap, indices[i])).squaredNorm(), 0.0, 1e-6);
    }

    auto voxelmap_pcl = std::make_shared<IncrementalVoxelMap<FlatContainer<>>>(1.0);
    voxelmap_pcl->insert(*points_pcl);
    test_voxelmap(*points, *voxelmap_pcl);
  } else if (method == "GVOX") {
    auto voxelmap = std::make_shared<GaussianVoxelMap>(1.0);
    voxelmap->insert(*points);
    test_voxelmap(*points, *voxelmap);

    auto indices = traits::point_indices(*voxelmap);
    auto voxel_points = traits::voxel_points(*voxelmap);
    auto voxel_covs = traits::voxel_covs(*voxelmap);

    EXPECT_EQ(indices.size(), voxel_points.size());
    EXPECT_EQ(indices.size(), voxel_covs.size());

    for (size_t i = 0; i < indices.size(); i++) {
      EXPECT_NEAR((voxel_points[i] - traits::point(*voxelmap, indices[i])).squaredNorm(), 0.0, 1e-6);
      EXPECT_NEAR((voxel_covs[i] - traits::cov(*voxelmap, indices[i])).squaredNorm(), 0.0, 1e-6);
    }
  } else {
    throw std::runtime_error("Invalid method: " + method);
  }
}
