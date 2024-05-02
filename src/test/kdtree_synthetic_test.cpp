#include <random>

#include <gtest/gtest.h>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/ann/kdtree_tbb.hpp>

using namespace small_gicp;

struct Problem {
  using Ptr = std::shared_ptr<Problem>;
  using ConstPtr = std::shared_ptr<const Problem>;

  std::string target_name;
  std::string query_name;
  PointCloud::Ptr target;
  PointCloud::Ptr query;

  std::vector<std::vector<size_t>> indices;
  std::vector<std::vector<double>> dists;
};

class KdTreeSyntheticTest : public testing::Test, public testing::WithParamInterface<std::string> {
public:
  void SetUp() override {
    std::mt19937 mt;

    std::vector<Eigen::Vector4d> points(256);

    // Geneate point sets with different distributions
    // Uniform real [-1.0, 1.0]
    auto udist = std::uniform_real_distribution<>(-1.0, 1.0);
    std::generate(points.begin(), points.end(), [&]() { return Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0); });
    names.emplace_back("r[-1.0, 1.0]");
    targets.emplace_back(std::make_shared<PointCloud>(points));

    // Uniform real with a wide band [-1e6, 1e6]
    udist = std::uniform_real_distribution<>(-1e6, 1e6);
    std::generate(points.begin(), points.end(), [&]() { return Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 1.0); });
    names.emplace_back("r[-1e6, 1e6]");
    targets.emplace_back(std::make_shared<PointCloud>(points));

    // Two separate uniform real distributions [-1.0, -0.5] + [0.5, 1.0]
    auto udist_l = std::uniform_real_distribution<>(-1.0, -0.5);
    auto udist_r = std::uniform_real_distribution<>(0.5, 1.0);
    std::generate(points.begin(), points.begin() + points.size() / 2, [&]() { return Eigen::Vector4d(udist_l(mt), udist_l(mt), udist_l(mt), 1.0); });
    std::generate(points.begin() + points.size() / 2, points.end(), [&]() { return Eigen::Vector4d(udist_r(mt), udist_r(mt), udist_r(mt), 1.0); });
    names.emplace_back("r[-1.0, -0.5]+R[0.5, 1.0]");
    targets.emplace_back(std::make_shared<PointCloud>(points));

    // Uniform integer [-3, 3]
    auto idist = std::uniform_int_distribution<>(-3, 3);
    std::generate(points.begin(), points.end(), [&]() { return Eigen::Vector4d(idist(mt), idist(mt), idist(mt), 1.0); });
    names.emplace_back("i[-3, 3]");
    targets.emplace_back(std::make_shared<PointCloud>(points));

    // Normal distribution (mean=0.0, std=1.0)
    auto ndist = std::normal_distribution<>(0.0, 1.0);
    std::generate(points.begin(), points.end(), [&]() { return Eigen::Vector4d(idist(mt), idist(mt), idist(mt), 1.0); });
    names.emplace_back("n(0.0,1.0)");
    targets.emplace_back(std::make_shared<PointCloud>(points));

    const int N = targets.size();
    for (int i = 0; i < N; i++) {
      // Create point sets with fewer points
      auto points = std::make_shared<PointCloud>(targets[i]->points);
      points->resize(10);
      names.emplace_back(names[i] + "(10pts)");
      targets.emplace_back(points);

      points = std::make_shared<PointCloud>(targets[i]->points);
      points->resize(5);
      names.emplace_back(names[i] + "(5pts)");
      targets.emplace_back(points);
    }

    // Generate problems and groundtruth
    for (int i = 0; i < targets.size(); i++) {
      for (int j = 0; j < targets.size(); j++) {
        auto problem = std::make_shared<Problem>();
        problem->target_name = names[i];
        problem->query_name = names[j];
        problem->target = targets[i];
        problem->query = targets[j];

        auto [indices, dists] = bruteforce_knn(*problem->target, *problem->query);
        problem->indices = std::move(indices);
        problem->dists = std::move(dists);

        problems.emplace_back(problem);
      }
    }
  }

  // Brute-force k-nearest neighbor search
  std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<double>>> bruteforce_knn(const PointCloud& target, const PointCloud& queries) {
    const int k = 20;
    std::vector<std::vector<size_t>> k_indices(queries.size());
    std::vector<std::vector<double>> k_sq_dists(queries.size());

    struct IndexDist {
      bool operator<(const IndexDist& rhs) const { return dist < rhs.dist; }
      int index;
      double dist;
    };

    for (int i = 0; i < queries.size(); i++) {
      const auto& query = queries.point(i);

      std::priority_queue<IndexDist> queue;
      for (int j = 0; j < target.size(); j++) {
        const double sq_dist = (target.point(j) - query).squaredNorm();
        if (queue.size() < k) {
          queue.push(IndexDist{j, sq_dist});
        } else if (sq_dist < queue.top().dist) {
          queue.pop();
          queue.push(IndexDist{j, sq_dist});
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

    return {k_indices, k_sq_dists};
  }

  template <typename KdTree>
  std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<double>>> nearest_neighbor_search(const KdTree& kdtree, const PointCloud& target, const PointCloud& queries) {
    std::vector<std::vector<size_t>> k_indices(queries.size());
    std::vector<std::vector<double>> k_sq_dists(queries.size());

    for (int i = 0; i < queries.size(); i++) {
      std::vector<size_t> indices(1, 0);
      std::vector<double> sq_dists(1, 0.0);

      const size_t num_results = traits::nearest_neighbor_search(kdtree, queries.point(i), indices.data(), sq_dists.data());
      indices.resize(num_results);
      sq_dists.resize(num_results);

      k_indices[i] = std::move(indices);
      k_sq_dists[i] = std::move(sq_dists);
    }

    return {k_indices, k_sq_dists};
  }

  template <typename KdTree>
  std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<double>>> knn_search(const KdTree& kdtree, const PointCloud& target, const PointCloud& queries, int k) {
    std::vector<std::vector<size_t>> k_indices(queries.size());
    std::vector<std::vector<double>> k_sq_dists(queries.size());

    for (int i = 0; i < queries.size(); i++) {
      std::vector<size_t> indices(k, 0);
      std::vector<double> sq_dists(k, 0.0);

      const size_t num_results = traits::knn_search(kdtree, queries.point(i), k, indices.data(), sq_dists.data());
      indices.resize(num_results);
      sq_dists.resize(num_results);

      k_indices[i] = std::move(indices);
      k_sq_dists[i] = std::move(sq_dists);
    }

    return {k_indices, k_sq_dists};
  }

  void
  validate(const Problem::Ptr& prob, const std::vector<std::vector<size_t>>& k_indices, const std::vector<std::vector<double>>& k_sq_dists, int k, const std::string& test_name) {
    EXPECT_EQ(k_indices.size(), prob->query->size()) << test_name;
    EXPECT_EQ(k_sq_dists.size(), prob->query->size()) << test_name;

    for (int i = 0; i < prob->query->size(); i++) {
      const int expected_n = std::min<int>(k, prob->target->size());
      EXPECT_EQ(k_indices[i].size(), expected_n) << test_name;
      EXPECT_EQ(k_sq_dists[i].size(), expected_n) << test_name;

      for (int j = 0; j < expected_n; j++) {
        const double sq_dist = (prob->target->point(k_indices[i][j]) - prob->query->point(i)).squaredNorm();
        EXPECT_NEAR(k_sq_dists[i][j], sq_dist, 1e-3) << test_name;
        EXPECT_NEAR(k_sq_dists[i][j], prob->dists[i][j], 1e-3) << test_name;
      }
    }
  }

protected:
  std::vector<std::string> names;
  std::vector<PointCloud::Ptr> targets;

  std::vector<std::shared_ptr<Problem>> problems;
};

// Check if the data is synthesized correctly
TEST_F(KdTreeSyntheticTest, LoadCheck) {
  EXPECT_EQ(names.size(), targets.size());
  EXPECT_NE(problems.size(), 0);

  for (const auto& prob : problems) {
    EXPECT_EQ(prob->indices.size(), prob->query->size());
    EXPECT_EQ(prob->dists.size(), prob->query->size());

    for (int i = 0; i < prob->query->size(); i++) {
      EXPECT_EQ(prob->indices[i].size(), prob->dists[i].size());
      EXPECT_EQ(prob->indices[i].size(), std::min<int>(20, prob->target->size()));
      EXPECT_TRUE(std::is_sorted(prob->dists[i].begin(), prob->dists[i].end()));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(KdTreeSyntheticTest, KdTreeSyntheticTest, testing::Values("SMALL", "TBB", "OMP"), [](const auto& info) { return info.param; });

// Check if kdtree works correctly
TEST_P(KdTreeSyntheticTest, KnnTest) {
  const auto method = GetParam();

  for (const auto& prob : problems) {
    KdTree<PointCloud>::Ptr kdtree;
    KdTree<PointCloud, NormalProjection>::Ptr kdtree_normal;

    if (method == "SMALL") {
      kdtree = std::make_shared<KdTree<PointCloud>>(prob->target);
      kdtree_normal = std::make_shared<KdTree<PointCloud, NormalProjection>>(prob->target);
    } else if (method == "TBB") {
      kdtree = std::make_shared<KdTree<PointCloud>>(prob->target, KdTreeBuilderTBB());
      kdtree_normal = std::make_shared<KdTree<PointCloud, NormalProjection>>(prob->target, KdTreeBuilderTBB());
    } else if (method == "OMP") {
      kdtree = std::make_shared<KdTree<PointCloud>>(prob->target, KdTreeBuilderOMP());
      kdtree_normal = std::make_shared<KdTree<PointCloud, NormalProjection>>(prob->target, KdTreeBuilderOMP());
    }

    std::vector<int> ks = {1, 2, 3, 5, 10, 20};
    for (auto k : ks) {
      std::stringstream test_name;
      test_name << prob->target_name << "(" << prob->target_name << "," << prob->query_name << "," << k << ")";

      const auto [k_indices, k_sq_dists] = knn_search(*kdtree, *prob->target, *prob->query, k);
      validate(prob, k_indices, k_sq_dists, k, test_name.str());

      test_name << "_normal";
      const auto [k_indices2, k_sq_dists2] = knn_search(*kdtree_normal, *prob->target, *prob->query, k);
      validate(prob, k_indices2, k_sq_dists2, k, test_name.str());
    }

    std::stringstream test_name;
    test_name << prob->target_name << "(" << prob->target_name << "," << prob->query_name << ",nn)";
    const auto [k_indices, k_sq_dists] = nearest_neighbor_search(*kdtree, *prob->target, *prob->query);
    validate(prob, k_indices, k_sq_dists, 1, test_name.str());

    test_name << "_normal";
    const auto [k_indices2, k_sq_dists2] = nearest_neighbor_search(*kdtree_normal, *prob->target, *prob->query);
    validate(prob, k_indices2, k_sq_dists2, 1, test_name.str());
  }
}
