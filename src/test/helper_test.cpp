#include <gtest/gtest.h>
#include <small_gicp/registration/registration_helper.hpp>

#include <small_gicp/benchmark/read_points.hpp>
#include <small_gicp/registration/registration_helper.hpp>

using namespace small_gicp;

class HelperTest : public testing::Test, public testing::WithParamInterface<const char*> {
public:
  void SetUp() override {
    // Load points
    target_raw = std::make_shared<PointCloud>(read_ply("data/target.ply"));
    source_raw = std::make_shared<PointCloud>(read_ply("data/source.ply"));

    std::ifstream ifs("data/T_target_source.txt");
    if (!ifs) {
      std::cerr << "error: failed to open T_target_source.txt" << std::endl;
    }
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        ifs >> T_target_source(i, j);
      }
    }
  }

  bool compare_transformation(const Eigen::Isometry3d& T1, const Eigen::Isometry3d& T2) {
    const Eigen::Isometry3d e = T1.inverse() * T2;
    const double error_rot = Eigen::AngleAxisd(e.linear()).angle();
    const double error_trans = e.translation().norm();

    const double rot_tol = 2.5 * M_PI / 180.0;
    const double trans_tol = 0.2;

    EXPECT_NEAR(error_rot, 0.0, rot_tol);
    EXPECT_NEAR(error_trans, 0.0, trans_tol);

    return error_rot < rot_tol && error_trans < trans_tol;
  }

protected:
  PointCloud::Ptr target_raw;         ///< Target points
  PointCloud::Ptr source_raw;         ///< Source points
  Eigen::Isometry3d T_target_source;  ///< Ground truth transformation
};

// Load check
TEST_F(HelperTest, LoadCheck) {
  EXPECT_FALSE(target_raw->empty());
  EXPECT_FALSE(source_raw->empty());
}

TEST_F(HelperTest, EmptyPreprocess) {
  auto empty_points = std::make_shared<PointCloud>();
  auto [target, target_tree] = preprocess_points(*empty_points, 0.1, 10, 1);
  EXPECT_TRUE(target);
  EXPECT_TRUE(target_tree);
  EXPECT_EQ(target->size(), 0);
}

TEST_F(HelperTest, Preprocess) {
  const std::vector<int> num_threads = {1, 2};

  for (int N : num_threads) {
    auto [target, target_tree] = preprocess_points(*target_raw, 0.1, 10, N);
    EXPECT_TRUE(target) << "N=" << N;
    EXPECT_TRUE(target_tree) << "N=" << N;
    EXPECT_GT(target->size(), 1000) << "N=" << N;
    EXPECT_LT(target->size(), target_raw->size()) << "N=" << N;

    auto [target2, target_tree2] = preprocess_points(target_raw->points, 0.1, 10, N);
    EXPECT_TRUE(target2) << "N=" << N;
    EXPECT_TRUE(target_tree2) << "N=" << N;
    EXPECT_NEAR(target->size(), target2->size(), 100) << "N=" << N;
  }
}

TEST_F(HelperTest, EmptyGaussianVoxelMap) {
  auto empty_points = std::make_shared<PointCloud>();
  auto voxelmap = create_gaussian_voxelmap(*empty_points, 0.1);
  EXPECT_TRUE(voxelmap);
  EXPECT_EQ(voxelmap->size(), 0);
}

TEST_F(HelperTest, GaussianVoxelMap) {
  auto voxelmap = create_gaussian_voxelmap(*target_raw, 0.1);
  EXPECT_TRUE(voxelmap);
  EXPECT_GT(voxelmap->size(), 1000);
  EXPECT_LT(voxelmap->size(), target_raw->size());
}

INSTANTIATE_TEST_SUITE_P(HelperTest, HelperTest, testing::Values("ICP", "PLANE_ICP", "GICP", "VGICP"), [](const auto& info) { return info.param; });

TEST_P(HelperTest, AlignEigen) {
  const std::string method = GetParam();

  RegistrationSetting setting;
  if (method == "ICP") {
    setting.type = RegistrationSetting::ICP;
  } else if (method == "PLANE_ICP") {
    setting.type = RegistrationSetting::PLANE_ICP;
  } else if (method == "GICP") {
    setting.type = RegistrationSetting::GICP;
  } else if (method == "VGICP") {
    setting.type = RegistrationSetting::VGICP;
  } else {
    std::cerr << "error: unknown method " << method << std::endl;
  }

  // Forward test
  auto result = align(target_raw->points, source_raw->points, Eigen::Isometry3d::Identity(), setting);
  EXPECT_TRUE(compare_transformation(T_target_source, result.T_target_source));

  // Backward test
  auto result2 = align(source_raw->points, target_raw->points, Eigen::Isometry3d::Identity(), setting);
  EXPECT_TRUE(compare_transformation(T_target_source, result2.T_target_source.inverse()));
}

TEST_P(HelperTest, AlignSmall) {
  const std::string method = GetParam();

  RegistrationSetting setting;
  if (method == "ICP") {
    setting.type = RegistrationSetting::ICP;
  } else if (method == "PLANE_ICP") {
    setting.type = RegistrationSetting::PLANE_ICP;
  } else if (method == "GICP") {
    setting.type = RegistrationSetting::GICP;
  } else if (method == "VGICP") {
    setting.type = RegistrationSetting::VGICP;
  } else {
    std::cerr << "error: unknown method " << method << std::endl;
  }

  RegistrationResult result;

  auto [target, target_tree] = preprocess_points(*target_raw, 0.1, 10, 1);
  auto [source, source_tree] = preprocess_points(*source_raw, 0.1, 10, 1);

  // Forward test
  if (method != "VGICP") {
    result = align(*target, *source, *target_tree, Eigen::Isometry3d::Identity(), setting);
  } else {
    auto target_voxelmap = create_gaussian_voxelmap(*target, 1.0);
    result = align(*target_voxelmap, *source, Eigen::Isometry3d::Identity(), setting);
  }
  EXPECT_TRUE(compare_transformation(T_target_source, result.T_target_source));

  // Backward test
  if (method != "VGICP") {
    result = align(*source, *target, *source_tree, Eigen::Isometry3d::Identity(), setting);
  } else {
    auto source_voxelmap = create_gaussian_voxelmap(*source, 1.0);
    result = align(*source_voxelmap, *target, Eigen::Isometry3d::Identity(), setting);
  }
  EXPECT_TRUE(compare_transformation(T_target_source, result.T_target_source.inverse()));
}
