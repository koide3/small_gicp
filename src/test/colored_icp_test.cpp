// SPDX-FileCopyrightText: Copyright 2026 Kenji Koide
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/points/traits.hpp>

#include <small_gicp/factors/colored_icp_factor.hpp>
#include <small_gicp/factors/general_factor.hpp>
#include <small_gicp/factors/robust_kernel.hpp>

#include <small_gicp/registration/reduction.hpp>
#ifdef BUILD_WITH_TBB
#include <small_gicp/registration/reduction_tbb.hpp>
#endif
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/registration_helper.hpp>

#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/lie.hpp>
#include <small_gicp/util/normal_estimation.hpp>
#include <small_gicp/util/color_gradient.hpp>
#include <small_gicp/util/color_gradient_omp.hpp>
#ifdef BUILD_WITH_TBB
#include <small_gicp/util/color_gradient_tbb.hpp>
#endif

#include <small_gicp/benchmark/read_points.hpp>

using namespace small_gicp;

// ---------------------------------------------------------------------------
// ColorGradientTest : color gradient estimation on synthetic linear fields
// ---------------------------------------------------------------------------
class ColorGradientTest : public testing::Test {
public:
  void SetUp() override {
    // 30x30 grid (spacing 0.1) on a tilted plane
    const Eigen::Vector3d plane_normal = Eigen::Vector3d(0.2, 0.1, 1.0).normalized();
    Eigen::Vector3d tangent1 = Eigen::Vector3d::UnitX() - plane_normal * plane_normal.x();
    tangent1.normalize();
    const Eigen::Vector3d tangent2 = plane_normal.cross(tangent1);

    const int num_grids = 30;
    const double spacing = 0.1;
    cloud->resize(num_grids * num_grids);
    for (int i = 0; i < num_grids; i++) {
      for (int j = 0; j < num_grids; j++) {
        const size_t index = i * num_grids + j;
        cloud->point(index) << (i - 14.5) * spacing * tangent1 + (j - 14.5) * spacing * tangent2, 1.0;
        // Linear intensity field I(p) = a . p + b
        const double intensity = field_gradient.dot(cloud->point(index).head<3>()) + field_bias;
        cloud->color(index) << intensity, intensity, intensity, 0.0;
      }
    }

    tree = std::make_shared<KdTree<PointCloud>>(cloud);
    estimate_normals(*cloud, *tree, 20);
  }

protected:
  const Eigen::Vector3d field_gradient{0.7, -0.3, 0.5};
  const double field_bias = 0.2;

  PointCloud::Ptr cloud = std::make_shared<PointCloud>();
  KdTree<PointCloud>::Ptr tree;
};

// Least squares over a planar neighborhood of a linear field recovers the tangential component a - (a . n) n exactly
TEST_F(ColorGradientTest, gradient_linear_field) {
  estimate_color_gradients(*cloud, *tree, 30, 0.5);
  EXPECT_TRUE(traits::has_color_grads(*cloud));

  size_t num_strict = 0;
  for (size_t i = 0; i < cloud->size(); i++) {
    const Eigen::Vector3d normal = cloud->normal(i).head<3>();
    const Eigen::Vector3d expected = field_gradient - field_gradient.dot(normal) * normal;
    const Eigen::Vector3d grad = traits::color_grad(*cloud, i).head<3>();

    for (int k = 0; k < 3; k++) {
      EXPECT_NEAR(grad[k], expected[k], 1e-5) << "point=" << i;
    }
    EXPECT_NEAR(cloud->color_grad(i).w(), 0.0, 1e-9) << "point=" << i;

    if ((grad - expected).cwiseAbs().maxCoeff() < 1e-6) {
      num_strict++;
    }
  }

  EXPECT_GE(num_strict, cloud->size() * 9 / 10);
}

// An isolated point with too few neighbors within the radius must keep an exactly zero gradient
TEST_F(ColorGradientTest, gradient_sparse_invalid) {
  auto sparse_cloud = std::make_shared<PointCloud>();
  const int num_grids = 5;
  const double spacing = 0.1;
  const Eigen::Vector3d grad{0.7, -0.3, 0.5};

  sparse_cloud->resize(num_grids * num_grids + 1);
  const size_t isolated_index = num_grids * num_grids;
  for (int i = 0; i < num_grids; i++) {
    for (int j = 0; j < num_grids; j++) {
      const size_t index = i * num_grids + j;
      sparse_cloud->point(index) << i * spacing, j * spacing, 0.0, 1.0;
      const double intensity = grad.dot(sparse_cloud->point(index).head<3>()) + 0.2;
      sparse_cloud->color(index) << intensity, intensity, intensity, 0.0;
    }
  }
  sparse_cloud->point(isolated_index) << 100.0, 100.0, 100.0, 1.0;
  sparse_cloud->color(isolated_index) << 1.0, 1.0, 1.0, 0.0;

  auto sparse_tree = std::make_shared<KdTree<PointCloud>>(sparse_cloud);
  estimate_normals(*sparse_cloud, *sparse_tree, 20);
  estimate_color_gradients(*sparse_cloud, *sparse_tree, 30, 1.0);

  EXPECT_TRUE(sparse_cloud->color_grad(isolated_index).isZero(0.0));

  // Sanity: the dense cluster itself is estimated (exact tangential gradient (0.7, -0.3, 0) on the z=0 plane)
  EXPECT_NEAR(sparse_cloud->color_grad(12).x(), 0.7, 1e-6);
  EXPECT_NEAR(sparse_cloud->color_grad(12).y(), -0.3, 1e-6);
  EXPECT_NEAR(sparse_cloud->color_grad(12).z(), 0.0, 1e-6);
}

// Serial and OpenMP estimation share the per-point accumulation order and must be bitwise identical
TEST_F(ColorGradientTest, gradient_matches_omp) {
  auto serial = std::make_shared<PointCloud>(*cloud);
  auto parallel = std::make_shared<PointCloud>(*cloud);

  estimate_color_gradients(*serial, *tree, 30, 0.5);
  estimate_color_gradients_omp(*parallel, *tree, 30, 2, 0.5);

  EXPECT_EQ(serial->color_grads.size(), parallel->color_grads.size());
  const size_t n = std::min(serial->color_grads.size(), parallel->color_grads.size());
  for (size_t i = 0; i < n; i++) {
    EXPECT_TRUE((serial->color_grad(i) - parallel->color_grad(i)).isZero(0.0)) << "point=" << i;
  }
}

// ---------------------------------------------------------------------------
// DownsamplingColorTest : color handling in voxelgrid_sampling
// ---------------------------------------------------------------------------
TEST(DownsamplingColorTest, voxelgrid_averages_colors) {
  auto cloud = std::make_shared<PointCloud>();
  cloud->resize(4);
  cloud->point(0) << 0.0, 0.0, 0.0, 1.0;
  cloud->point(1) << 0.5, 0.0, 0.0, 1.0;
  cloud->point(2) << 0.0, 0.5, 0.0, 1.0;
  cloud->point(3) << 0.5, 0.5, 0.5, 1.0;
  cloud->color(0) << 0.25, 0.5, 0.75, 0.0;
  cloud->color(1) << 0.5, 0.25, 0.5, 0.0;
  cloud->color(2) << 0.75, 0.75, 0.25, 0.0;
  cloud->color(3) << 0.125, 0.375, 0.625, 0.0;

  auto down = voxelgrid_sampling(*cloud, 1.0);
  ASSERT_EQ(down->size(), 1);

  const Eigen::Vector4d expected_point(0.25, 0.25, 0.125, 1.0);
  const Eigen::Vector4d expected_color(0.40625, 0.46875, 0.53125, 0.0);
  for (int k = 0; k < 4; k++) {
    EXPECT_NEAR(down->point(0)[k], expected_point[k], 1e-12);
    EXPECT_NEAR(down->color(0)[k], expected_color[k], 1e-12);
  }

  // Cloud without colors must keep working.
  // Note that, as with normals and covariances, color values of a cloud that never had colors set are unspecified.
  auto plain = std::make_shared<PointCloud>(std::vector<Eigen::Vector4d>{
    Eigen::Vector4d(0.0, 0.0, 0.0, 1.0),
    Eigen::Vector4d(0.5, 0.0, 0.0, 1.0),
    Eigen::Vector4d(0.0, 0.5, 0.0, 1.0),
    Eigen::Vector4d(0.5, 0.5, 0.5, 1.0),
  });
  auto plain_down = voxelgrid_sampling(*plain, 1.0);
  EXPECT_EQ(plain_down->size(), 1);
  EXPECT_NEAR(plain_down->point(0)[0], expected_point[0], 1e-12);
}

TEST(DownsamplingColorTest, voxelgrid_colors_multi_voxel) {
  auto cloud = std::make_shared<PointCloud>();
  cloud->resize(4);
  cloud->point(0) << 0.0, 0.0, 0.0, 1.0;
  cloud->point(1) << 0.5, 0.5, 0.5, 1.0;
  cloud->point(2) << 1.0, 0.0, 0.0, 1.0;
  cloud->point(3) << 1.5, 0.5, 0.5, 1.0;
  cloud->color(0) << 1.0, 0.0, 0.0, 0.0;
  cloud->color(1) << 0.0, 1.0, 0.0, 0.0;
  cloud->color(2) << 0.0, 0.0, 1.0, 0.0;
  cloud->color(3) << 1.0, 1.0, 1.0, 0.0;

  auto down = voxelgrid_sampling(*cloud, 1.0);
  EXPECT_EQ(down->size(), 2);

  const std::array<std::pair<Eigen::Vector4d, Eigen::Vector4d>, 2> expected{
    std::make_pair(Eigen::Vector4d(0.25, 0.25, 0.25, 1.0), Eigen::Vector4d(0.5, 0.5, 0.0, 0.0)),
    std::make_pair(Eigen::Vector4d(1.25, 0.25, 0.25, 1.0), Eigen::Vector4d(0.5, 0.5, 1.0, 0.0)),
  };

  for (const auto& [expected_point, expected_color] : expected) {
    size_t nearest = 0;
    double nearest_dist = std::numeric_limits<double>::max();
    for (size_t j = 0; j < down->size(); j++) {
      const double dist = (down->point(j) - expected_point).norm();
      if (dist < nearest_dist) {
        nearest_dist = dist;
        nearest = j;
      }
    }
    EXPECT_LT(nearest_dist, 1e-6);
    for (int k = 0; k < 4; k++) {
      EXPECT_NEAR(down->point(nearest)[k], expected_point[k], 1e-12);
      EXPECT_NEAR(down->color(nearest)[k], expected_color[k], 1e-12);
    }
  }
}

// ---------------------------------------------------------------------------
// ColoredICPFactorTest : hand-derived linearization of a single correspondence
//
// Right perturbation T <- T * exp(delta) (consistent with ICPFactor), T = identity:
//   r1 = sqrt(lambda) * n^T (p_t - v_s)
//   J1 = sqrt(lambda)     * [n^T skew(p_s), -n^T]
//   r2 = sqrt(1-lambda) * (I_t + a^T (v_s - p_t) - I_s)
//   J2 = sqrt(1-lambda) * [-a^T skew(p_s), a^T]
// (the source point enters r1 negatively and r2 positively, hence the flipped J signs)
// H = J^T J and e = 0.5 * r^T r are invariant to residual row sign choices.
// ---------------------------------------------------------------------------
struct ColoredFactorScene {
  static constexpr double lambda = 0.968;

  const Eigen::Vector3d target_normal{0.0, 0.0, 1.0};
  const Eigen::Vector3d target_color_grad{0.1, 0.2, 0.0};  // tangent to the plane (a . n = 0)
  const double target_intensity = 0.5;

  PointCloud::Ptr target = std::make_shared<PointCloud>();
  PointCloud::Ptr source = std::make_shared<PointCloud>();
  KdTree<PointCloud>::Ptr target_tree;
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
};

static ColoredFactorScene make_colored_factor_scene() {
  ColoredFactorScene scene;

  scene.target->resize(1);
  scene.target->point(0) << 0.0, 0.0, 0.0, 1.0;
  scene.target->normal(0) << scene.target_normal, 0.0;
  scene.target->color(0) << scene.target_intensity, scene.target_intensity, scene.target_intensity, 0.0;
  scene.target->color_grad(0) << scene.target_color_grad, 0.0;

  scene.source->resize(1);
  scene.source->point(0) << 0.1, 0.0, 0.2, 1.0;
  const double source_intensity = scene.target_intensity;  // zeroes the photometric offset term
  traits::set_color(*scene.source, 0, Eigen::Vector4d(source_intensity, source_intensity, source_intensity, 0.0));

  scene.target_tree = std::make_shared<KdTree<PointCloud>>(scene.target);
  return scene;
}

static std::tuple<Eigen::Matrix<double, 2, 6>, Eigen::Vector2d> colored_factor_reference(const ColoredFactorScene& scene, double lambda) {
  const Eigen::Matrix3d R = scene.T.linear();
  const Eigen::Vector3d pt = scene.target->point(0).head<3>();
  const Eigen::Vector3d ps = scene.source->point(0).head<3>();
  const Eigen::Vector3d vs = (scene.T * scene.source->point(0)).head<3>();
  const Eigen::Vector3d n = scene.target_normal;
  const Eigen::Vector3d a = scene.target_color_grad;
  const double It = scene.target_intensity;
  const double Is = scene.source->color(0).x();

  Eigen::Matrix<double, 2, 6> J = Eigen::Matrix<double, 2, 6>::Zero();
  J.row(0).head<3>() = std::sqrt(lambda) * (n.transpose() * skew(R * ps));
  J.row(0).tail<3>() = -std::sqrt(lambda) * (n.transpose() * R);
  J.row(1).head<3>() = -std::sqrt(1.0 - lambda) * (a.transpose() * skew(R * ps));
  J.row(1).tail<3>() = std::sqrt(1.0 - lambda) * (a.transpose() * R);

  Eigen::Vector2d r;
  r(0) = std::sqrt(lambda) * n.dot(pt - vs);
  r(1) = std::sqrt(1.0 - lambda) * (It + a.dot(vs - pt) - Is);
  return {J, r};
}

TEST(ColoredICPFactorTest, factor_matches_hand_computed) {
  auto scene = make_colored_factor_scene();

  const auto [J, r] = colored_factor_reference(scene, ColoredFactorScene::lambda);
  const Eigen::Matrix<double, 6, 6> H_expected = J.transpose() * J;
  const Eigen::Matrix<double, 6, 1> b_expected = J.transpose() * r;
  const double e_expected = 0.5 * r.squaredNorm();

  ColoredICPFactor factor;
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
  double e = 0.0;

  DistanceRejector rejector;
  rejector.max_dist_sq = 100.0;
  EXPECT_TRUE(factor.linearize(*scene.target, *scene.source, *scene.target_tree, scene.T, 0, rejector, &H, &b, &e));

  EXPECT_TRUE(factor.inlier());
  EXPECT_EQ(factor.target_index, 0);
  EXPECT_EQ(factor.source_index, 0);

  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      EXPECT_NEAR(H(i, j), H_expected(i, j), 1e-12) << "H(" << i << "," << j << ")";
    }
    EXPECT_NEAR(b[i], b_expected[i], 1e-12) << "b(" << i << ")";
  }
  EXPECT_NEAR(e, e_expected, 1e-12);
  EXPECT_NEAR(factor.error(*scene.target, *scene.source, scene.T), e, 1e-12);
}

TEST(ColoredICPFactorTest, factor_rejects_far_correspondence) {
  auto scene = make_colored_factor_scene();

  ColoredICPFactor factor;
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double e = 0.0;

  // |p_t - v_s|^2 = 0.05 exceeds max_dist_sq
  DistanceRejector rejector;
  rejector.max_dist_sq = 0.01;
  EXPECT_FALSE(factor.linearize(*scene.target, *scene.source, *scene.target_tree, scene.T, 0, rejector, &H, &b, &e));
  EXPECT_FALSE(factor.inlier());
  EXPECT_NEAR(factor.error(*scene.target, *scene.source, scene.T), 0.0, 1e-12);
}

TEST(ColoredICPFactorTest, factor_lambda_zero) {
  auto scene = make_colored_factor_scene();

  const auto linearize = [&scene](double lambda, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b, double* e) {
    ColoredICPFactor::Setting setting;
    setting.lambda_geometric = lambda;
    ColoredICPFactor factor(setting);
    DistanceRejector rejector;
    rejector.max_dist_sq = 100.0;
    return factor.linearize(*scene.target, *scene.source, *scene.target_tree, scene.T, 0, rejector, H, b, e);
  };

  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double e = 0.0;

  // lambda = 0.0 -> photometric row only
  const auto [J0, r0] = colored_factor_reference(scene, 0.0);
  EXPECT_TRUE(linearize(0.0, &H, &b, &e));
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      EXPECT_NEAR(H(i, j), (J0.transpose() * J0)(i, j), 1e-12) << "lambda=0 H(" << i << "," << j << ")";
    }
    EXPECT_NEAR(b[i], (J0.transpose() * r0)[i], 1e-12) << "lambda=0 b(" << i << ")";
  }
  EXPECT_NEAR(e, 0.5 * r0.squaredNorm(), 1e-12);

  // lambda = 1.0 -> geometric row only
  const auto [J1, r1] = colored_factor_reference(scene, 1.0);
  EXPECT_TRUE(linearize(1.0, &H, &b, &e));
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      EXPECT_NEAR(H(i, j), (J1.transpose() * J1)(i, j), 1e-12) << "lambda=1 H(" << i << "," << j << ")";
    }
    EXPECT_NEAR(b[i], (J1.transpose() * r1)[i], 1e-12) << "lambda=1 b(" << i << ")";
  }
  EXPECT_NEAR(e, 0.5 * r1.squaredNorm(), 1e-12);
}

TEST(ColoredICPFactorTest, robust_factor_wraps) {
  auto scene = make_colored_factor_scene();

  DistanceRejector rejector;
  rejector.max_dist_sq = 100.0;

  Eigen::Matrix<double, 6, 6> H0 = Eigen::Matrix<double, 6, 6>::Zero(), Hr = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> b0 = Eigen::Matrix<double, 6, 1>::Zero(), br = Eigen::Matrix<double, 6, 1>::Zero();
  double e0 = 0.0, er = 0.0;

  ColoredICPFactor plain;
  EXPECT_TRUE(plain.linearize(*scene.target, *scene.source, *scene.target_tree, scene.T, 0, rejector, &H0, &b0, &e0));

  RobustFactor<Huber, ColoredICPFactor>::Setting setting;
  setting.robust_kernel.c = 0.01;  // < sqrt(e0) so that the Huber weight is strictly < 1
  RobustFactor<Huber, ColoredICPFactor> robust(setting);
  EXPECT_TRUE(robust.linearize(*scene.target, *scene.source, *scene.target_tree, scene.T, 0, rejector, &Hr, &br, &er));
  EXPECT_TRUE(robust.inlier());

  const double w = robust.robust_kernel.weight(std::sqrt(e0));
  EXPECT_LT(w, 1.0);
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      EXPECT_NEAR(Hr(i, j), w * H0(i, j), 1e-12) << "H(" << i << "," << j << ")";
    }
    EXPECT_NEAR(br[i], w * b0[i], 1e-12) << "b(" << i << ")";
  }
  EXPECT_NEAR(er, w * e0, 1e-12);
  EXPECT_NEAR(robust.error(*scene.target, *scene.source, scene.T), w * plain.error(*scene.target, *scene.source, scene.T), 1e-12);
}

// ---------------------------------------------------------------------------
// ColoredRegistrationTest : end-to-end Colored-ICP registration on data/*.ply
// ---------------------------------------------------------------------------
class ColoredRegistrationTest : public testing::Test {
public:
  void SetUp() override {
    std::ifstream ifs("data/T_target_source.txt");
    if (!ifs) {
      std::cerr << "error: failed to open T_target_source.txt" << std::endl;
    }
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        ifs >> T_target_source(i, j);
      }
    }

    // Attach a linear photometric field in the target frame to the RAW clouds before downsampling,
    // so voxel averaging preserves it (mean of a linear field = field of the mean).
    const auto attach_colors = [&](PointCloud& cloud, bool transform_to_target) {
      for (size_t i = 0; i < cloud.size(); i++) {
        const Eigen::Vector4d p = transform_to_target ? T_target_source * cloud.point(i) : cloud.point(i);
        cloud.color(i) << 0.5 + 0.3 * p.x(), 0.2 * p.y() + 0.1, 0.5 + 0.25 * p.z(), 0.0;
      }
    };

    auto target_raw = std::make_shared<PointCloud>(read_ply("data/target.ply"));
    attach_colors(*target_raw, false);
    target = voxelgrid_sampling(*target_raw, downsampling_resolution);
    estimate_normals_covariances(*target);

    auto source_raw = std::make_shared<PointCloud>(read_ply("data/source.ply"));
    attach_colors(*source_raw, true);
    source = voxelgrid_sampling(*source_raw, downsampling_resolution);
    estimate_normals_covariances(*source);

    target_tree = std::make_shared<KdTree<PointCloud>>(target);
    estimate_color_gradients(*target, *target_tree, 30, 2.0 * max_correspondence_distance);
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
  const double downsampling_resolution = 0.3;
  const double max_correspondence_distance = 1.0;

  PointCloud::Ptr target;
  PointCloud::Ptr source;
  KdTree<PointCloud>::Ptr target_tree;
  Eigen::Isometry3d T_target_source;
};

// Load check
TEST_F(ColoredRegistrationTest, LoadCheck) {
  EXPECT_FALSE(target->empty());
  EXPECT_FALSE(source->empty());
  EXPECT_TRUE(traits::has_colors(*target));
  EXPECT_TRUE(traits::has_colors(*source));
  EXPECT_TRUE(traits::has_color_grads(*target));
}

TEST_F(ColoredRegistrationTest, registration_serial) {
  RegistrationSetting setting;
  setting.type = RegistrationSetting::COLORED_ICP;
  setting.max_correspondence_distance = max_correspondence_distance;
  setting.max_iterations = 20;
  setting.num_threads = 1;

  auto result = align(*target, *source, *target_tree, Eigen::Isometry3d::Identity(), setting);
  EXPECT_TRUE(compare_transformation(T_target_source, result.T_target_source));
}

TEST_F(ColoredRegistrationTest, registration_omp) {
  RegistrationSetting setting;
  setting.type = RegistrationSetting::COLORED_ICP;
  setting.max_correspondence_distance = max_correspondence_distance;
  setting.max_iterations = 20;
  setting.num_threads = 4;

  auto result = align(*target, *source, *target_tree, Eigen::Isometry3d::Identity(), setting);
  EXPECT_TRUE(compare_transformation(T_target_source, result.T_target_source));
}

// Raw template path with the serial reduction and a Gauss-Newton optimizer
TEST_F(ColoredRegistrationTest, registration_template) {
  Registration<ColoredICPFactor, SerialReduction, NullFactor, DistanceRejector, GaussNewtonOptimizer> registration;
  registration.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;
  registration.optimizer.max_iterations = 20;

  auto result = registration.align(*target, *source, *target_tree, Eigen::Isometry3d::Identity());
  EXPECT_TRUE(compare_transformation(T_target_source, result.T_target_source));
}

#ifdef BUILD_WITH_TBB
TEST_F(ColoredRegistrationTest, registration_tbb) {
  Registration<ColoredICPFactor, ParallelReductionTBB, NullFactor, DistanceRejector, GaussNewtonOptimizer> registration;
  registration.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;
  registration.optimizer.max_iterations = 20;

  auto result = registration.align(*target, *source, *target_tree, Eigen::Isometry3d::Identity());
  EXPECT_TRUE(compare_transformation(T_target_source, result.T_target_source));
}
#endif

// Degraded inputs (missing color attributes) must return an identity result without crashing
TEST_F(ColoredRegistrationTest, degraded_inputs_error) {
  RegistrationSetting setting;
  setting.type = RegistrationSetting::COLORED_ICP;
  setting.max_correspondence_distance = max_correspondence_distance;
  setting.num_threads = 1;

  // Target without color gradients
  auto target_no_grads = std::make_shared<PointCloud>(*target);
  target_no_grads->color_grads.clear();
  auto result = align(*target_no_grads, *source, *target_tree, Eigen::Isometry3d::Identity(), setting);
  EXPECT_TRUE(result.T_target_source.matrix().isIdentity(1e-9));

  // Target without colors
  auto target_no_colors = std::make_shared<PointCloud>(*target);
  target_no_colors->colors.clear();
  result = align(*target_no_colors, *source, *target_tree, Eigen::Isometry3d::Identity(), setting);
  EXPECT_TRUE(result.T_target_source.matrix().isIdentity(1e-9));

  // Source without colors
  auto source_no_colors = std::make_shared<PointCloud>(*source);
  source_no_colors->colors.clear();
  result = align(*target, *source_no_colors, *target_tree, Eigen::Isometry3d::Identity(), setting);
  EXPECT_TRUE(result.T_target_source.matrix().isIdentity(1e-9));
}
