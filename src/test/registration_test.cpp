#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/pcl/pcl_registration_impl.hpp>
#include <small_gicp/points/point_cloud.hpp>

#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/factors/plane_icp_factor.hpp>
#include <small_gicp/factors/robust_kernel.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/reduction_tbb.hpp>
#include <small_gicp/registration/registration.hpp>

#include <small_gicp/benchmark/read_points.hpp>
#include <small_gicp/registration/registration_helper.hpp>

using namespace small_gicp;

class RegistrationTest : public testing::Test, public testing::WithParamInterface<std::tuple<const char*, const char*>> {
public:
  void SetUp() override {
    // Load points
    const double downsampling_resolution = 0.3;

    target = std::make_shared<PointCloud>(read_ply("data/target.ply"));
    target = voxelgrid_sampling(*target, downsampling_resolution);
    estimate_normals_covariances_omp(*target);

    target_pcl = pcl::make_shared<pcl::PointCloud<pcl::PointNormalCovariance>>();
    target_pcl->resize(target->size());
    for (size_t i = 0; i < target->size(); i++) {
      target_pcl->at(i).getVector4fMap() = target->point(i).cast<float>();
    }
    estimate_normals_covariances_omp(*target_pcl);

    source = std::make_shared<PointCloud>(read_ply("data/source.ply"));
    source = voxelgrid_sampling(*source, downsampling_resolution);
    estimate_normals_covariances_omp(*source);

    source_pcl = pcl::make_shared<pcl::PointCloud<pcl::PointNormalCovariance>>();
    source_pcl->resize(source->size());
    for (size_t i = 0; i < source->size(); i++) {
      source_pcl->at(i).getVector4fMap() = source->point(i).cast<float>();
    }
    estimate_normals_covariances_omp(*source_pcl);

    target_tree = std::make_shared<KdTree<PointCloud>>(target);
    source_tree = std::make_shared<KdTree<PointCloud>>(source);
    target_voxelmap = std::make_shared<GaussianVoxelMap>(1.0);
    target_voxelmap->insert(*target);
    source_voxelmap = std::make_shared<GaussianVoxelMap>(1.0);
    source_voxelmap->insert(*source);

    std::mt19937 mt;
    std::uniform_real_distribution<> udist(-1.0, 1.0);

    const int num_noise_poses = 4;
    T_noise.resize(num_noise_poses);
    for (int i = 0; i < num_noise_poses; i++) {
      T_noise[i] = Eigen::Isometry3d::Identity();
      if (i != 0) {
        T_noise[i].translation() = Eigen::Vector3d(udist(mt), udist(mt), udist(mt)) * 0.5;
        T_noise[i].linear() = Eigen::AngleAxisd(udist(mt) * 10.0 * M_PI / 180.0, Eigen::Vector3d(udist(mt), udist(mt), udist(mt)).normalized()).toRotationMatrix();
      }
    }

    const int num_shifted_poses = 2;
    T_source_shifted.resize(num_shifted_poses);
    shifted.resize(num_shifted_poses);
    for (int i = 0; i < num_shifted_poses; i++) {
      T_source_shifted[i] = Eigen::Isometry3d::Identity();
      T_source_shifted[i].translation() = Eigen::Vector3d(udist(mt), udist(mt), udist(mt)) * 2.0;
      T_source_shifted[i].linear() = Eigen::AngleAxisd(udist(mt) * M_PI_2 + M_PI_2, Eigen::Vector3d(udist(mt), udist(mt), udist(mt)).normalized()).toRotationMatrix();

      const Eigen::Isometry3d T_shifted_source = T_source_shifted[i].inverse();

      shifted[i] = std::make_shared<PointCloud>(source->points);
      std::transform(source->points.begin(), source->points.end(), shifted[i]->points.begin(), [&](const auto& p) -> Eigen::Vector4d { return T_shifted_source * p; });
      estimate_normals_covariances_omp(*shifted[i]);
    }

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

  template <typename Registration>
  void test_registration(Registration& registration) {
    for (int i = 0; i < T_noise.size(); i++) {
      auto result = registration.align(*target, *source, *target_tree, T_noise[i]);
      EXPECT_TRUE(compare_transformation(T_target_source, result.T_target_source)) << "Forward transformation T_noise=" << i;

      result = registration.align(*target_pcl, *source_pcl, *target_tree, T_noise[i]);
      EXPECT_TRUE(compare_transformation(T_target_source, result.T_target_source)) << "Forward transformation (pcl) T_noise=" << i;

      result = registration.align(*source, *target, *source_tree, T_noise[i]);
      EXPECT_TRUE(compare_transformation(T_target_source.inverse(), result.T_target_source)) << "Inverse transformation T_noise=" << i;

      result = registration.align(*source_pcl, *target_pcl, *source_tree, T_noise[i]);
      EXPECT_TRUE(compare_transformation(T_target_source.inverse(), result.T_target_source)) << "Inverse transformation (pcl) T_noise=" << i;

      for (int j = 0; j < T_source_shifted.size(); j++) {
        const Eigen::Isometry3d T_target_shifted = T_target_source * T_source_shifted[j];
        auto result = registration.align(*target, *shifted[j], *target_tree, T_source_shifted[j] * T_noise[i]);
        EXPECT_TRUE(compare_transformation(T_target_shifted, result.T_target_source)) << "Forward transformation T_source_shifted=" << j << " T_noise=" << i;
      }
    }
  }

  template <typename Registration>
  void test_registration_vgicp(Registration& registration) {
    for (int i = 0; i < T_noise.size(); i++) {
      auto result = registration.align(*target_voxelmap, *source, *target_voxelmap, T_noise[i]);
      EXPECT_TRUE(compare_transformation(T_target_source, result.T_target_source)) << "Forward transformation T_noise=" << i;

      result = registration.align(*source_voxelmap, *target, *source_voxelmap, T_noise[i]);
      EXPECT_TRUE(compare_transformation(T_target_source.inverse(), result.T_target_source)) << "Inverse transformation T_noise=" << i;

      for (int j = 0; j < T_source_shifted.size(); j++) {
        const Eigen::Isometry3d T_target_shifted = T_target_source * T_source_shifted[j];
        auto result = registration.align(*target_voxelmap, *shifted[j], *target_voxelmap, T_source_shifted[j] * T_noise[i]);
        EXPECT_TRUE(compare_transformation(T_target_shifted, result.T_target_source)) << "Forward transformation T_source_shifted=" << j << " T_noise=" << i;
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
  PointCloud::Ptr target;                                       ///< Target points
  PointCloud::Ptr source;                                       ///< Source points
  pcl::PointCloud<pcl::PointNormalCovariance>::Ptr target_pcl;  ///< Target points (pcl)
  pcl::PointCloud<pcl::PointNormalCovariance>::Ptr source_pcl;  ///< Source points (pcl)

  KdTree<PointCloud>::Ptr target_tree;    ///< Nearest neighbor search for the target points
  KdTree<PointCloud>::Ptr source_tree;    ///< Nearest neighbor search for the source points
  GaussianVoxelMap::Ptr target_voxelmap;  ///< Gaussian voxel map for the target points
  GaussianVoxelMap::Ptr source_voxelmap;  ///< Gaussian voxel map for the target points

  Eigen::Isometry3d T_target_source;  ///< Ground truth transformation

  std::vector<Eigen::Isometry3d> T_noise;
  std::vector<Eigen::Isometry3d> T_source_shifted;
  std::vector<PointCloud::Ptr> shifted;
};

// Load check
TEST_F(RegistrationTest, LoadCheck) {
  EXPECT_FALSE(target->empty());
  EXPECT_FALSE(source->empty());
  EXPECT_FALSE(target_pcl->empty());
  EXPECT_FALSE(source_pcl->empty());
}

// PCL interface test
TEST_F(RegistrationTest, PCLInterfaceTest) {
  RegistrationPCL<pcl::PointNormalCovariance, pcl::PointNormalCovariance> registration;
  registration.setNumThreads(2);
  registration.setCorrespondenceRandomness(20);
  registration.setRotationEpsilon(1e-4);
  registration.setTransformationEpsilon(1e-4);
  registration.setMaxCorrespondenceDistance(1.0);
  registration.setRegistrationType("GICP");

  // Forward align
  registration.setInputTarget(target_pcl);
  registration.setInputSource(source_pcl);

  pcl::PointCloud<pcl::PointNormalCovariance> aligned;
  registration.align(aligned);

  EXPECT_EQ(aligned.size(), source_pcl->size());
  EXPECT_TRUE(compare_transformation(T_target_source, Eigen::Isometry3d(registration.getFinalTransformation().cast<double>())));

  // Swap and backward align
  registration.swapSourceAndTarget();
  registration.align(aligned);

  EXPECT_EQ(aligned.size(), target_pcl->size());
  EXPECT_TRUE(compare_transformation(T_target_source.inverse(), Eigen::Isometry3d(registration.getFinalTransformation().cast<double>())));

  // Clear and forward align
  registration.clearTarget();
  registration.clearSource();

  registration.setInputTarget(target_pcl);
  registration.setInputSource(source_pcl);
  registration.align(aligned);

  EXPECT_EQ(aligned.size(), source_pcl->size());
  EXPECT_TRUE(compare_transformation(T_target_source, Eigen::Isometry3d(registration.getFinalTransformation().cast<double>())));

  Eigen::Matrix<double, 6, 6> H = registration.getFinalHessian();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig(H);
  EXPECT_GT(eig.eigenvalues()[0], 10.0);
  for (int i = 0; i < 6; i++) {
    for (int j = i + 1; j < 6; j++) {
      EXPECT_NEAR(H(i, j), H(j, i), 1e-3);
    }
  }

  registration.setRegistrationType("VGICP");
  registration.setVoxelResolution(1.0);

  // Forward align
  registration.setInputTarget(target_pcl);
  registration.setInputSource(source_pcl);

  registration.align(aligned);

  EXPECT_EQ(aligned.size(), source_pcl->size());
  EXPECT_TRUE(compare_transformation(T_target_source, Eigen::Isometry3d(registration.getFinalTransformation().cast<double>())));

  H = registration.getFinalHessian();
  eig.compute(H);
  EXPECT_GT(eig.eigenvalues()[0], 10.0);
  for (int i = 0; i < 6; i++) {
    for (int j = i + 1; j < 6; j++) {
      EXPECT_NEAR(H(i, j), H(j, i), 1e-3);
    }
  }

  // Re-use covariances
  std::vector<Eigen::Matrix4d> target_covs = registration.getTargetCovariances();
  std::vector<Eigen::Matrix4d> source_covs = registration.getSourceCovariances();
  EXPECT_EQ(target_covs.size(), target_pcl->size());
  EXPECT_EQ(source_covs.size(), source_pcl->size());

  registration.clearTarget();
  registration.clearSource();

  registration.setInputTarget(target_pcl);
  registration.setTargetCovariances(target_covs);
  registration.setInputSource(source_pcl);
  registration.setSourceCovariances(source_covs);

  registration.align(aligned);
  EXPECT_EQ(aligned.size(), source_pcl->size());
  EXPECT_TRUE(compare_transformation(T_target_source, Eigen::Isometry3d(registration.getFinalTransformation().cast<double>())));

  // Swap and backward align
  registration.swapSourceAndTarget();
  registration.align(aligned);

  EXPECT_EQ(aligned.size(), target_pcl->size());
  EXPECT_TRUE(compare_transformation(T_target_source.inverse(), Eigen::Isometry3d(registration.getFinalTransformation().cast<double>())));

  // Clear and forward align
  registration.clearTarget();
  registration.clearSource();

  registration.setInputTarget(target_pcl);
  registration.setInputSource(source_pcl);
  registration.align(aligned);

  EXPECT_EQ(aligned.size(), source_pcl->size());
  EXPECT_TRUE(compare_transformation(T_target_source, Eigen::Isometry3d(registration.getFinalTransformation().cast<double>())));
}

INSTANTIATE_TEST_SUITE_P(
  RegistrationTest,
  RegistrationTest,
  testing::Combine(testing::Values("ICP", "PLANE_ICP", "GICP", "VGICP", "HUBER_GICP", "CAUCHY_GICP"), testing::Values("SERIAL", "TBB", "OMP")),
  [](const auto& info) {
    std::stringstream sst;
    sst << std::get<0>(info.param) << "_" << std::get<1>(info.param);
    return sst.str();
  });

TEST_P(RegistrationTest, RegistrationTest) {
  const std::string factor = std::get<0>(GetParam());
  const std::string reduction = std::get<1>(GetParam());

  if (factor == "ICP") {
    if (reduction == "SERIAL") {
      Registration<ICPFactor, SerialReduction> reg;
      test_registration(reg);
    } else if (reduction == "TBB") {
      Registration<ICPFactor, ParallelReductionTBB> reg;
      test_registration(reg);
    } else if (reduction == "OMP") {
      Registration<ICPFactor, ParallelReductionOMP> reg;
      test_registration(reg);
    }
  } else if (factor == "PLANE_ICP") {
    if (reduction == "SERIAL") {
      Registration<PointToPlaneICPFactor, SerialReduction> reg;
      test_registration(reg);
    } else if (reduction == "TBB") {
      Registration<PointToPlaneICPFactor, ParallelReductionTBB> reg;
      test_registration(reg);
    } else if (reduction == "OMP") {
      Registration<PointToPlaneICPFactor, ParallelReductionOMP> reg;
      test_registration(reg);
    }
  } else if (factor == "GICP") {
    if (reduction == "SERIAL") {
      Registration<GICPFactor, SerialReduction> reg;
      test_registration(reg);
    } else if (reduction == "TBB") {
      Registration<GICPFactor, ParallelReductionTBB> reg;
      test_registration(reg);
    } else if (reduction == "OMP") {
      Registration<GICPFactor, ParallelReductionOMP> reg;
      test_registration(reg);
    }
  } else if (factor == "VGICP") {
    if (reduction == "SERIAL") {
      Registration<GICPFactor, SerialReduction> reg;
      test_registration_vgicp(reg);
    } else if (reduction == "TBB") {
      Registration<GICPFactor, ParallelReductionTBB> reg;
      test_registration_vgicp(reg);
    } else if (reduction == "OMP") {
      Registration<GICPFactor, ParallelReductionOMP> reg;
      test_registration_vgicp(reg);
    }
  } else if (factor == "HUBER_GICP") {
    if (reduction == "SERIAL") {
      Registration<RobustFactor<Huber, GICPFactor>, SerialReduction> reg;
      test_registration(reg);
    } else if (reduction == "TBB") {
      Registration<RobustFactor<Huber, GICPFactor>, ParallelReductionTBB> reg;
      test_registration(reg);
    } else if (reduction == "OMP") {
      Registration<RobustFactor<Huber, GICPFactor>, ParallelReductionOMP> reg;
      test_registration(reg);
    }
  } else if (factor == "CAUCHY_GICP") {
    if (reduction == "SERIAL") {
      Registration<RobustFactor<Cauchy, GICPFactor>, SerialReduction> reg;
      test_registration(reg);
    } else if (reduction == "TBB") {
      Registration<RobustFactor<Cauchy, GICPFactor>, ParallelReductionTBB> reg;
      test_registration(reg);
    } else if (reduction == "OMP") {
      Registration<RobustFactor<Cauchy, GICPFactor>, ParallelReductionOMP> reg;
      test_registration(reg);
    }
  } else {
    EXPECT_TRUE(false) << "error: unknown factor type " << factor;
  }
}