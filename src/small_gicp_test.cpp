#include <iostream>
#include <filesystem>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <small_gicp/util/read_points.hpp>

#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/points/gaussian_voxelmap.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/cached_kdtree.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_tbb.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/util/normal_estimation_tbb.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/reduction_tbb.hpp>
#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/factors/plane_icp_factor.hpp>

#include <glk/io/ply_io.hpp>
#include <glk/normal_distributions.hpp>
#include <guik/viewer/light_viewer.hpp>

#include <easy_profiler.hpp>

int main(int argc, char** argv) {
  using namespace small_gicp;

  const int num_threads = 32;
  tbb::task_scheduler_init init(num_threads);

  const std::string dataset_path = "/home/koide/datasets/kitti/velodyne_filtered";
  std::vector<std::string> filenames;
  for (const auto& path : std::filesystem::directory_iterator(dataset_path)) {
    if (path.path().extension() != ".bin") {
      continue;
    }

    filenames.emplace_back(path.path().string());
  }
  std::ranges::sort(filenames);

  auto viewer = guik::viewer();
  viewer->disable_vsync();

  GaussianVoxelMap::Ptr voxelmap;

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  for (const auto& filename : filenames) {
    glim::EasyProfiler prof("prof");
    prof.push("read_points");
    const auto raw_points = read_points(filename);

    prof.push("copy");
    auto points = std::make_shared<PointCloud>(raw_points);
    prof.push("downsample");
    points = voxelgrid_sampling_tbb(*points, 0.1);
    prof.push("estimate covs");
    estimate_covariances_tbb(*points, 10);
    std::cout << raw_points.size() << " => " << points->size() << std::endl;

    if (voxelmap == nullptr) {
      voxelmap = std::make_shared<GaussianVoxelMap>(1.0);
      voxelmap->insert(*points, T);
      continue;
    }

    //
    prof.push("create_tbb");
    Registration<GaussianVoxelMap, PointCloud, GaussianVoxelMap, GICPFactor, NullRejector, ParallelReductionTBB, LevenbergMarquardtOptimizer> registration_tbb;
    registration_tbb.optimizer.verbose = true;
    prof.push("registration_tbb");
    auto result = registration_tbb.align(*voxelmap, *points, *voxelmap, T);

    prof.push("update");
    T = result.T_target_source;
    voxelmap->insert(*points, T);

    prof.push("show");
    // viewer->update_points("current", raw_points[0].data(), sizeof(float) * 4, raw_points.size(), guik::FlatOrange(T).set_point_scale(2.0f));

    // std::vector<Eigen::Vector4d> means;
    // std::vector<Eigen::Matrix4d> covs;
    // for (const auto& voxel : voxelmap->flat_voxels) {
    //   means.emplace_back(voxel.mean);
    //   covs.emplace_back(voxel.cov);
    // }
    // viewer->update_normal_dists("target", means, covs, 0.5, guik::Rainbow());

    std::cout << "--- T ---" << std::endl << T.matrix() << std::endl;

    if (!viewer->spin_once()) {
      break;
    }
  }

  return 0;
}