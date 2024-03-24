#include <iostream>
#include <filesystem>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <small_gicp/util/read_points.hpp>

#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_tbb.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/util/normal_estimation_tbb.hpp>
#include <small_gicp/ann/flat_voxelmap.hpp>

#include <guik/viewer/light_viewer.hpp>

#include <easy_profiler.hpp>

int main(int argc, char** argv) {
  using namespace small_gicp;

  const int num_threads = 6;
  tbb::task_arena arena(num_threads);
  // tbb::task_scheduler_init init(num_threads);

  const std::string dataset_path = "/home/koide/datasets/velodyne";
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

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  for (const auto& filename : filenames) {
    EasyProfiler prof;
    prof.push("read_points");
    const auto raw_points = read_points(filename);

    prof.push("copy");
    auto points = std::make_shared<PointCloud>(raw_points);
    prof.push("downsample");
    auto downsampled = approx_voxelgrid_sampling_tbb(*points, 0.2);
    prof.push("estimate covs");
    estimate_covariances_omp(*downsampled, 10);
    prof.push("create flat voxels");
    auto voxels = std::make_shared<FlatVoxelMap>(0.5, *downsampled);
    prof.push("estimate covs2");
    estimate_covariances_tbb(*downsampled, *voxels, 10);
    prof.push("search");

    prof.push("done");

    viewer->update_points("points", downsampled->points, guik::Rainbow());

    if (!viewer->spin_once()) {
      break;
    }
  }

  return 0;
}