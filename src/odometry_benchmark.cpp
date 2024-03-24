#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/util/downsampling_tbb.hpp>
#include <small_gicp/util/normal_estimation_tbb.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/benchmark.hpp>

#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/factors/plane_icp_factor.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/registration/reduction_tbb.hpp>
#include <small_gicp/registration/registration.hpp>

#include <guik/viewer/light_viewer.hpp>

int main(int argc, char** argv) {
  using namespace small_gicp;

  std::cout << "SIMD in use: " << Eigen::SimdInstructionSetsInUse() << std::endl;
  const std::string dataset_path = "/home/koide/datasets/velodyne";

  std::cout << "Load dataset from " << dataset_path << std::endl;
  KittiDataset kitti(dataset_path, 1000);
  std::cout << "num_frames=" << kitti.points.size() << std::endl;
  std::cout << fmt::format("num_points={} [points]", summarize(kitti.points, [](const auto& pts) { return pts.size(); })) << std::endl;

  auto raw_points = kitti.convert<PointCloud>();
  kitti.points.clear();

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  auto target = voxelgrid_sampling_tbb(*raw_points.front(), 0.25);
  estimate_normals_covariances_tbb(*target, 10);
  auto target_tree = std::make_shared<KdTree<PointCloud>>(target);

  auto viewer = guik::viewer();
  viewer->disable_vsync();

  Stopwatch sw;
  Summarizer preprocess_times;
  Summarizer registration_times;

  for (size_t i = 0; i < raw_points.size() && viewer->spin_once(); i++) {
    sw.start();
    auto points = voxelgrid_sampling_tbb(*raw_points[i], 0.25);
    auto tree = std::make_shared<KdTree<PointCloud>>(points);
    estimate_normals_covariances_tbb(*points, *tree, 10);
    sw.lap();
    preprocess_times.push(sw.msec());

    Registration<GICPFactor, DistanceRejector, ParallelReductionTBB, LevenbergMarquardtOptimizer> registration;
    auto result = registration.align(*target, *points, *target_tree, Eigen::Isometry3d::Identity());
    sw.lap();
    registration_times.push(sw.msec());

    std::cout << "preprocess=" << preprocess_times.str() << "[msec]  registration=" << registration_times.str() << "[msec]" << std::endl;

    T = T * result.T_target_source;

    target = points;
    target_tree = tree;

    viewer->update_points("current", points->points, guik::FlatOrange(T).set_point_scale(2.0f));
    viewer->update_points(guik::anon(), voxelgrid_sampling_tbb(*points, 1.0)->points, guik::Rainbow(T));
  }

  return 0;
}
