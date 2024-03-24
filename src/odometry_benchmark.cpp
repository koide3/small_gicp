#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_tbb.hpp>
#include <small_gicp/util/normal_estimation.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/util/normal_estimation_tbb.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/points/pcl_point.hpp>
#include <small_gicp/points/pcl_point_traits.hpp>
#include <small_gicp/util/benchmark.hpp>

#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/factors/plane_icp_factor.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/registration/reduction_tbb.hpp>
#include <small_gicp/registration/registration.hpp>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/impl/fast_gicp_impl.hpp>
#include <fast_gicp/gicp/impl/lsq_registration_impl.hpp>

#include <glk/pointcloud_buffer_pcl.hpp>
#include <guik/viewer/light_viewer.hpp>

template <typename PointT>
pcl::PointCloud<pcl::PointXYZ>::Ptr convert_pcl(const pcl::PointCloud<PointT>& points) {
  auto converted = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  converted->is_dense = false;
  converted->width = points.size();
  converted->height = 1;

  converted->resize(points.size());
  for (size_t i = 0; i < points.size(); i++) {
    converted->at(i).getVector4fMap() = points.at(i).getVector4fMap();
  }
  return converted;
}

int main(int argc, char** argv) {
  using namespace small_gicp;

  std::cout << "SIMD in use: " << Eigen::SimdInstructionSetsInUse() << std::endl;
  const std::string dataset_path = "/home/koide/datasets/velodyne";

  std::cout << "Load dataset from " << dataset_path << std::endl;
  KittiDataset kitti(dataset_path, 1000);
  std::cout << "num_frames=" << kitti.points.size() << std::endl;
  std::cout << fmt::format("num_points={} [points]", summarize(kitti.points, [](const auto& pts) { return pts.size(); })) << std::endl;

  auto raw_points = kitti.convert<pcl::PointCloud<pcl::PointCovariance>>();
  kitti.points.clear();

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  auto target = voxelgrid_sampling(*raw_points.front(), 0.25);
  estimate_covariances_tbb(*target, 10);
  auto target_tree = std::make_shared<KdTree<pcl::PointCloud<pcl::PointCovariance>>>(target);

  auto viewer = guik::viewer();
  viewer->disable_vsync();

  Stopwatch sw;
  Summarizer gicp_times(10);
  Summarizer registration_times(10);

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
  gicp.setCorrespondenceRandomness(10);
  gicp.setMaxCorrespondenceDistance(1.0);
  gicp.setInputTarget(convert_pcl(*target));
  gicp.setNumThreads(6);

  for (size_t i = 0; i < raw_points.size() && viewer->spin_once(); i++) {
    auto points = voxelgrid_sampling(*raw_points[i], 0.25);
    auto source_pts = convert_pcl(*points);

    /*
    sw.start();
    gicp.setInputSource(source_pts);
    pcl::PointCloud<pcl::PointXYZ> aligned;
    gicp.align(aligned);
    const Eigen::Matrix4f T_ = gicp.getFinalTransformation();
    T = T * Eigen::Isometry3d(T_.cast<double>());
    sw.stop();

    gicp.swapSourceAndTarget();
    gicp_times.push(sw.msec());
    //*/

    //*
    sw.start();
    auto tree = std::make_shared<KdTree<pcl::PointCloud<pcl::PointCovariance>>>(points);
    estimate_covariances_tbb(*points, *tree, 10);
    Registration<GICPFactor, DistanceRejector, ParallelReductionTBB, LevenbergMarquardtOptimizer> registration;
    auto result = registration.align(*target, *points, *target_tree, Eigen::Isometry3d::Identity());
    sw.stop();
    registration_times.push(sw.msec());

    T = T * result.T_target_source;

    target = points;
    target_tree = tree;
    //*/

    std::cout << "gicp=" << gicp_times.str() << "[msec]  registration=" << registration_times.str() << "[msec]" << std::endl;

    auto cloud_buffer = glk::create_point_cloud_buffer(*points);
    viewer->update_drawable("current", cloud_buffer, guik::FlatOrange(T).set_point_scale(2.0f));
    viewer->update_drawable(guik::anon(), cloud_buffer, guik::Rainbow(T));
  }

  return 0;
}
