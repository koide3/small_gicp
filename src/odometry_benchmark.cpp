#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/kdtree_mt.hpp>
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
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/reduction_tbb.hpp>
#include <small_gicp/registration/registration.hpp>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/impl/fast_gicp_impl.hpp>
#include <fast_gicp/gicp/impl/lsq_registration_impl.hpp>

#include <glk/pointcloud_buffer_pcl.hpp>
#include <guik/viewer/light_viewer.hpp>

namespace small_gicp {

class OdometryEstimater {
public:
  OdometryEstimater() = default;
  virtual ~OdometryEstimater() = default;

  virtual Eigen::Isometry3d estimate(const PointCloud::Ptr& points) = 0;

  const Summarizer& registration_times() const { return reg_times; }

protected:
  Summarizer reg_times;
};

class FastGICPOdometryEstimater : public OdometryEstimater {
public:
  FastGICPOdometryEstimater(int num_threads) : T(Eigen::Isometry3d::Identity()) {
    gicp.setCorrespondenceRandomness(10);
    gicp.setMaxCorrespondenceDistance(1.0);
    gicp.setNumThreads(num_threads);
  }

  Eigen::Isometry3d estimate(const PointCloud::Ptr& points) override {
    auto points_pcl = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    points_pcl->resize(points->size());
    for (size_t i = 0; i < points->size(); i++) {
      points_pcl->at(i).getVector4fMap() = points->point(i).template cast<float>();
    }

    Stopwatch sw;
    sw.start();

    if (!gicp.getInputTarget()) {
      gicp.setInputTarget(points_pcl);
      return Eigen::Isometry3d::Identity();
    }

    gicp.setInputSource(points_pcl);
    pcl::PointCloud<pcl::PointXYZ> aligned;
    gicp.align(aligned);

    sw.stop();
    reg_times.push(sw.msec());

    T = T * Eigen::Isometry3d(gicp.getFinalTransformation().cast<double>());
    gicp.swapSourceAndTarget();

    return T;
  }

private:
  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
  Eigen::Isometry3d T;
};

class SmallGICPOdometryEstimater : public OdometryEstimater {
public:
  SmallGICPOdometryEstimater(int num_threads) : num_threads(num_threads), T(Eigen::Isometry3d::Identity()) {}

  Eigen::Isometry3d estimate(const PointCloud::Ptr& points) override {
    Stopwatch sw;
    sw.start();

    auto tree = std::make_shared<KdTreeMT<PointCloud>>(points);
    estimate_covariances_tbb(*points, *tree, 10);

    if (!target_points) {
      target_points = points;
      target_tree = tree;
      return Eigen::Isometry3d::Identity();
    }

    Registration<GICPFactor, DistanceRejector, ParallelReductionTBB, LevenbergMarquardtOptimizer> registration;
    const auto result = registration.align(*target_points, *points, *target_tree, Eigen::Isometry3d::Identity());

    sw.stop();
    reg_times.push(sw.msec());

    T = T * result.T_target_source;
    target_points = points;
    target_tree = tree;

    return T;
  }

private:
  int num_threads;
  PointCloud::ConstPtr target_points;
  KdTreeMT<PointCloud>::Ptr target_tree;

  Eigen::Isometry3d T;
};

}  // namespace small_gicp

int main(int argc, char** argv) {
  using namespace small_gicp;

  if (argc < 3) {
    std::cout << "usage: odometry_benchmark <dataset_path> <output_path> (--engine small|fast) (--num_threads 4) (--resolution 0.25)" << std::endl;
    return 0;
  }

  const std::string dataset_path = argv[1];
  const std::string output_path = argv[2];

  int num_threads = 4;
  double downsampling_resolution = 0.25;
  std::string engine = "small";

  for (auto arg = argv + 3; arg != argv + argc; arg++) {
    if (std::string(*arg) == "--num_threads") {
      num_threads = std::stoi(*(arg + 1));
    } else if (std::string(*arg) == "--resolution") {
      downsampling_resolution = std::stod(*(arg + 1));
    } else if (std::string(*arg) == "--engine") {
      engine = *(arg + 1);
    }
  }

  std::cout << "dataset_path=" << dataset_path << std::endl;
  std::cout << "output_path=" << output_path << std::endl;
  std::cout << "registration_engine=" << engine << std::endl;
  std::cout << "num_threads=" << num_threads << std::endl;
  std::cout << "downsampling_resolution=" << downsampling_resolution << std::endl;

  tbb::global_control control(tbb::global_control::max_allowed_parallelism, num_threads);

  std::shared_ptr<OdometryEstimater> odom;
  if (engine == "small") {
    odom = std::make_shared<SmallGICPOdometryEstimater>(num_threads);
  } else if (engine == "fast") {
    odom = std::make_shared<FastGICPOdometryEstimater>(num_threads);
  } else {
    std::cerr << "Unknown engine: " << engine << std::endl;
    return 1;
  }

  KittiDataset kitti(dataset_path);
  std::cout << "num_frames=" << kitti.points.size() << std::endl;
  std::cout << fmt::format("num_points={} [points]", summarize(kitti.points, [](const auto& pts) { return pts.size(); })) << std::endl;

  auto raw_points = kitti.convert<PointCloud>(true);

  std::vector<Eigen::Isometry3d> traj;
  for (size_t i = 0; i < raw_points.size(); i++) {
    auto downsampled = voxelgrid_sampling(*raw_points[i], downsampling_resolution);
    const Eigen::Isometry3d T = odom->estimate(downsampled);

    if (i && i % 256 == 0) {
      std::cout << fmt::format("{}/{} : {}", i, raw_points.size(), odom->registration_times().str()) << std::endl;
    }
    traj.emplace_back(T);
  }

  std::cout << "registration_time_stats=" << odom->registration_times().str() << std::endl;

  std::ofstream ofs(output_path);
  for (const auto& T : traj) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4; j++) {
        if (i || j) {
          ofs << " ";
        }

        ofs << fmt::format("{:.6f}", T(i, j));
      }
    }
    ofs << std::endl;
  }

  return 0;
}
