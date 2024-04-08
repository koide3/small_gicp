#ifdef BUILD_WITH_PCL

#include <small_gicp/benchmark/benchmark_odom.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/gicp.h>

namespace small_gicp {

class PCLOnlineOdometryEstimation : public OnlineOdometryEstimation {
public:
  explicit PCLOnlineOdometryEstimation(const OdometryEstimationParams& params) : OnlineOdometryEstimation(params), T(Eigen::Isometry3d::Identity()) {
    gicp.setCorrespondenceRandomness(params.num_neighbors);
    gicp.setMaxCorrespondenceDistance(params.max_correspondence_distance);
  }

  Eigen::Isometry3d estimate(const PointCloud::Ptr& points) override {
    auto points_pcl = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    points_pcl->resize(points->size());
    for (size_t i = 0; i < points->size(); i++) {
      points_pcl->at(i).getVector4fMap() = points->point(i).template cast<float>();
    }

    Stopwatch sw;
    sw.start();

    if (!target_points) {
      target_points = points_pcl;
      return Eigen::Isometry3d::Identity();
    }

    gicp.setInputTarget(target_points);
    gicp.setInputSource(points_pcl);
    pcl::PointCloud<pcl::PointXYZ> aligned;
    gicp.align(aligned);

    sw.stop();
    reg_times.push(sw.msec());

    T = T * Eigen::Isometry3d(gicp.getFinalTransformation().cast<double>());
    target_points = points_pcl;

    return T;
  }

  void report() override {  //
    std::cout << "registration_time_stats=" << reg_times.str() << " [msec/scan]  total_throughput=" << total_times.str() << " [msec/scan]" << std::endl;
  }

private:
  Summarizer reg_times;

  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_points;
  Eigen::Isometry3d T;
};

static auto pcl_odom_registry = register_odometry("pcl", [](const OdometryEstimationParams& params) { return std::make_shared<PCLOnlineOdometryEstimation>(params); });

}  // namespace small_gicp

#endif