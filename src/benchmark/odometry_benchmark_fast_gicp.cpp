#ifdef BUILD_WITH_FAST_GICP

#include <small_gicp/benchmark/benchmark_odom.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/gicp.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/impl/fast_gicp_impl.hpp>
#include <fast_gicp/gicp/impl/lsq_registration_impl.hpp>

namespace small_gicp {

class FastGICPOdometryEstimation : public OnlineOdometryEstimation {
public:
  explicit FastGICPOdometryEstimation(const OdometryEstimationParams& params) : OnlineOdometryEstimation(params), T(Eigen::Isometry3d::Identity()) {
    gicp.setCorrespondenceRandomness(params.num_neighbors);
    gicp.setMaxCorrespondenceDistance(params.max_correspondence_distance);
    gicp.setNumThreads(params.num_threads);
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
      return T;
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

  void report() override {  //
    std::cout << "registration_time_stats=" << reg_times.str() << " [msec/scan]  total_throughput=" << total_times.str() << " [msec/scan]" << std::endl;
  }

private:
  Summarizer reg_times;

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
  Eigen::Isometry3d T;
};

static auto fast_gicp_registry = register_odometry("fast_gicp", [](const OdometryEstimationParams& params) { return std::make_shared<FastGICPOdometryEstimation>(params); });

}  // namespace small_gicp

#endif