#ifdef BUILD_WITH_PCL

#include <small_gicp/benchmark/benchmark_odom.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/pcl/pcl_registration_impl.hpp>

namespace small_gicp {

class SmallGICPPCLOdometryEstimation : public OnlineOdometryEstimation {
public:
  explicit SmallGICPPCLOdometryEstimation(const OdometryEstimationParams& params) : OnlineOdometryEstimation(params), T(Eigen::Isometry3d::Identity()) {
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

  small_gicp::RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ> gicp;
  Eigen::Isometry3d T;
};

static auto small_gicp_pcl_registry =
  register_odometry("small_gicp_pcl", [](const OdometryEstimationParams& params) { return std::make_shared<SmallGICPPCLOdometryEstimation>(params); });

}  // namespace small_gicp

#endif