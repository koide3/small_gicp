#ifdef BUILD_WITH_FAST_GICP

#include <small_gicp/benchmark/benchmark_odom.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/gicp.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <fast_gicp/gicp/impl/fast_gicp_impl.hpp>
#include <fast_gicp/gicp/impl/fast_vgicp_impl.hpp>
#include <fast_gicp/gicp/impl/lsq_registration_impl.hpp>

namespace small_gicp {

class FastVGICPOdometryEstimation : public OnlineOdometryEstimation {
public:
  explicit FastVGICPOdometryEstimation(const OdometryEstimationParams& params) : OnlineOdometryEstimation(params), T(Eigen::Isometry3d::Identity()) {
    vgicp.setCorrespondenceRandomness(params.num_neighbors);
    vgicp.setResolution(params.voxel_resolution);
    vgicp.setMaxCorrespondenceDistance(params.max_correspondence_distance);
    vgicp.setNumThreads(params.num_threads);
  }

  Eigen::Isometry3d estimate(const PointCloud::Ptr& points) override {
    auto points_pcl = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    points_pcl->resize(points->size());
    for (size_t i = 0; i < points->size(); i++) {
      points_pcl->at(i).getVector4fMap() = points->point(i).template cast<float>();
    }

    Stopwatch sw;
    sw.start();

    if (!vgicp.getInputTarget()) {
      vgicp.setInputTarget(points_pcl);
      return T;
    }

    vgicp.setInputSource(points_pcl);
    pcl::PointCloud<pcl::PointXYZ> aligned;
    vgicp.align(aligned);

    sw.stop();
    reg_times.push(sw.msec());

    T = T * Eigen::Isometry3d(vgicp.getFinalTransformation().cast<double>());
    vgicp.swapSourceAndTarget();

    return T;
  }

  void report() override {  //
    std::cout << "registration_time_stats=" << reg_times.str() << " [msec/scan]  total_throughput=" << total_times.str() << " [msec/scan]" << std::endl;
  }

private:
  Summarizer reg_times;

  fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> vgicp;
  Eigen::Isometry3d T;
};

static auto fast_vgicp_registry = register_odometry("fast_vgicp", [](const OdometryEstimationParams& params) { return std::make_shared<FastVGICPOdometryEstimation>(params); });

}  // namespace small_gicp

#endif