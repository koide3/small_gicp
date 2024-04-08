#ifdef BUILD_WITH_TBB

#include <small_gicp/benchmark/benchmark_odom.hpp>

#include <tbb/tbb.h>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/util/normal_estimation_tbb.hpp>
#include <small_gicp/registration/reduction_tbb.hpp>
#include <small_gicp/registration/registration.hpp>

namespace small_gicp {

class SmallVGICPModelOnlineOdometryEstimationTBB : public OnlineOdometryEstimation {
public:
  explicit SmallVGICPModelOnlineOdometryEstimationTBB(const OdometryEstimationParams& params)
  : OnlineOdometryEstimation(params),
    control(tbb::global_control::max_allowed_parallelism, params.num_threads),
    T_world_lidar(Eigen::Isometry3d::Identity()) {}

  Eigen::Isometry3d estimate(const PointCloud::Ptr& points) override {
    Stopwatch sw;
    sw.start();

    // Note that input points here is already downsampled
    // Estimate per-point covariances
    estimate_covariances_tbb(*points, params.num_neighbors);

    if (voxelmap == nullptr) {
      // This is the very first frame
      voxelmap = std::make_shared<GaussianVoxelMap>(params.voxel_resolution);
      voxelmap->insert(*points);
      return T_world_lidar;
    }

    // Registration using GICP + TBB-based parallel reduction
    Registration<GICPFactor, ParallelReductionTBB> registration;
    auto result = registration.align(*voxelmap, *points, *voxelmap, T_world_lidar);

    // Update T_world_lidar with the estimated transformation
    T_world_lidar = result.T_target_source;

    // Insert points to the target voxel map
    voxelmap->insert(*points, T_world_lidar);

    sw.stop();
    reg_times.push(sw.msec());

    if (params.visualize) {
      update_model_points(T_world_lidar, traits::voxel_points(*voxelmap));
    }

    return T_world_lidar;
  }

  void report() override {  //
    std::cout << "registration_time_stats=" << reg_times.str() << " [msec/scan]  total_throughput=" << total_times.str() << " [msec/scan]" << std::endl;
  }

private:
  tbb::global_control control;

  Summarizer reg_times;

  GaussianVoxelMap::Ptr voxelmap;   // Target voxel map that is an accumulation of past point clouds
  Eigen::Isometry3d T_world_lidar;  // Current world-to-lidar transformation
};

static auto small_gicp_model_tbb_registry =
  register_odometry("small_vgicp_model_tbb", [](const OdometryEstimationParams& params) { return std::make_shared<SmallVGICPModelOnlineOdometryEstimationTBB>(params); });

}  // namespace small_gicp

#endif