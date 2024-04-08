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

class SmallVGICPOnlineOdometryEstimationTBB : public OnlineOdometryEstimation {
public:
  explicit SmallVGICPOnlineOdometryEstimationTBB(const OdometryEstimationParams& params)
  : OnlineOdometryEstimation(params),
    control(tbb::global_control::max_allowed_parallelism, params.num_threads),
    T(Eigen::Isometry3d::Identity()) {}

  Eigen::Isometry3d estimate(const PointCloud::Ptr& points) override {
    Stopwatch sw;
    sw.start();

    estimate_covariances_tbb(*points, params.num_neighbors);

    auto voxelmap = std::make_shared<GaussianVoxelMap>(params.voxel_resolution);
    voxelmap->insert(*points);

    if (target_points == nullptr) {
      target_points = points;
      target_voxelmap = voxelmap;
      return T;
    }

    Registration<GICPFactor, ParallelReductionTBB> registration;
    registration.rejector.max_dist_sq = params.max_correspondence_distance * params.max_correspondence_distance;

    auto result = registration.align(*target_voxelmap, *points, *target_voxelmap, Eigen::Isometry3d::Identity());

    sw.stop();
    reg_times.push(sw.msec());

    T = T * result.T_target_source;

    target_points = points;
    target_voxelmap = voxelmap;

    return T;
  }

  void report() override {  //
    std::cout << "registration_time_stats=" << reg_times.str() << " [msec/scan]  total_throughput=" << total_times.str() << " [msec/scan]" << std::endl;
  }

private:
  tbb::global_control control;

  Summarizer reg_times;

  PointCloud::Ptr target_points;
  GaussianVoxelMap::Ptr target_voxelmap;

  Eigen::Isometry3d T;
};

static auto small_gicp_tbb_registry =
  register_odometry("small_vgicp_tbb", [](const OdometryEstimationParams& params) { return std::make_shared<SmallVGICPOnlineOdometryEstimationTBB>(params); });

}  // namespace small_gicp

#endif