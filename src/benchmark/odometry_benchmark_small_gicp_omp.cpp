#include <small_gicp/benchmark/benchmark_odom.hpp>

#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>

namespace small_gicp {

class SmallGICPOnlineOdometryEstimationOMP : public OnlineOdometryEstimation {
public:
  explicit SmallGICPOnlineOdometryEstimationOMP(const OdometryEstimationParams& params) : OnlineOdometryEstimation(params), T_world_lidar(Eigen::Isometry3d::Identity()) {}

  Eigen::Isometry3d estimate(const PointCloud::Ptr& points) override {
    Stopwatch sw;
    sw.start();

    // Preprocess input points (kdtree construction & covariance estimation)
    // Note that input points here is already downsampled
    auto tree = std::make_shared<KdTree<PointCloud>>(points, KdTreeBuilderOMP(params.num_threads));
    estimate_covariances_omp(*points, *tree, params.num_neighbors, params.num_threads);

    if (target_points == nullptr) {
      // This is the very first frame
      target_points = points;
      target_tree = tree;
      return T_world_lidar;
    }

    // Registration using GICP + OMP-based parallel reduction
    Registration<GICPFactor, ParallelReductionOMP> registration;
    registration.rejector.max_dist_sq = params.max_correspondence_distance * params.max_correspondence_distance;
    registration.reduction.num_threads = params.num_threads;

    // Perform registration
    auto result = registration.align(*target_points, *points, *target_tree, Eigen::Isometry3d::Identity());

    // Update T_world_lidar and target points
    T_world_lidar = T_world_lidar * result.T_target_source;
    target_points = points;
    target_tree = tree;

    sw.stop();
    reg_times.push(sw.msec());

    return T_world_lidar;
  }

  void report() override {  //
    std::cout << "registration_time_stats=" << reg_times.str() << " [msec/scan]  total_throughput=" << total_times.str() << " [msec/scan]" << std::endl;
  }

private:
  Summarizer reg_times;

  PointCloud::Ptr target_points;        // Last point cloud to be registration target
  KdTree<PointCloud>::Ptr target_tree;  // KdTree of the last point cloud

  Eigen::Isometry3d T_world_lidar;  // T_world_lidar
};

static auto small_gicp_omp_registry =
  register_odometry("small_gicp_omp", [](const OdometryEstimationParams& params) { return std::make_shared<SmallGICPOnlineOdometryEstimationOMP>(params); });

}  // namespace small_gicp
