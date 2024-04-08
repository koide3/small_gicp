#include <small_gicp/benchmark/benchmark_odom.hpp>

#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/util/normal_estimation.hpp>
#include <small_gicp/registration/reduction.hpp>
#include <small_gicp/registration/registration.hpp>

namespace small_gicp {

class SmallGICPOnlineOdometryEstimation : public OnlineOdometryEstimation {
public:
  explicit SmallGICPOnlineOdometryEstimation(const OdometryEstimationParams& params) : OnlineOdometryEstimation(params), T(Eigen::Isometry3d::Identity()) {}

  Eigen::Isometry3d estimate(const PointCloud::Ptr& points) override {
    Stopwatch sw;
    sw.start();

    auto tree = std::make_shared<KdTree<PointCloud>>(points);
    estimate_covariances(*points, *tree, params.num_neighbors);

    if (target_points == nullptr) {
      target_points = points;
      target_tree = tree;
      return T;
    }

    Registration<GICPFactor, SerialReduction> registration;
    registration.rejector.max_dist_sq = params.max_correspondence_distance * params.max_correspondence_distance;

    auto result = registration.align(*target_points, *points, *target_tree, Eigen::Isometry3d::Identity());

    sw.stop();
    reg_times.push(sw.msec());

    T = T * result.T_target_source;
    target_points = points;
    target_tree = tree;

    return T;
  }

  void report() override {  //
    std::cout << "registration_time_stats=" << reg_times.str() << " [msec/scan]  total_throughput=" << total_times.str() << " [msec/scan]" << std::endl;
  }

private:
  Summarizer reg_times;

  PointCloud::Ptr target_points;
  KdTree<PointCloud>::Ptr target_tree;

  Eigen::Isometry3d T;
};

static auto small_gicp_registry =
  register_odometry("small_gicp", [](const OdometryEstimationParams& params) { return std::make_shared<SmallGICPOnlineOdometryEstimation>(params); });

}  // namespace small_gicp