#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/benchmark/benchmark.hpp>

#include <guik/viewer/async_light_viewer.hpp>

namespace small_gicp {

struct OdometryEstimationParams {
public:
  int num_threads = 4;
  double downsample_resolution = 0.25;
  double voxel_resolution = 1.0;
  double max_correspondence_distance = 1.0;
};

class OdometryEstimation {
public:
  using Ptr = std::shared_ptr<OdometryEstimation>;

  OdometryEstimation(const OdometryEstimationParams& params) : params(params) {}
  virtual ~OdometryEstimation() = default;

  virtual std::vector<Eigen::Isometry3d> estimate(std::vector<PointCloud::Ptr>& points) = 0;

  virtual void report() {}

protected:
  const OdometryEstimationParams params;
};

class OnlineOdometryEstimation : public OdometryEstimation {
public:
  OnlineOdometryEstimation(const OdometryEstimationParams& params) : OdometryEstimation(params) {}
  ~OnlineOdometryEstimation() { guik::async_destroy(); }

  std::vector<Eigen::Isometry3d> estimate(std::vector<PointCloud::Ptr>& points) override {
    std::vector<Eigen::Isometry3d> traj;

    Stopwatch sw;
    sw.start();
    for (size_t i = 0; i < points.size(); i++) {
      if (i && i % 256 == 0) {
        report();
      }

      auto downsampled = voxelgrid_sampling(*points[i], params.downsample_resolution);
      const Eigen::Isometry3d T = estimate(downsampled);
      traj.emplace_back(T);

      auto async_viewer = guik::async_viewer();
      async_viewer->update_points("current", downsampled->points, guik::FlatOrange(T).set_point_scale(2.0f));
      async_viewer->update_points(guik::anon(), voxelgrid_sampling(*downsampled, 1.0)->points, guik::Rainbow(T));

      points[i].reset();

      sw.lap();
      total_times.push(sw.msec());
    }

    return traj;
  }

  virtual Eigen::Isometry3d estimate(const PointCloud::Ptr& points) = 0;

protected:
  Summarizer total_times;
};

size_t register_odometry(const std::string& name, std::function<OdometryEstimation::Ptr(const OdometryEstimationParams&)> factory);

std::vector<std::string> odometry_names();

OdometryEstimation::Ptr create_odometry(const std::string& name, const OdometryEstimationParams& params);

}  // namespace small_gicp
