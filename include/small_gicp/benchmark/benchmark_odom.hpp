// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/benchmark/benchmark.hpp>

#ifdef BUILD_WITH_IRIDESCENCE
#include <guik/viewer/async_light_viewer.hpp>
#endif

namespace small_gicp {

struct OdometryEstimationParams {
public:
  bool visualize = false;
  int num_threads = 4;
  int num_neighbors = 20;
  double downsampling_resolution = 0.25;
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
  OnlineOdometryEstimation(const OdometryEstimationParams& params) : OdometryEstimation(params), z_range(-5.0f, 5.0f) {}
  ~OnlineOdometryEstimation() {}

  std::vector<Eigen::Isometry3d> estimate(std::vector<PointCloud::Ptr>& points) override {
    std::vector<Eigen::Isometry3d> traj;

    Stopwatch sw;
    sw.start();
    for (size_t i = 0; i < points.size(); i++) {
      if (i && i % 256 == 0) {
        report();
      }

      auto downsampled = voxelgrid_sampling(*points[i], params.downsampling_resolution);
      const Eigen::Isometry3d T = estimate(downsampled);
      traj.emplace_back(T);

#ifdef BUILD_WITH_IRIDESCENCE
      if (params.visualize) {
        auto async_viewer = guik::async_viewer();
        z_range[0] = std::min<double>(z_range[0], T.translation().z() - 5.0f);
        z_range[1] = std::max<double>(z_range[1], T.translation().z() + 5.0f);
        async_viewer->invoke([=] { guik::viewer()->shader_setting().add("z_range", z_range); });
        async_viewer->update_points("current", downsampled->points, guik::FlatOrange(T).set_point_scale(2.0f));
        async_viewer->update_points(guik::anon(), voxelgrid_sampling(*downsampled, 1.0)->points, guik::Rainbow(T));
        async_viewer->lookat(T.translation().cast<float>());
      }
#endif

      points[i].reset();

      sw.lap();
      total_times.push(sw.msec());
    }

    return traj;
  }

  void update_model_points(const Eigen::Isometry3d& T, const std::vector<Eigen::Vector4d>& points) {
    if (!params.visualize) {
      return;
    }

#ifdef BUILD_WITH_IRIDESCENCE
    if (!async_sub_initialized) {
      async_sub_initialized = true;
      async_sub = guik::async_viewer()->async_sub_viewer("model");
    }

    async_sub.update_points("model", points, guik::Rainbow());
    async_sub.lookat(T.translation().cast<float>());
#endif
  }

  virtual Eigen::Isometry3d estimate(const PointCloud::Ptr& points) = 0;

protected:
  Eigen::Vector2f z_range;
  Summarizer total_times;

#ifdef BUILD_WITH_IRIDESCENCE
  bool async_sub_initialized = false;
  guik::AsyncLightViewerContext async_sub;
#endif
};

size_t register_odometry(const std::string& name, std::function<OdometryEstimation::Ptr(const OdometryEstimationParams&)> factory);

std::vector<std::string> odometry_names();

OdometryEstimation::Ptr create_odometry(const std::string& name, const OdometryEstimationParams& params);

}  // namespace small_gicp
