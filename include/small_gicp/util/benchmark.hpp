#pragma once

#include <chrono>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <Eigen/Core>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/util/read_points.hpp>

namespace small_gicp {

struct Stopwatch {
public:
  void start() { t1 = std::chrono::high_resolution_clock::now(); }
  void stop() { t2 = std::chrono::high_resolution_clock::now(); }
  double elapsed_sec() const { return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e9; }
  double elapsed_msec() const { return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6; }

public:
  std::chrono::high_resolution_clock::time_point t1;
  std::chrono::high_resolution_clock::time_point t2;
};

struct KittiDataset {
public:
  KittiDataset(const std::string& dataset_path, size_t max_num_data = std::numeric_limits<size_t>::max()) {
    std::vector<std::string> filenames;
    for (auto path : std::filesystem::directory_iterator(dataset_path)) {
      if (path.path().extension() != ".bin") {
        continue;
      }

      filenames.emplace_back(path.path().string());
    }

    std::ranges::sort(filenames);
    if (filenames.size() > max_num_data) {
      filenames.resize(max_num_data);
    }

    points.resize(max_num_data);
    std::ranges::transform(filenames, points.begin(), [](const std::string& filename) { return read_points(filename); });
    std::erase_if(points, [](const auto& p) { return p.empty(); });
  }

  template <typename PointCloud>
  std::vector<std::shared_ptr<PointCloud>> convert() const {
    std::vector<std::shared_ptr<PointCloud>> converted(points.size());
    std::ranges::transform(points, converted.begin(), [](const auto& raw_points) {
      auto points = std::make_shared<PointCloud>();
      traits::resize(*points, raw_points.size());
      for (size_t i = 0; i < raw_points.size(); i++) {
        traits::set_point(*points, i, raw_points[i].template cast<double>());
      }
      return points;
    });
    return converted;
  }

public:
  std::vector<std::vector<Eigen::Vector4f>> points;
};

}  // namespace small_gicp
