#pragma once

#include <chrono>
#include <deque>
#include <vector>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <Eigen/Core>
#include <fmt/format.h>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/util/read_points.hpp>

namespace small_gicp {

struct Stopwatch {
public:
  void start() { t1 = t2 = std::chrono::high_resolution_clock::now(); }
  void stop() { t2 = std::chrono::high_resolution_clock::now(); }
  void lap() {
    t1 = t2;
    t2 = std::chrono::high_resolution_clock::now();
  }

  double sec() const { return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e9; }
  double msec() const { return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6; }

public:
  std::chrono::high_resolution_clock::time_point t1;
  std::chrono::high_resolution_clock::time_point t2;
};

struct Summarizer {
public:
  Summarizer() : num_data(0), sum(0.0), sq_sum(0.0), last_x(0.0) {}

  void push(double x) {
    num_data++;
    sum += x;
    sq_sum += x * x;
    last_x = x;
  }

  std::pair<double, double> mean_std() const {
    const double mean = sum / num_data;
    const double var = (sq_sum - mean * sum) / num_data;
    return {mean, std::sqrt(var)};
  }

  double last() const { return last_x; }

  std::string str() const {
    const auto [mean, std] = mean_std();
    return fmt::format("{:.3f} +- {:.3f} (last={:.3f})", mean, std, last_x);
  }

private:
  size_t num_data;
  double sum;
  double sq_sum;
  double last_x;
};

template <typename Container, typename Transform>
std::string summarize(const Container& container, const Transform& transform) {
  Summarizer summarizer;
  for (auto itr = std::begin(container); itr != std::end(container); itr++) {
    summarizer.push(transform(*itr));
  }
  return summarizer.str();
}

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

    points.resize(filenames.size());
    std::ranges::transform(filenames, points.begin(), [](const std::string& filename) { return read_points(filename); });
  }

  template <typename PointCloud>
  std::vector<std::shared_ptr<PointCloud>> convert(bool release = false) {
    std::vector<std::shared_ptr<PointCloud>> converted(points.size());
    std::ranges::transform(points, converted.begin(), [=](auto& raw_points) {
      auto points = std::make_shared<PointCloud>();
      traits::resize(*points, raw_points.size());
      for (size_t i = 0; i < raw_points.size(); i++) {
        traits::set_point(*points, i, raw_points[i].template cast<double>());
      }

      if (release) {
        raw_points.clear();
        raw_points.shrink_to_fit();
      }
      return points;
    });

    if (release) {
      points.clear();
      points.shrink_to_fit();
    }

    return converted;
  }

public:
  std::vector<std::vector<Eigen::Vector4f>> points;
};

}  // namespace small_gicp
