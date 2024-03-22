#pragma once

#include <fstream>
#include <Eigen/Core>

namespace small_gicp {

static std::vector<Eigen::Vector4f> read_points(const std::string& filename) {
  std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
  if (!ifs) {
    std::cerr << "error: failed to open " << filename << std::endl;
    return {};
  }

  std::streamsize points_bytes = ifs.tellg();
  size_t num_points = points_bytes / (sizeof(Eigen::Vector4f));

  ifs.seekg(0, std::ios::beg);
  std::vector<Eigen::Vector4f> points(num_points);
  ifs.read(reinterpret_cast<char*>(points.data()), sizeof(Eigen::Vector4f) * num_points);

  return points;
}

}  // namespace small_gicp
