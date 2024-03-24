#pragma once

#include <fstream>
#include <iostream>
#include <Eigen/Core>

namespace small_gicp {

/// @brief Read points from file (simple float4 array)
/// @param filename  Filename
/// @return          Points
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
  for (auto& pt : points) {
    pt(3) = 1.0;
  }

  return points;
}

}  // namespace small_gicp
