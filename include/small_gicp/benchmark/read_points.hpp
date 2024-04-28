// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <Eigen/Core>

namespace small_gicp {

/// @brief Read points from file (simple float4 array).
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

/// @brief Write points to file (simple float4 array).
/// @param filename  Filename
/// @param points    Points
static void write_points(const std::string& filename, const std::vector<Eigen::Vector4f>& points) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    std::cerr << "error: failed to open " << filename << std::endl;
    return;
  }

  ofs.write(reinterpret_cast<const char*>(points.data()), sizeof(Eigen::Vector4f) * points.size());
}

/// @brief Read point cloud from a PLY file.
/// @note  This function can only handle simple PLY files with float properties (Property names must be "x", "y", "z"). Do not use this for general PLY IO.
/// @param filename  Filename
/// @return          Points
static std::vector<Eigen::Vector4f> read_ply(const std::string& filename) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    std::cerr << "error: failed to open " << filename << std::endl;
    return {};
  }

  std::vector<std::string> properties;
  std::vector<Eigen::Vector4f> points;

  std::string line;
  while (!ifs.eof() && std::getline(ifs, line) && !line.empty()) {
    if (line == "end_header") {
      break;
    }

    if (line.find("element") == 0) {
      std::stringstream sst(line);
      std::string token, vertex, num_points;
      sst >> token >> vertex >> num_points;
      if (token != "element" || vertex != "vertex") {
        std::cerr << "error: invalid ply format (line=" << line << ")" << std::endl;
        return {};
      }

      points.resize(std::stol(num_points));
    } else if (line.find("property") == 0) {
      std::stringstream sst(line);
      std::string token, type, name;
      sst >> token >> type >> name;
      if (type != "float") {
        std::cerr << "error: only float properties are supported!! (line=" << line << ")" << std::endl;
        return {};
      }

      properties.emplace_back(name);
    }
  }

  if (
    line.size() < 3 || properties[0].size() != 1 || properties[1].size() != 1 || properties[2].size() != 1 || std::tolower(properties[0][0]) != 'x' ||
    std::tolower(properties[1][0]) != 'y' || std::tolower(properties[2][0]) != 'z') {
    std::cerr << "warning: invalid properties!!" << std::endl;
    for (const auto& prop : properties) {
      std::cerr << " - " << prop << std::endl;
    }
  }

  std::vector<float> buffer(properties.size() * points.size());
  ifs.read(reinterpret_cast<char*>(buffer.data()), sizeof(Eigen::Vector4f) * points.size());

  for (size_t i = 0; i < points.size(); i++) {
    const int stride = properties.size();
    points[i] = Eigen::Vector4f(buffer[i * stride + 0], buffer[i * stride + 1], buffer[i * stride + 2], 1.0);
  }

  return points;
}

}  // namespace small_gicp
