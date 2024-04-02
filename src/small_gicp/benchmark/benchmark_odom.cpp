// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <small_gicp/benchmark/benchmark_odom.hpp>

namespace small_gicp {

std::vector<std::pair<std::string, std::function<OdometryEstimation::Ptr(const OdometryEstimationParams&)>>> odometry_registry;

size_t register_odometry(const std::string& name, std::function<OdometryEstimation::Ptr(const OdometryEstimationParams&)> factory) {
  odometry_registry.emplace_back(name, factory);
  return odometry_registry.size() - 1;
}

std::vector<std::string> odometry_names() {
  std::vector<std::string> names(odometry_registry.size());
  std::transform(odometry_registry.begin(), odometry_registry.end(), names.begin(), [](const auto& p) { return p.first; });
  return names;
}

OdometryEstimation::Ptr create_odometry(const std::string& name, const OdometryEstimationParams& params) {
  auto found = std::find_if(odometry_registry.begin(), odometry_registry.end(), [&](const auto& p) { return p.first == name; });
  if (found == odometry_registry.end()) {
    std::cerr << "error: unknown odometry engine: " << name << std::endl;
    return nullptr;
  }
  return found->second(params);
}

}  // namespace small_gicp
