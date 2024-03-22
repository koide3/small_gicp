#pragma once

#include <memory>
#include <random>
#include <unordered_map>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/util/vector3i_hash.hpp>

namespace small_gicp {

template <typename InputPointCloud, typename OutputPointCloud = InputPointCloud>
std::shared_ptr<OutputPointCloud> voxelgrid_sampling(const InputPointCloud& points, double leaf_size) {
  const double inv_leaf_size = 1.0 / leaf_size;

  std::unordered_map<Eigen::Vector3i, Eigen::Vector4d, XORVector3iHash> voxels;
  for (size_t i = 0; i < traits::size(points); i++) {
    const auto& pt = traits::point(points, i);

    const Eigen::Vector3i coord = (pt * inv_leaf_size).array().floor().template cast<int>().template head<3>();
    auto found = voxels.find(coord);
    if (found == voxels.end()) {
      found = voxels.emplace_hint(found, coord, Eigen::Vector4d::Zero());
    }
    found->second += pt;
  }

  auto downsampled = std::make_shared<OutputPointCloud>();
  traits::resize(*downsampled, voxels.size());
  size_t i = 0;
  for (const auto& v : voxels) {
    traits::set_point(*downsampled, i++, v.second / v.second.w());
  }

  return downsampled;
}

template <typename InputPointCloud, typename OutputPointCloud = InputPointCloud>
std::shared_ptr<OutputPointCloud> randomgrid_sampling(const InputPointCloud& points, double leaf_size, size_t target_num_points, std::mt19937& rng) {
  if (traits::size(points) <= target_num_points) {
    auto downsampled = std::make_shared<OutputPointCloud>();
    traits::resize(*downsampled, traits::size(points));
    for (size_t i = 0; i < traits::size(points); i++) {
      traits::set_point(*downsampled, i, traits::point(points, i));
    }
    return downsampled;
  }

  const double inv_leaf_size = 1.0 / leaf_size;

  using Indices = std::shared_ptr<std::vector<size_t>>;
  std::unordered_map<Eigen::Vector3i, Indices, XORVector3iHash> voxels;
  for (size_t i = 0; i < traits::size(points); i++) {
    const auto& pt = traits::point(points, i);

    const Eigen::Vector3i coord = (pt * inv_leaf_size).array().floor().template cast<int>().template head<3>();
    auto found = voxels.find(coord);
    if (found == voxels.end()) {
      found = voxels.emplace_hint(found, coord, std::make_shared<std::vector<size_t>>());
      found->second->reserve(20);
    }

    found->second->emplace_back(i);
  }

  const size_t points_per_voxel = std::ceil(static_cast<double>(target_num_points) / voxels.size());

  std::vector<size_t> indices;
  indices.reserve(points_per_voxel * voxels.size());

  for (const auto& voxel : voxels) {
    const auto& voxel_indices = *voxel.second;
    if (voxel_indices.size() <= points_per_voxel) {
      indices.insert(indices.end(), voxel_indices.begin(), voxel_indices.end());
    } else {
      std::ranges::sample(voxel_indices, std::back_inserter(indices), points_per_voxel, rng);
    }
  }
  std::ranges::sort(indices);

  auto downsampled = std::make_shared<OutputPointCloud>();
  traits::resize(*downsampled, indices.size());
  for (size_t i = 0; i < indices.size(); i++) {
    traits::set_point(*downsampled, i, traits::point(points, indices[i]));
  }

  return downsampled;
}

}  // namespace small_gicp
