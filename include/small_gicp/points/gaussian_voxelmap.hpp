#pragma once

#include <unordered_map>

#include <small_gicp/ann/traits.hpp>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/util/vector3i_hash.hpp>

namespace small_gicp {

struct GaussianVoxel {
public:
  GaussianVoxel(const Eigen::Vector3i& coord) : finalized(false), lru(0), num_points(0), coord(coord), mean(Eigen::Vector4d::Zero()), cov(Eigen::Matrix4d::Zero()) {}
  ~GaussianVoxel() {}

  void add(const Eigen::Vector4d& mean, const Eigen::Matrix4d& cov, size_t lru) {
    if (finalized) {
      this->finalized = false;
      this->mean *= num_points;
      this->cov *= num_points;
    }

    num_points++;
    this->mean += mean;
    this->cov += cov;
    this->lru = lru;
  }

  void finalize() {
    if (finalized) {
      return;
    }

    mean /= num_points;
    cov /= num_points;
    finalized = true;
  }

public:
  bool finalized;
  size_t lru;
  size_t num_points;
  Eigen::Vector3i coord;

  Eigen::Vector4d mean;
  Eigen::Matrix4d cov;
};

struct GaussianVoxelMap {
public:
  using Ptr = std::shared_ptr<GaussianVoxelMap>;
  using ConstPtr = std::shared_ptr<const GaussianVoxelMap>;

  GaussianVoxelMap(double leaf_size) : inv_leaf_size(1.0 / leaf_size), lru_horizon(100), lru_clear_cycle(10), lru_counter(0) {}
  ~GaussianVoxelMap() {}

  size_t size() const { return flat_voxels.size(); }

  template <typename PointCloud>
  void insert(const PointCloud& points, const Eigen::Isometry3d& T = Eigen::Isometry3d::Identity()) {
    for (size_t i = 0; i < traits::size(points); i++) {
      const Eigen::Vector4d pt = T * traits::point(points, i);
      const Eigen::Vector3i coord = (pt * inv_leaf_size).array().floor().cast<int>().head<3>();

      auto found = voxels.find(coord);
      if (found == voxels.end()) {
        found = voxels.emplace_hint(found, coord, flat_voxels.size());
        flat_voxels.emplace_back(coord);
      }

      auto& voxel = flat_voxels[found->second];
      const Eigen::Matrix4d cov = T.matrix() * traits::cov(points, i) * T.matrix().transpose();
      voxel.add(pt, cov, lru_counter);
    }

    if ((lru_counter++) % lru_clear_cycle == 0) {
      std::erase_if(flat_voxels, [&](const GaussianVoxel& voxel) { return voxel.lru + lru_horizon < lru_counter; });
      voxels.clear();
      for (size_t i = 0; i < flat_voxels.size(); i++) {
        voxels[flat_voxels[i].coord] = i;
      }
    }

    for (auto& voxel : flat_voxels) {
      voxel.finalize();
    }
  }

  size_t knn_search(const Eigen::Vector4d& pt, size_t k, size_t* k_indices, double* k_sq_dists) const {
    if (k != 1) {
      std::cerr << "warning:!!" << std::endl;
    }

    const Eigen::Vector3i coord = (pt * inv_leaf_size).array().floor().cast<int>().head<3>();
    const auto found = voxels.find(coord);
    if (found == voxels.end()) {
      return 0;
    }

    const GaussianVoxel& voxel = flat_voxels[found->second];
    k_indices[0] = found->second;
    k_sq_dists[0] = (voxel.mean - pt).squaredNorm();
    return 1;
  }

public:
  const double inv_leaf_size;
  size_t lru_horizon;
  size_t lru_clear_cycle;
  size_t lru_counter;

  std::vector<GaussianVoxel> flat_voxels;
  std::unordered_map<Eigen::Vector3i, size_t, XORVector3iHash> voxels;
};

namespace traits {

template <>
struct Traits<GaussianVoxelMap> {
  static size_t size(const GaussianVoxelMap& voxelmap) { return voxelmap.size(); }

  static bool has_points(const GaussianVoxelMap& voxelmap) { return !voxelmap.flat_voxels.empty(); }
  static bool has_covs(const GaussianVoxelMap& voxelmap) { return !voxelmap.flat_voxels.empty(); }

  static const Eigen::Vector4d& point(const GaussianVoxelMap& voxelmap, size_t i) { return voxelmap.flat_voxels[i].mean; }
  static const Eigen::Matrix4d& cov(const GaussianVoxelMap& voxelmap, size_t i) { return voxelmap.flat_voxels[i].cov; }

  static size_t knn_search(const GaussianVoxelMap& voxelmap, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
    return voxelmap.knn_search(point, k, k_indices, k_sq_dists);
  }
};

}  // namespace traits

}  // namespace small_gicp
