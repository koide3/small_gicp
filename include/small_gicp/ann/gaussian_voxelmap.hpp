#pragma once

#include <unordered_map>

#include <small_gicp/ann/traits.hpp>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/util/vector3i_hash.hpp>

namespace small_gicp {

/// @brief Gaussian Voxel
struct GaussianVoxel {
public:
  /// @brief Constructor
  /// @param coord  Voxel coordinates
  GaussianVoxel(const Eigen::Vector3i& coord) : finalized(false), lru(0), num_points(0), coord(coord), mean(Eigen::Vector4d::Zero()), cov(Eigen::Matrix4d::Zero()) {}
  ~GaussianVoxel() {}

  /// @brief Add a point (Gaussian distribution) to the voxel
  /// @param mean  Mean of the point
  /// @param cov   Covariance of the point
  /// @param lru   LRU cache counter
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

  /// @brief Finalize mean and covariance
  void finalize() {
    if (finalized) {
      return;
    }

    mean /= num_points;
    cov /= num_points;
    finalized = true;
  }

public:
  bool finalized;         ///< If true, mean and cov are finalized, otherwise they represent the sum of input points
  size_t lru;             ///< LRU counter
  size_t num_points;      ///< Number of input points
  Eigen::Vector3i coord;  ///< Voxel coordinates

  Eigen::Vector4d mean;  ///< Mean
  Eigen::Matrix4d cov;   ///< Covariance
};

/// @brief Gaussian VoxelMap.
///        This class can be used as PointCloud as well as NearestNeighborSearch.
///        It also supports incremental points insertion and LRU-based voxel deletion.
struct GaussianVoxelMap {
public:
  using Ptr = std::shared_ptr<GaussianVoxelMap>;
  using ConstPtr = std::shared_ptr<const GaussianVoxelMap>;

  /// @brief Constructor
  /// @param leaf_size  Voxel resolution
  GaussianVoxelMap(double leaf_size) : inv_leaf_size(1.0 / leaf_size), lru_horizon(100), lru_clear_cycle(10), lru_counter(0) {}
  ~GaussianVoxelMap() {}

  /// @brief Number of voxels
  size_t size() const { return flat_voxels.size(); }

  /// @brief Insert points to the voxelmap
  /// @param points  Input points
  /// @param T       Transformation
  template <typename PointCloud>
  void insert(const PointCloud& points, const Eigen::Isometry3d& T = Eigen::Isometry3d::Identity()) {
    // Insert points to the voxelmap
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

    if ((++lru_counter) % lru_clear_cycle == 0) {
      // Remove least recently used voxels
      std::erase_if(flat_voxels, [&](const GaussianVoxel& voxel) { return voxel.lru + lru_horizon < lru_counter; });
      voxels.clear();

      // Rehash
      for (size_t i = 0; i < flat_voxels.size(); i++) {
        voxels[flat_voxels[i].coord] = i;
      }
    }

    // Finalize voxel means and covs
    for (auto& voxel : flat_voxels) {
      voxel.finalize();
    }
  }

  /// @brief Find the nearest neighbor
  size_t nearest_neighbor_search(const Eigen::Vector4d& pt, size_t* k_index, double* k_sq_dist) const {
    const Eigen::Vector3i coord = (pt * inv_leaf_size).array().floor().cast<int>().head<3>();
    const auto found = voxels.find(coord);
    if (found == voxels.end()) {
      return 0;
    }

    const GaussianVoxel& voxel = flat_voxels[found->second];
    *k_index = found->second;
    *k_sq_dist = (voxel.mean - pt).squaredNorm();
    return 1;
  }

public:
  const double inv_leaf_size;  ///< Inverse of the voxel resolution
  size_t lru_horizon;          ///< LRU horizon size
  size_t lru_clear_cycle;      ///< LRU clear cycle
  size_t lru_counter;          ///< LRU counter

  std::vector<GaussianVoxel> flat_voxels;                               ///< Voxelmap contents
  std::unordered_map<Eigen::Vector3i, size_t, XORVector3iHash> voxels;  ///< Voxel index map
};

namespace traits {

template <>
struct Traits<GaussianVoxelMap> {
  static size_t size(const GaussianVoxelMap& voxelmap) { return voxelmap.size(); }

  static bool has_points(const GaussianVoxelMap& voxelmap) { return !voxelmap.flat_voxels.empty(); }
  static bool has_covs(const GaussianVoxelMap& voxelmap) { return !voxelmap.flat_voxels.empty(); }

  static const Eigen::Vector4d& point(const GaussianVoxelMap& voxelmap, size_t i) { return voxelmap.flat_voxels[i].mean; }
  static const Eigen::Matrix4d& cov(const GaussianVoxelMap& voxelmap, size_t i) { return voxelmap.flat_voxels[i].cov; }

  static size_t nearest_neighbor_search(const GaussianVoxelMap& voxelmap, const Eigen::Vector4d& point, size_t* k_index, double* k_sq_dist) {
    return voxelmap.nearest_neighbor_search(point, k_index, k_sq_dist);
  }
};

}  // namespace traits

}  // namespace small_gicp
