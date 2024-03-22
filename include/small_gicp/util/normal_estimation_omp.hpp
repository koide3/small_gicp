#pragma once

#include <small_gicp/util/normal_estimation.hpp>

namespace small_gicp {

template <typename PointCloud, typename Setter>
void estimate_local_features_omp(PointCloud& cloud, int num_neighbors, int num_threads) {
  traits::resize(cloud, traits::size(cloud));

  UnsafeKdTree<PointCloud> kdtree(cloud);

#pragma omp parallel for num_threads(num_threads)
  for (size_t i = 0; i < traits::size(cloud); i++) {
    estimate_local_features<PointCloud, UnsafeKdTree<PointCloud>, Setter>(cloud, kdtree, num_neighbors, i);
  }
}

template <typename PointCloud>
void estimate_normals_omp(PointCloud& cloud, int num_neighbors = 20, int num_threads = 4) {
  estimate_local_features_omp<PointCloud, NormalSetter<PointCloud>>(cloud, num_neighbors, num_threads);
}

template <typename PointCloud>
void estimate_covariances_omp(PointCloud& cloud, int num_neighbors = 20, int num_threads = 4) {
  estimate_local_features_omp<PointCloud, CovarianceSetter<PointCloud>>(cloud, num_neighbors, num_threads);
}

template <typename PointCloud>
void estimate_normals_covariances_omp(PointCloud& cloud, int num_neighbors = 20, int num_threads = 4) {
  estimate_local_features_omp<PointCloud, NormalCovarianceSetter<PointCloud>>(cloud, num_neighbors, num_threads);
}

}  // namespace small_gicp
