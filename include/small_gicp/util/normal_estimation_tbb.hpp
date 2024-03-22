#pragma once

#include <tbb/tbb.h>
#include <small_gicp/util/normal_estimation.hpp>

namespace small_gicp {

template <typename PointCloud, typename Setter>
void estimate_local_features_tbb(PointCloud& cloud, int num_neighbors) {
  traits::resize(cloud, traits::size(cloud));

  UnsafeKdTree<PointCloud> kdtree(cloud);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, traits::size(cloud)), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i < range.end(); i++) {
      estimate_local_features<PointCloud, UnsafeKdTree<PointCloud>, Setter>(cloud, kdtree, num_neighbors, i);
    }
  });
}

template <typename PointCloud>
void estimate_normals_tbb(PointCloud& cloud, int num_neighbors = 20) {
  estimate_local_features_tbb<PointCloud, NormalSetter<PointCloud>>(cloud, num_neighbors);
}

template <typename PointCloud>
void estimate_covariances_tbb(PointCloud& cloud, int num_neighbors = 20) {
  estimate_local_features_tbb<PointCloud, CovarianceSetter<PointCloud>>(cloud, num_neighbors);
}

template <typename PointCloud>
void estimate_normals_covariances_tbb(PointCloud& cloud, int num_neighbors = 20) {
  estimate_local_features_tbb<PointCloud, NormalCovarianceSetter<PointCloud>>(cloud, num_neighbors);
}

}  // namespace small_gicp
