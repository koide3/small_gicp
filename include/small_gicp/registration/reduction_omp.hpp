// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>

namespace small_gicp {

#ifndef _OPENMP
#warning "OpenMP is not available. Parallel reduction will be disabled."
inline int omp_get_thread_num() {
  return 0;
}
#endif

/// @brief Parallel reduction with OpenMP backend
struct ParallelReductionOMP {
  ParallelReductionOMP() : num_threads(4) {}

  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree, typename CorrespondenceRejector, typename Factor>
  std::tuple<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>, double> linearize(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const CorrespondenceRejector& rejector,
    const Eigen::Isometry3d& T,
    std::vector<Factor>& factors) const {
    std::vector<Eigen::Matrix<double, 6, 6>> Hs(num_threads, Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> bs(num_threads, Eigen::Matrix<double, 6, 1>::Zero());
    std::vector<double> es(num_threads, 0.0);

#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
    for (std::int64_t i = 0; i < factors.size(); i++) {
      Eigen::Matrix<double, 6, 6> H;
      Eigen::Matrix<double, 6, 1> b;
      double e;

      if (!factors[i].linearize(target, source, target_tree, T, i, rejector, &H, &b, &e)) {
        continue;
      }

      const int thread_id = omp_get_thread_num();
      Hs[thread_id] += H;
      bs[thread_id] += b;
      es[thread_id] += e;
    }

    for (int i = 1; i < num_threads; i++) {
      Hs[0] += Hs[i];
      bs[0] += bs[i];
      es[0] += es[i];
    }

    return {Hs[0], bs[0], es[0]};
  }

  template <typename TargetPointCloud, typename SourcePointCloud, typename Factor>
  double error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T, std::vector<Factor>& factors) const {
    double sum_e = 0.0;

#pragma omp parallel for num_threads(num_threads) schedule(guided, 8) reduction(+ : sum_e)
    for (std::int64_t i = 0; i < factors.size(); i++) {
      sum_e += factors[i].error(target, source, T);
    }
    return sum_e;
  }

  int num_threads;  ///< Number of threads
};

}  // namespace small_gicp
