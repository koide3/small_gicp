// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <tbb/tbb.h>
#include <Eigen/Core>

namespace small_gicp {

/// @brief Summation for linearized systems
template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree, typename CorrespondenceRejector, typename Factor>
struct LinearizeSum {
  LinearizeSum(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const CorrespondenceRejector& rejector,
    const Eigen::Isometry3d& T,
    std::vector<Factor>& factors)
  : target(target),
    source(source),
    target_tree(target_tree),
    rejector(rejector),
    T(T),
    factors(factors),
    H(Eigen::Matrix<double, 6, 6>::Zero()),
    b(Eigen::Matrix<double, 6, 1>::Zero()),
    e(0.0) {}

  LinearizeSum(LinearizeSum& x, tbb::split)
  : target(x.target),
    source(x.source),
    target_tree(x.target_tree),
    rejector(x.rejector),
    T(x.T),
    factors(x.factors),
    H(Eigen::Matrix<double, 6, 6>::Zero()),
    b(Eigen::Matrix<double, 6, 1>::Zero()),
    e(0.0) {}

  void operator()(const tbb::blocked_range<size_t>& r) {
    Eigen::Matrix<double, 6, 6> Ht = H;
    Eigen::Matrix<double, 6, 1> bt = b;
    double et = e;

    for (size_t i = r.begin(); i != r.end(); i++) {
      Eigen::Matrix<double, 6, 6> Hi;
      Eigen::Matrix<double, 6, 1> bi;
      double ei;

      if (!factors[i].linearize(target, source, target_tree, T, i, rejector, &Hi, &bi, &ei)) {
        continue;
      }

      Ht += Hi;
      bt += bi;
      et += ei;
    }

    H = Ht;
    b = bt;
    e = et;
  }

  void join(const LinearizeSum& y) {
    H += y.H;
    b += y.b;
    e += y.e;
  }

  const TargetPointCloud& target;
  const SourcePointCloud& source;
  const TargetTree& target_tree;
  const CorrespondenceRejector& rejector;
  const Eigen::Isometry3d& T;
  std::vector<Factor>& factors;

  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double e;
};

/// @brief Summation for evaluated errors
template <typename TargetPointCloud, typename SourcePointCloud, typename Factor>
struct ErrorSum {
  ErrorSum(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T, std::vector<Factor>& factors)
  : target(target),
    source(source),
    T(T),
    factors(factors),
    e(0.0) {}

  ErrorSum(ErrorSum& x, tbb::split) : target(x.target), source(x.source), T(x.T), factors(x.factors), e(0.0) {}

  void operator()(const tbb::blocked_range<size_t>& r) {
    double et = e;
    for (size_t i = r.begin(); i != r.end(); i++) {
      et += factors[i].error(target, source, T);
    }
    e = et;
  }

  void join(const ErrorSum& y) { e += y.e; }

  const TargetPointCloud& target;
  const SourcePointCloud& source;
  const Eigen::Isometry3d& T;
  std::vector<Factor>& factors;

  double e;
};

/// @brief Parallel reduction with TBB backend
struct ParallelReductionTBB {
  ParallelReductionTBB() {}

  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree, typename CorrespondenceRejector, typename Factor>
  std::tuple<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>, double> linearize(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const CorrespondenceRejector& rejector,
    const Eigen::Isometry3d& T,
    std::vector<Factor>& factors) const {
    //
    LinearizeSum<TargetPointCloud, SourcePointCloud, TargetTree, CorrespondenceRejector, Factor> sum(target, source, target_tree, rejector, T, factors);

    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, factors.size(), 8), sum);

    return {sum.H, sum.b, sum.e};
  }

  template <typename TargetPointCloud, typename SourcePointCloud, typename Factor>
  double error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T, std::vector<Factor>& factors) const {
    ErrorSum<TargetPointCloud, SourcePointCloud, Factor> sum(target, source, T, factors);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, factors.size(), 16), sum);
    return sum.e;
  }
};

}  // namespace small_gicp
