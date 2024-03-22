#pragma once

#include <Eigen/Core>

namespace small_gicp {

struct SerialReduction {
  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree, typename CorrespondenceRejector, typename Factor>
  std::tuple<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>, double> linearize(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const CorrespondenceRejector& rejector,
    const Eigen::Isometry3d& T,
    std::vector<Factor>& factors) {
    Eigen::Matrix<double, 6, 6> sum_H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> sum_b = Eigen::Matrix<double, 6, 1>::Zero();
    double sum_e = 0.0;

    for (size_t i = 0; i < factors.size(); i++) {
      Eigen::Matrix<double, 6, 6> H;
      Eigen::Matrix<double, 6, 1> b;
      double e;

      if (!factors[i].linearize(target, source, target_tree, T, i, rejector, &H, &b, &e)) {
        continue;
      }

      sum_H += H;
      sum_b += b;
      sum_e += e;
    }

    return {sum_H, sum_b, sum_e};
  }

  template <typename TargetPointCloud, typename SourcePointCloud, typename Factor>
  double error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T, std::vector<Factor>& factors) {
    double sum_e = 0.0;
    for (size_t i = 0; i < factors.size(); i++) {
      sum_e += factors[i].error(target, source, T);
    }
    return sum_e;
  }
};

}  // namespace small_gicp
