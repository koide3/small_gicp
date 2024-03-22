#pragma once

#include <small_gicp/ann/traits.hpp>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/registration/rejector.hpp>
#include <small_gicp/registration/reduction.hpp>
#include <small_gicp/registration/registration_result.hpp>
#include <small_gicp/registration/optimizer.hpp>

#include <guik/viewer/light_viewer.hpp>

namespace small_gicp {

struct TerminationCriteria {
  TerminationCriteria() : translation_eps(1e-3), rotation_eps(0.1 * M_PI / 180.0) {}

  bool converged(const Eigen::Matrix<double, 6, 1>& delta) const { return delta.template head<3>().norm() < rotation_eps && delta.template tail<3>().norm() < translation_eps; }

  double translation_eps;
  double rotation_eps;
};

template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree, typename Factor, typename CorrespondenceRejector, typename Reduction, typename Optimizer>
struct Registration {
public:
  RegistrationResult align(const TargetPointCloud& target, const SourcePointCloud& source, const TargetTree& target_tree, const Eigen::Isometry3d& init_T) {
    std::vector<Factor> factors(traits::size(source));
    return optimizer.optimize(target, source, target_tree, rejector, criteria, reduction, init_T, factors);
  }

public:
  TerminationCriteria criteria;
  CorrespondenceRejector rejector;
  Reduction reduction;
  Optimizer optimizer;
};

}  // namespace small_gicp
