// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/ann/gaussian_voxelmap.hpp>

namespace small_gicp {

template <typename PointSource, typename PointTarget>
class RegistrationPCL : public pcl::Registration<PointSource, PointTarget, float> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

  using Ptr = pcl::shared_ptr<RegistrationPCL<PointSource, PointTarget>>;
  using ConstPtr = pcl::shared_ptr<const RegistrationPCL<PointSource, PointTarget>>;

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::reg_name_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;

  using pcl::Registration<PointSource, PointTarget, Scalar>::nr_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::max_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::final_transformation_;

  using pcl::Registration<PointSource, PointTarget, Scalar>::transformation_epsilon_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::converged_;

public:
  RegistrationPCL();
  virtual ~RegistrationPCL();

  void setNumThreads(int n) { num_threads_ = n; }
  void setCorrespondenceRandomness(int k) { k_correspondences_ = k; }
  void setVoxelResolution(double r) { voxel_resolution_ = r; }
  void setRotationEpsilon(double eps) { rotation_epsilon_ = eps; }
  void setRegistrationType(const std::string& type);

  const Eigen::Matrix<double, 6, 6>& getFinalHessian() const { return final_hessian_; }

  void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

  void swapSourceAndTarget();
  void clearSource();
  void clearTarget();

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

protected:
  int num_threads_;
  int k_correspondences_;
  double rotation_epsilon_;
  double voxel_resolution_;
  bool verbose_;
  std::string registration_type_;

  std::shared_ptr<KdTreeOMP<pcl::PointCloud<PointSource>>> target_tree_;
  std::shared_ptr<KdTreeOMP<pcl::PointCloud<PointSource>>> source_tree_;

  std::shared_ptr<GaussianVoxelMap> target_voxelmap_;
  std::shared_ptr<GaussianVoxelMap> source_voxelmap_;

  std::vector<Eigen::Matrix4d> target_covs_;
  std::vector<Eigen::Matrix4d> source_covs_;

  Eigen::Matrix<double, 6, 6> final_hessian_;
};

}  // namespace small_gicp

#include <small_gicp/pcl/pcl_registration_impl.hpp>