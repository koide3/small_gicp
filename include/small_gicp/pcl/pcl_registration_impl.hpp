// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <pcl/impl/pcl_base.hpp>
#include <pcl/search/impl/search.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/search/impl/flann_search.hpp>

#include <small_gicp/pcl/pcl_registration.hpp>

#include <small_gicp/pcl/pcl_proxy.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>

namespace small_gicp {

template <typename PointSource, typename PointTarget>
RegistrationPCL<PointSource, PointTarget>::RegistrationPCL() {
  reg_name_ = "RegistrationPCL";

  num_threads_ = 4;
  k_correspondences_ = 20;
  corr_dist_threshold_ = 1000.0;
  rotation_epsilon_ = 2e-3;
  transformation_epsilon_ = 5e-4;

  voxel_resolution_ = 1.0;
  verbose_ = false;
  registration_type_ = "GICP";
}

template <typename PointSource, typename PointTarget>
RegistrationPCL<PointSource, PointTarget>::~RegistrationPCL() {}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  source_tree_ = std::make_shared<small_gicp::KdTree<pcl::PointCloud<PointSource>>>(input_, KdTreeBuilderOMP(num_threads_));
  source_covs_.clear();
  source_voxelmap_.reset();
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  target_tree_ = std::make_shared<small_gicp::KdTree<pcl::PointCloud<PointTarget>>>(target_, KdTreeBuilderOMP(num_threads_));
  target_covs_.clear();
  target_voxelmap_.reset();
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::setSourceCovariances(const std::vector<Eigen::Matrix4d>& covs) {
  if (input_ == nullptr) {
    PCL_ERROR("[RegistrationPCL::setSourceCovariances] Target cloud is not set\n");
    return;
  }

  if (covs.size() != input_->size()) {
    PCL_ERROR("[RegistrationPCL::setSourceCovariances] Invalid number of covariances: %lu\n", covs.size());
    return;
  }

  source_covs_ = covs;
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::setTargetCovariances(const std::vector<Eigen::Matrix4d>& covs) {
  if (target_ == nullptr) {
    PCL_ERROR("[RegistrationPCL::setTargetCovariances] Target cloud is not set\n");
    return;
  }

  if (covs.size() != target_->size()) {
    PCL_ERROR("[RegistrationPCL::setTargetCovariances] Invalid number of covariances: %lu\n", covs.size());
    return;
  }

  target_covs_ = covs;
}

template <typename PointSource, typename PointTarget>
const std::vector<Eigen::Matrix4d>& RegistrationPCL<PointSource, PointTarget>::getSourceCovariances() const {
  if (source_covs_.empty()) {
    PCL_WARN("[RegistrationPCL::getSourceCovariances] Covariances are not estimated\n");
  }

  return source_covs_;
}

template <typename PointSource, typename PointTarget>
const std::vector<Eigen::Matrix4d>& RegistrationPCL<PointSource, PointTarget>::getTargetCovariances() const {
  if (target_covs_.empty()) {
    PCL_WARN("[RegistrationPCL::getTargetCovariances] Covariances are not estimated\n");
  }

  return target_covs_;
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  source_tree_.swap(target_tree_);
  source_covs_.swap(target_covs_);
  source_voxelmap_.swap(target_voxelmap_);
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::clearSource() {
  input_.reset();
  source_tree_.reset();
  source_covs_.clear();
  source_voxelmap_.reset();
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::clearTarget() {
  target_.reset();
  target_tree_.reset();
  target_covs_.clear();
  target_voxelmap_.reset();
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::setNumThreads(int n) {
  if (n <= 0) {
    PCL_ERROR("[RegistrationPCL::setNumThreads] Invalid number of threads: %d\n", n);
    n = 1;
  }

  num_threads_ = n;
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::setCorrespondenceRandomness(int k) {
  setNumNeighborsForCovariance(k);
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::setNumNeighborsForCovariance(int k) {
  if (k < 5) {
    PCL_ERROR("[RegistrationPCL::setNumNeighborsForCovariance] Invalid number of neighbors: %d\n", k);
    k = 5;
  }
  k_correspondences_ = k;
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::setVoxelResolution(double r) {
  if (voxel_resolution_ <= 0) {
    PCL_ERROR("[RegistrationPCL::setVoxelResolution] Invalid voxel resolution: %f\n", r);
    r = 1.0;
  }

  voxel_resolution_ = r;
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::setRotationEpsilon(double eps) {
  rotation_epsilon_ = eps;
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::setRegistrationType(const std::string& type) {
  if (type == "GICP") {
    registration_type_ = type;
  } else if (type == "VGICP") {
    registration_type_ = type;
  } else {
    PCL_ERROR("[RegistrationPCL::setRegistrationType] Invalid registration type: %s\n", type.c_str());
  }
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::setVerbosity(bool verbose) {
  verbose_ = verbose;
}

template <typename PointSource, typename PointTarget>
const Eigen::Matrix<double, 6, 6>& RegistrationPCL<PointSource, PointTarget>::getFinalHessian() const {
  return result_.H;
}

template <typename PointSource, typename PointTarget>
const RegistrationResult& RegistrationPCL<PointSource, PointTarget>::getRegistrationResult() const {
  return result_;
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  if (output.points.data() == input_->points.data() || output.points.data() == target_->points.data()) {
    throw std::invalid_argument("FastGICP: destination cloud cannot be identical to source or target");
  }

  PointCloudProxy<PointSource> source_proxy(*input_, source_covs_);
  PointCloudProxy<PointTarget> target_proxy(*target_, target_covs_);

  if (source_covs_.size() != input_->size()) {
    estimate_covariances_omp(source_proxy, *source_tree_, k_correspondences_, num_threads_);
  }
  if (target_covs_.size() != target_->size()) {
    estimate_covariances_omp(target_proxy, *target_tree_, k_correspondences_, num_threads_);
  }

  small_gicp::Registration<GICPFactor, ParallelReductionOMP> registration;
  registration.criteria.rotation_eps = rotation_epsilon_;
  registration.criteria.translation_eps = transformation_epsilon_;
  registration.reduction.num_threads = num_threads_;
  registration.rejector.max_dist_sq = corr_dist_threshold_ * corr_dist_threshold_;
  registration.optimizer.verbose = verbose_;
  registration.optimizer.max_iterations = max_iterations_;

  if (registration_type_ == "GICP") {
    result_ = registration.align(target_proxy, source_proxy, *target_tree_, Eigen::Isometry3d(guess.template cast<double>()));
  } else if (registration_type_ == "VGICP") {
    if (!target_voxelmap_) {
      target_voxelmap_ = std::make_shared<GaussianVoxelMap>(voxel_resolution_);
      target_voxelmap_->insert(target_proxy);
    }
    if (!source_voxelmap_) {
      source_voxelmap_ = std::make_shared<GaussianVoxelMap>(voxel_resolution_);
      source_voxelmap_->insert(source_proxy);
    }

    result_ = registration.align(*target_voxelmap_, source_proxy, *target_voxelmap_, Eigen::Isometry3d(guess.template cast<double>()));
  } else {
    PCL_ERROR("[RegistrationPCL::computeTransformation] Invalid registration type: %s\n", registration_type_.c_str());
    return;
  }

  converged_ = result_.converged;
  final_transformation_ = result_.T_target_source.matrix().template cast<float>();
  pcl::transformPointCloud(*input_, output, final_transformation_);
}

}  // namespace small_gicp
