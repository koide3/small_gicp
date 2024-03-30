#pragma once

#include <pcl/impl/pcl_base.hpp>
#include <pcl/search/impl/search.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/search/impl/flann_search.hpp>
#include <pcl/registration/impl/registration.hpp>

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

  final_hessian_.setIdentity();
}

template <typename PointSource, typename PointTarget>
RegistrationPCL<PointSource, PointTarget>::~RegistrationPCL() {}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  source_tree_ = std::make_shared<KdTreeOMP<pcl::PointCloud<PointSource>>>(input_, num_threads_);
  source_covs_.clear();
  source_voxelmap_.reset();
}

template <typename PointSource, typename PointTarget>
void RegistrationPCL<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  target_tree_ = std::make_shared<KdTreeOMP<pcl::PointCloud<PointTarget>>>(target_, num_threads_);
  target_covs_.clear();
  target_voxelmap_.reset();
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

  Registration<GICPFactor, ParallelReductionOMP> registration;
  registration.reduction.num_threads = num_threads_;
  registration.rejector.max_dist_sq = corr_dist_threshold_ * corr_dist_threshold_;
  registration.optimizer.verbose = verbose_;
  registration.optimizer.max_iterations = max_iterations_;

  RegistrationResult result(Eigen::Isometry3d::Identity());
  if (registration_type_ == "GICP") {
    result = registration.align(target_proxy, source_proxy, *target_tree_, Eigen::Isometry3d(guess.template cast<double>()));
  } else if (registration_type_ == "VGICP") {
    if (!target_voxelmap_) {
      target_voxelmap_ = std::make_shared<GaussianVoxelMap>(voxel_resolution_);
      target_voxelmap_->insert(target_proxy);
    }
    if (!source_voxelmap_) {
      source_voxelmap_ = std::make_shared<GaussianVoxelMap>(voxel_resolution_);
      source_voxelmap_->insert(source_proxy);
    }

    result = registration.align(*target_voxelmap_, source_proxy, *target_voxelmap_, Eigen::Isometry3d(guess.template cast<double>()));
  } else {
    PCL_ERROR("[RegistrationPCL::computeTransformation] Invalid registration type: %s\n", registration_type_.c_str());
    return;
  }

  final_transformation_ = result.T_target_source.matrix().template cast<float>();
  final_hessian_ = result.H;
  pcl::transformPointCloud(*input_, output, final_transformation_);
}

}  // namespace small_gicp
