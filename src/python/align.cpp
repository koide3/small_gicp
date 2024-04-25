// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>

#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/registration_helper.hpp>

namespace py = pybind11;
using namespace small_gicp;

void define_align(py::module& m) {
  // align
  m.def(
    "align",
    [](
      const Eigen::MatrixXd& target_points,
      const Eigen::MatrixXd& source_points,
      const Eigen::Matrix4d& init_T_target_source,
      const std::string& registration_type,
      double voxel_resolution,
      double downsampling_resolution,
      double max_corresponding_distance,
      int num_threads) {
      if (target_points.cols() != 3 && target_points.cols() != 4) {
        std::cerr << "target_points must be Nx3 or Nx4" << std::endl;
        return RegistrationResult(Eigen::Isometry3d::Identity());
      }
      if (source_points.cols() != 3 && source_points.cols() != 4) {
        std::cerr << "source_points must be Nx3 or Nx4" << std::endl;
        return RegistrationResult(Eigen::Isometry3d::Identity());
      }

      RegistrationSetting setting;
      if (registration_type == "ICP") {
        setting.type = RegistrationSetting::ICP;
      } else if (registration_type == "PLANE_ICP") {
        setting.type = RegistrationSetting::PLANE_ICP;
      } else if (registration_type == "GICP") {
        setting.type = RegistrationSetting::GICP;
      } else if (registration_type == "VGICP") {
        setting.type = RegistrationSetting::VGICP;
      } else {
        std::cerr << "invalid registration type" << std::endl;
        return RegistrationResult(Eigen::Isometry3d::Identity());
      }

      setting.voxel_resolution = voxel_resolution;
      setting.downsampling_resolution = downsampling_resolution;
      setting.max_correspondence_distance = max_corresponding_distance;
      setting.num_threads = num_threads;

      std::vector<Eigen::Vector4d> target(target_points.rows());
      if (target_points.cols() == 3) {
        for (size_t i = 0; i < target_points.rows(); i++) {
          target[i] << target_points.row(i).transpose(), 1.0;
        }
      } else {
        for (size_t i = 0; i < target_points.rows(); i++) {
          target[i] << target_points.row(i).transpose();
        }
      }

      std::vector<Eigen::Vector4d> source(source_points.rows());
      if (source_points.cols() == 3) {
        for (size_t i = 0; i < source_points.rows(); i++) {
          source[i] << source_points.row(i).transpose(), 1.0;
        }
      } else {
        for (size_t i = 0; i < source_points.rows(); i++) {
          source[i] << source_points.row(i).transpose();
        }
      }

      return align(target, source, Eigen::Isometry3d(init_T_target_source), setting);
    },
    py::arg("target_points"),
    py::arg("source_points"),
    py::arg("init_T_target_source") = Eigen::Matrix4d::Identity(),
    py::arg("registration_type") = "GICP",
    py::arg("voxel_resolution") = 1.0,
    py::arg("downsampling_resolution") = 0.25,
    py::arg("max_corresponding_distance") = 1.0,
    py::arg("num_threads") = 1);

  // align
  m.def(
    "align",
    [](
      const PointCloud::ConstPtr& target,
      const PointCloud::ConstPtr& source,
      KdTreeOMP<PointCloud>::ConstPtr target_tree,
      const Eigen::Matrix4d& init_T_target_source,
      double max_correspondence_distance,
      int num_threads,
      int max_iterations) {
      Registration<GICPFactor, ParallelReductionOMP> registration;
      registration.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;
      registration.reduction.num_threads = num_threads;
      registration.optimizer.max_iterations = max_iterations;

      if (target_tree == nullptr) {
        target_tree = std::make_shared<KdTreeOMP<PointCloud>>(target, num_threads);
      }
      return registration.align(*target, *source, *target_tree, Eigen::Isometry3d(init_T_target_source));
    },
    py::arg("target"),
    py::arg("source"),
    py::arg("target_tree") = nullptr,
    py::arg("init_T_target_source") = Eigen::Matrix4d::Identity(),
    py::arg("max_correspondence_distance") = 1.0,
    py::arg("num_threads") = 1,
    py::arg("max_iterations") = 20);

  // align
  m.def(
    "align",
    [](
      const GaussianVoxelMap& target_voxelmap,
      const PointCloud& source,
      const Eigen::Matrix4d& init_T_target_source,
      double max_correspondence_distance,
      int num_threads,
      int max_iterations) {
      Registration<GICPFactor, ParallelReductionOMP> registration;
      registration.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;
      registration.reduction.num_threads = num_threads;
      registration.optimizer.max_iterations = max_iterations;

      return registration.align(target_voxelmap, source, target_voxelmap, Eigen::Isometry3d(init_T_target_source));
    },
    py::arg("target_voxelmap"),
    py::arg("source"),
    py::arg("init_T_target_source") = Eigen::Matrix4d::Identity(),
    py::arg("max_correspondence_distance") = 1.0,
    py::arg("num_threads") = 1,
    py::arg("max_iterations") = 20);
}