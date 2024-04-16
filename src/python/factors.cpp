// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/factors/plane_icp_factor.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/registration/rejector.hpp>

namespace py = pybind11;
using namespace small_gicp;

void define_factors(py::module& m) {
  // DistanceRejector
  py::class_<DistanceRejector>(m, "DistanceRejector")  //
    .def(py::init<>())
    .def("set_max_distance", [](DistanceRejector& rejector, double dist) { rejector.max_dist_sq = dist * dist; });

  // ICPFactor
  py::class_<ICPFactor>(m, "ICPFactor")  //
    .def(py::init<>())
    .def(
      "linearize",
      [](
        ICPFactor& factor,
        const PointCloud& target,
        const PointCloud& source,
        const KdTreeOMP<PointCloud>& kdtree,
        const Eigen::Matrix4d& T,
        size_t source_index,
        const DistanceRejector& rejector) -> std::tuple<bool, Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>, double> {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
        double e = 0.0;
        bool succ = factor.linearize(target, source, kdtree, Eigen::Isometry3d(T), source_index, rejector, &H, &b, &e);
        return std::make_tuple(succ, H, b, e);
      });

  // PointToPlaneICPFactor
  py::class_<PointToPlaneICPFactor>(m, "PointToPlaneICPFactor")  //
    .def(py::init<>())
    .def(
      "linearize",
      [](
        PointToPlaneICPFactor& factor,
        const PointCloud& target,
        const PointCloud& source,
        const KdTreeOMP<PointCloud>& kdtree,
        const Eigen::Matrix4d& T,
        size_t source_index,
        const DistanceRejector& rejector) -> std::tuple<bool, Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>, double> {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
        double e = 0.0;
        bool succ = factor.linearize(target, source, kdtree, Eigen::Isometry3d(T), source_index, rejector, &H, &b, &e);
        return std::make_tuple(succ, H, b, e);
      });

  // GICPFactor
  py::class_<GICPFactor>(m, "GICPFactor")  //
    .def(py::init<>())
    .def(
      "linearize",
      [](
        GICPFactor& factor,
        const PointCloud& target,
        const PointCloud& source,
        const KdTreeOMP<PointCloud>& kdtree,
        const Eigen::Matrix4d& T,
        size_t source_index,
        const DistanceRejector& rejector) -> std::tuple<bool, Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>, double> {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
        double e = 0.0;
        bool succ = factor.linearize(target, source, kdtree, Eigen::Isometry3d(T), source_index, rejector, &H, &b, &e);
        return std::make_tuple(succ, H, b, e);
      });
}