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
  py::class_<DistanceRejector>(m, "DistanceRejector", "Correspondence rejection based on the distance between points.")
    .def(py::init<>())
    .def(
      "set_max_distance",
      [](DistanceRejector& rejector, double dist) { rejector.max_dist_sq = dist * dist; },
      py::arg("dist"),
      R"pbdoc(
        Set maximum correspondence distance.

        Parameters
        ----------
        dist : float
            Maximum correspondence distance.
        )pbdoc");

  // ICPFactor
  py::class_<ICPFactor>(m, "ICPFactor", "ICP per-point factor")
    .def(py::init<>())
    .def(
      "linearize",
      [](
        ICPFactor& factor,
        const PointCloud& target,
        const PointCloud& source,
        const KdTree<PointCloud>& kdtree,
        const Eigen::Matrix4d& T,
        size_t source_index,
        const DistanceRejector& rejector) -> std::tuple<bool, Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>, double> {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
        double e = 0.0;
        bool succ = factor.linearize(target, source, kdtree, Eigen::Isometry3d(T), source_index, rejector, &H, &b, &e);
        return std::make_tuple(succ, H, b, e);
      },
      py::arg("target"),
      py::arg("source"),
      py::arg("kdtree"),
      py::arg("T"),
      py::arg("source_index"),
      py::arg("rejector"),
      R"pbdoc(
        Linearize the factor.

        Parameters
        ----------
        target : PointCloud
            Target point cloud.
        source : PointCloud
            Source point cloud.
        kdtree : KdTree
            KdTree for the target point cloud.
        T : numpy.ndarray
            Transformation matrix. (4x4)
        source_index : int
            Index of the source point.
        rejector : DistanceRejector
            Correspondence rejector.

        Returns
        -------
        success: bool
            Success flag.
        H : numpy.ndarray
            Hessian matrix (6x6).
        b : numpy.ndarray
            Gradient vector (6,).
        e : float
            Error.
        )pbdoc");

  // PointToPlaneICPFactor
  py::class_<PointToPlaneICPFactor>(m, "PointToPlaneICPFactor", "Point-to-plane ICP per-point factor")
    .def(py::init<>())
    .def(
      "linearize",
      [](
        PointToPlaneICPFactor& factor,
        const PointCloud& target,
        const PointCloud& source,
        const KdTree<PointCloud>& kdtree,
        const Eigen::Matrix4d& T,
        size_t source_index,
        const DistanceRejector& rejector) -> std::tuple<bool, Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>, double> {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
        double e = 0.0;
        bool succ = factor.linearize(target, source, kdtree, Eigen::Isometry3d(T), source_index, rejector, &H, &b, &e);
        return std::make_tuple(succ, H, b, e);
      },
      py::arg("target"),
      py::arg("source"),
      py::arg("kdtree"),
      py::arg("T"),
      py::arg("source_index"),
      py::arg("rejector"),
      R"pbdoc(
        Linearize the factor.

        Parameters
        ----------
        target : PointCloud
            Target point cloud.
        source : PointCloud
            Source point cloud.
        kdtree : KdTree
            KdTree for the target point cloud.
        T : numpy.ndarray
            Transformation matrix. (4x4)
        source_index : int
            Index of the source point.
        rejector : DistanceRejector
            Correspondence rejector.

        Returns
        -------
        success: bool
            Success flag.
        H : numpy.ndarray
            Hessian matrix (6x6).
        b : numpy.ndarray
            Gradient vector (6,).
        e : float
            Error.
        )pbdoc");

  // GICPFactor
  py::class_<GICPFactor>(m, "GICPFactor", "Generalized ICP per-point factor")  //
    .def(py::init<>())
    .def(
      "linearize",
      [](
        GICPFactor& factor,
        const PointCloud& target,
        const PointCloud& source,
        const KdTree<PointCloud>& kdtree,
        const Eigen::Matrix4d& T,
        size_t source_index,
        const DistanceRejector& rejector) -> std::tuple<bool, Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>, double> {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
        double e = 0.0;
        bool succ = factor.linearize(target, source, kdtree, Eigen::Isometry3d(T), source_index, rejector, &H, &b, &e);
        return std::make_tuple(succ, H, b, e);
      },
      py::arg("target"),
      py::arg("source"),
      py::arg("kdtree"),
      py::arg("T"),
      py::arg("source_index"),
      py::arg("rejector"),
      R"pbdoc(
        Linearize the factor.

        Parameters
        ----------
        target : PointCloud
            Target point cloud.
        source : PointCloud
            Source point cloud.
        kdtree : KdTree
            KdTree for the target point cloud.
        T : numpy.ndarray
            Transformation matrix. (4x4)
        source_index : int
            Index of the source point.
        rejector : DistanceRejector
            Correspondence rejector.

        Returns
        -------
        success: bool
            Success flag.
        H : numpy.ndarray
            Hessian matrix (6x6).
        b : numpy.ndarray
            Gradient vector (6,).
        e : float
            Error.
        )pbdoc");
}