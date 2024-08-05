// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <small_gicp/points/point_cloud.hpp>

namespace py = pybind11;
using namespace small_gicp;

void define_pointcloud(py::module& m) {
  // PointCloud
  py::class_<PointCloud, std::shared_ptr<PointCloud>>(m, "PointCloud")  //
    .def(
      py::init([](const Eigen::MatrixXd& points) {
        if (points.cols() != 3 && points.cols() != 4) {
          std::cerr << "points must be Nx3 or Nx4" << std::endl;
          return std::make_shared<PointCloud>();
        }

        auto pc = std::make_shared<PointCloud>();
        pc->resize(points.rows());
        if (points.cols() == 3) {
          for (size_t i = 0; i < points.rows(); i++) {
            pc->point(i) << points.row(i).transpose(), 1.0;
          }
        } else {
          for (size_t i = 0; i < points.rows(); i++) {
            pc->point(i) << points.row(i).transpose();
          }
        }

        return pc;
      }),
      py::arg("points"),
      R"""(
      PointCloud(points: numpy.ndarray)

      Construct a PointCloud from a numpy array.

      Parameters
      ----------
      points : numpy.ndarray, shape (n, 3) or (n, 4)
          The input point cloud.
      )""")
    .def("__repr__", [](const PointCloud& points) { return "small_gicp.PointCloud (" + std::to_string(points.size()) + " points)"; })
    .def("__len__", [](const PointCloud& points) { return points.size(); })
    .def(
      "empty",
      &PointCloud::empty,
      R"pbdoc(
        Check if the point cloud is empty

        Returns
        -------
        empty : bool
            True if the point cloud is empty.
        )pbdoc")
    .def(
      "size",
      &PointCloud::size,
      R"pbdoc(
        Get the number of points.

        Returns
        -------
        num_points : int
            Number of points.
        )pbdoc")
    .def(
      "points",
      [](const PointCloud& points) -> Eigen::MatrixXd { return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(points.points[0].data(), points.size(), 4); },
      R"pbdoc(
        Get the points as a Nx4 matrix.

        Returns
        -------
        points : numpy.ndarray
            Points.
        )pbdoc")
    .def(
      "normals",
      [](const PointCloud& points) -> Eigen::MatrixXd { return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(points.normals[0].data(), points.size(), 4); },
      R"pbdoc(
        Get the normals as a Nx4 matrix.

        Returns
        -------
        normals : numpy.ndarray
            Normals.
        )pbdoc")
    .def(
      "covs",
      [](const PointCloud& points) { return points.covs; },
      R"pbdoc(
        Get the covariance matrices as a list of 4x4 matrices.

        Returns
        -------
        covs : list of numpy.ndarray
            Covariance matrices.
        )pbdoc")
    .def(
      "point",
      [](const PointCloud& points, size_t i) -> Eigen::Vector4d { return points.point(i); },
      py::arg("i"),
      R"pbdoc(
        Get the i-th point.

        Parameters
        ----------
        i : int
            Index of the point.

        Returns
        -------
        point : numpy.ndarray, shape (4,)
            Point.
        )pbdoc")
    .def(
      "normal",
      [](const PointCloud& points, size_t i) -> Eigen::Vector4d { return points.normal(i); },
      py::arg("i"),
      R"pbdoc(
        Get the i-th normal.

        Parameters
        ----------
        i : int
            Index of the point.

        Returns
        -------
        normal : numpy.ndarray, shape (4,)
            Normal.
        )pbdoc")
    .def(
      "cov",
      [](const PointCloud& points, size_t i) -> Eigen::Matrix4d { return points.cov(i); },
      py::arg("i"),
      R"pbdoc(
        Get the i-th covariance matrix.

        Parameters
        ----------
        i : int
            Index of the point.

        Returns
        -------
        cov : numpy.ndarray, shape (4, 4)
            Covariance matrix.
        )pbdoc");
}