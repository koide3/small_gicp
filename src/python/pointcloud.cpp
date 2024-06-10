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
    .def(py::init([](const Eigen::MatrixXd& points) {
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
    }))  //
    .def("__repr__", [](const PointCloud& points) { return "small_gicp.PointCloud (" + std::to_string(points.size()) + " points)"; })
    .def("__len__", [](const PointCloud& points) { return points.size(); })
    .def("empty", &PointCloud::empty, "Check if the point cloud is empty.")
    .def("size", &PointCloud::size, "Get the number of points.")
    .def(
      "points",
      [](const PointCloud& points) -> Eigen::MatrixXd { return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(points.points[0].data(), points.size(), 4); },
      "Get the points as a Nx4 matrix.")
    .def(
      "normals",
      [](const PointCloud& points) -> Eigen::MatrixXd { return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(points.normals[0].data(), points.size(), 4); },
      "Get the normals as a Nx4 matrix.")
    .def(
      "covs",
      [](const PointCloud& points) { return points.covs; },
      "Get the covariance matrices.")
    .def(
      "point",
      [](const PointCloud& points, size_t i) -> Eigen::Vector4d { return points.point(i); },
      py::arg("i"),
      "Get the i-th point.")
    .def(
      "normal",
      [](const PointCloud& points, size_t i) -> Eigen::Vector4d { return points.normal(i); },
      py::arg("i"),
      "Get the i-th normal.")
    .def("cov", [](const PointCloud& points, size_t i) -> Eigen::Matrix4d { return points.cov(i); }, py::arg("i"), "Get the i-th covariance matrix.");
}