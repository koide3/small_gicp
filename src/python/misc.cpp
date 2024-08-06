// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/benchmark/read_points.hpp>

namespace py = pybind11;
using namespace small_gicp;

void define_misc(py::module& m) {
  // read_ply
  m.def(
    "read_ply",
    [](const std::string& filename) {
      const auto points = read_ply(filename);
      return std::make_shared<PointCloud>(points);
    },
    "Read PLY file. This function can only read simple point clouds with XYZ properties for testing purposes. Do not use this for general PLY IO.",
    py::arg("filename"));
}