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
#include <small_gicp/ann/gaussian_voxelmap.hpp>

namespace py = pybind11;
using namespace small_gicp;

void define_voxelmap(py::module& m) {
  // GaussianVoxelMap
  py::class_<GaussianVoxelMap>(m, "GaussianVoxelMap")  //
    .def(py::init<double>())
    .def(
      "__repr__",
      [](const GaussianVoxelMap& voxelmap) {
        std::stringstream sst;
        sst << "small_gicp.GaussianVoxelMap (" << 1.0 / voxelmap.inv_leaf_size << " m / " << voxelmap.size() << " voxels)" << std::endl;
        return sst.str();
      })
    .def("__len__", [](const GaussianVoxelMap& voxelmap) { return voxelmap.size(); })
    .def("size", &GaussianVoxelMap::size)
    .def(
      "insert",
      [](GaussianVoxelMap& voxelmap, const PointCloud& points, const Eigen::Matrix4d& T) { voxelmap.insert(points, Eigen::Isometry3d(T)); },
      py::arg("points"),
      py::arg("T") = Eigen::Matrix4d::Identity());
}