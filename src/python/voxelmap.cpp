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

template <typename VoxelMap, bool has_normals, bool has_covs>
auto define_class(py::module& m, const std::string& name) {
  py::class_<VoxelMap> vox(m, name.c_str());
  vox.def(py::init<double>())
    .def(
      "__repr__",
      [=](const VoxelMap& voxelmap) {
        std::stringstream sst;
        sst << "small_gicp." << name << "(" << 1.0 / voxelmap.inv_leaf_size << " m / " << voxelmap.size() << " voxels)" << std::endl;
        return sst.str();
      })
    .def("__len__", [](const VoxelMap& voxelmap) { return voxelmap.size(); })
    .def("size", &VoxelMap::size)
    .def(
      "insert",
      [](VoxelMap& voxelmap, const PointCloud& points, const Eigen::Matrix4d& T) { voxelmap.insert(points, Eigen::Isometry3d(T)); },
      py::arg("points"),
      py::arg("T") = Eigen::Matrix4d::Identity())
    .def(
      "set_lru",
      [](VoxelMap& voxelmap, size_t horizon, size_t clear_cycle) {
        voxelmap.lru_horizon = horizon;
        voxelmap.lru_clear_cycle = clear_cycle;
      },
      py::arg("horizon") = 100,
      py::arg("clear_cycle") = 10)
    .def("voxel_points", [](const VoxelMap& voxelmap) -> Eigen::MatrixXd {
      auto points = traits::voxel_points(voxelmap);
      return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(points[0].data(), points.size(), 4);
    });

  if constexpr (has_normals) {
    vox.def("voxel_normals", [](const VoxelMap& voxelmap) -> Eigen::MatrixXd {
      auto normals = traits::voxel_normals(voxelmap);
      return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(normals[0].data(), normals.size(), 4);
    });
  }

  if constexpr (has_covs) {
    vox.def("voxel_covs", [](const VoxelMap& voxelmap) -> Eigen::MatrixXd {
      auto covs = traits::voxel_covs(voxelmap);
      return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(covs[0].data(), covs.size(), 16);
    });
  }
};

void define_voxelmap(py::module& m) {
  define_class<IncrementalVoxelMap<FlatContainerPoints>, false, false>(m, "IncrementalVoxelMap");
  define_class<IncrementalVoxelMap<FlatContainerNormal>, true, false>(m, "IncrementalVoxelMapNormal");
  define_class<IncrementalVoxelMap<FlatContainerCov>, false, true>(m, "IncrementalVoxelMapCov");
  define_class<IncrementalVoxelMap<FlatContainerNormalCov>, true, true>(m, "FlatContainerNormalCov");
  define_class<GaussianVoxelMap, false, true>(m, "GaussianVoxelMap");
}