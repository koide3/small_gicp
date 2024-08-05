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
  vox
    .def(
      py::init<double>(),
      py::arg("leaf_size"),
      R"pbdoc(
        Construct a Incremental voxelmap.

        Notes
        -----
        This class supports incremental point cloud insertion and LRU-based voxel deletion that removes voxels that are not recently referenced.
        It can handle arbitrary number of voxels and arbitrary range of voxel coordinates (in 32-bit int range).

        Parameters
        ----------
        leaf_size : float
            Voxel size.
        )pbdoc")
    .def(
      "__repr__",
      [=](const VoxelMap& voxelmap) {
        std::stringstream sst;
        sst << "small_gicp." << name << "(" << 1.0 / voxelmap.inv_leaf_size << " m / " << voxelmap.size() << " voxels)" << std::endl;
        return sst.str();
      })
    .def("__len__", [](const VoxelMap& voxelmap) { return voxelmap.size(); })
    .def(
      "size",
      &VoxelMap::size,
      R"pbdoc(
        Get the number of voxels.

        Returns
        -------
        num_voxels : int
            Number of voxels.
        )pbdoc")
    .def(
      "insert",
      [](VoxelMap& voxelmap, const PointCloud& points, const Eigen::Matrix4d& T) { voxelmap.insert(points, Eigen::Isometry3d(T)); },
      py::arg("points"),
      py::arg("T") = Eigen::Matrix4d::Identity(),
      R"pbdoc(
        Insert a point cloud into the voxel map and delete voxels that are not recently accessed.

        Note
        ----
        If this class is based on FlatContainer (i.e., IncrementalVoxelMap*), input points are ignored if
        1) there are too many points in the cell or
        2) the input point is too close to existing points in the cell.

        Parameters
        ----------
        points : :class:`PointCloud`
            Input source point cloud.
        T : numpy.ndarray, optional
            Transformation matrix to be applied to the input point cloud (i.e., T_voxelmap_source). (default: identity)
        )pbdoc")
    .def(
      "set_lru",
      [](VoxelMap& voxelmap, size_t horizon, size_t clear_cycle) {
        voxelmap.lru_horizon = horizon;
        voxelmap.lru_clear_cycle = clear_cycle;
      },
      py::arg("horizon") = 100,
      py::arg("clear_cycle") = 10,
      R"pbdoc(
        Set the LRU cache parameters.

        Parameters
        ----------
        horizon : int, optional
            LRU horizon size. Voxels that have not been accessed for lru_horizon steps are deleted. (default: 100)
        clear_cycle : int, optional
            LRU clear cycle. Voxel deletion is performed every lru_clear_cycle steps. (default: 10)
        )pbdoc")
    .def(
      "voxel_points",
      [](const VoxelMap& voxelmap) -> Eigen::MatrixXd {
        auto points = traits::voxel_points(voxelmap);
        return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(points[0].data(), points.size(), 4);
      },
      R"pbdoc(
        Get the voxel points.

        Returns
        -------
        voxel_points : numpy.ndarray
            Voxel points. (Nx4)
        )pbdoc");

  if constexpr (has_normals) {
    vox.def(
      "voxel_normals",
      [](const VoxelMap& voxelmap) -> Eigen::MatrixXd {
        auto normals = traits::voxel_normals(voxelmap);
        return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(normals[0].data(), normals.size(), 4);
      },
      R"pbdoc(
        Get the voxel normals.

        Returns
        -------
        voxel_normals : numpy.ndarray
            Voxel normals. (Nx4)
        )pbdoc");
  }

  if constexpr (has_covs) {
    vox.def(
      "voxel_covs",
      [](const VoxelMap& voxelmap) -> Eigen::MatrixXd {
        auto covs = traits::voxel_covs(voxelmap);
        return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(covs[0].data(), covs.size(), 16);
      },
      R"pbdoc(
        Get the voxel normals.

        Returns
        -------
        voxel_covs : list of numpy.ndarray
            Voxel covariance matrices. (Nx4x4)
        )pbdoc");
  }
};

void define_voxelmap(py::module& m) {
  define_class<IncrementalVoxelMap<FlatContainerPoints>, false, false>(m, "IncrementalVoxelMap");
  define_class<IncrementalVoxelMap<FlatContainerNormal>, true, false>(m, "IncrementalVoxelMapNormal");
  define_class<IncrementalVoxelMap<FlatContainerCov>, false, true>(m, "IncrementalVoxelMapCov");
  define_class<IncrementalVoxelMap<FlatContainerNormalCov>, true, true>(m, "IncrementalVoxelMapNormalCov");
  define_class<GaussianVoxelMap, false, true>(m, "GaussianVoxelMap");
}