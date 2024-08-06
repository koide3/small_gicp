// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <small_gicp/points/eigen.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

namespace py = pybind11;
using namespace small_gicp;

void define_preprocess(py::module& m) {
  // voxelgrid_sampling
  m.def(
    "voxelgrid_sampling",
    [](const PointCloud& points, double resolution, int num_threads) {
      if (num_threads == 1) {
        return voxelgrid_sampling(points, resolution);
      }
      return voxelgrid_sampling_omp(points, resolution, num_threads);
    },
    py::arg("points"),
    py::arg("downsampling_resolution"),
    py::arg("num_threads") = 1,
    R"pbdoc(
        Voxelgrid downsampling.

        Notes
        -----
        When multi-threading is enabled, this function has minor run-by-run non-deterministic behavior due to parallel data collection that results
        in a deviation of the number of points in the downsampling results (up to 10% increase from the single-thread version).

        Discretized voxel coords must be in 21bit range [-1048576, 1048575].
        For example, if the downsampling resolution is 0.01 m, point coordinates must be in [-10485.76, 10485.75] m.
        Points outside the valid range will be ignored.

        Parameters
        ----------
        points : :class:`PointCloud`
            Input point cloud.
        resolution : float
            Voxel size.
        num_threads : int, optional
            Number of threads. (default: 1)

        Returns
        -------
        :class:`PointCloud`
            Downsampled point cloud.
        )pbdoc");

  // voxelgrid_sampling (numpy)
  m.def(
    "voxelgrid_sampling",
    [](const Eigen::MatrixXd& points, double resolution, int num_threads) -> std::shared_ptr<PointCloud> {
      if (points.cols() != 3 && points.cols() != 4) {
        std::cerr << "points must be Nx3 or Nx4" << std::endl;
        return nullptr;
      }

      if (num_threads == 1) {
        return voxelgrid_sampling<Eigen::MatrixXd, PointCloud>(points, resolution);
      } else {
        return voxelgrid_sampling_omp<Eigen::MatrixXd, PointCloud>(points, resolution, num_threads);
      }
    },
    py::arg("points"),
    py::arg("downsampling_resolution"),
    py::arg("num_threads") = 1,
    R"pbdoc(
        Voxelgrid downsampling.

        Notes
        -----
        When multi-threading is enabled, this function has minor run-by-run non-deterministic behavior due to parallel data collection that results
        in a deviation of the number of points in the downsampling results (up to 10% increase from the single-thread version).

        Discretized voxel coords must be in 21bit range [-1048576, 1048575].
        For example, if the downsampling resolution is 0.01 m, point coordinates must be in [-10485.76, 10485.75] m.
        Points outside the valid range will be ignored.

        Parameters
        ----------
        points : [np.float64]
            Input point cloud. Nx3 or Nx4.
        resolution : float
            Voxel size.
        num_threads : int, optional
            Number of threads. (default: 1)

        Returns
        -------
        :class:`PointCloud`
            Downsampled point cloud.
        )pbdoc");

  // estimate_normals
  m.def(
    "estimate_normals",
    [](PointCloud::Ptr points, std::shared_ptr<KdTree<PointCloud>> tree, int num_neighbors, int num_threads) {
      if (tree == nullptr) {
        tree = std::make_shared<KdTree<PointCloud>>(points, KdTreeBuilderOMP(num_threads));
      }

      if (num_threads == 1) {
        estimate_normals(*points, *tree, num_neighbors);
      } else {
        estimate_normals_omp(*points, *tree, num_neighbors, num_threads);
      }
    },
    py::arg("points"),
    py::arg("tree") = nullptr,
    py::arg("num_neighbors") = 20,
    py::arg("num_threads") = 1,
    R"pbdoc(
        Estimate point normals.
        If a sufficient number of neighbor points (5 points) is not found, a zero vector is set to the point.

        Parameters
        ----------
        points : :class:`PointCloud`
            Input point cloud. Normals will be estimated in-place. (in/out)
        tree : :class:`KdTree`, optional
            Nearest neighbor search. If None, create a new KdTree (default: None)
        num_neighbors : int, optional
            Number of neighbors. (default: 20)
        num_threads : int, optional
            Number of threads. (default: 1)
        )pbdoc");

  // estimate_covariances
  m.def(
    "estimate_covariances",
    [](PointCloud::Ptr points, std::shared_ptr<KdTree<PointCloud>> tree, int num_neighbors, int num_threads) {
      if (tree == nullptr) {
        tree = std::make_shared<KdTree<PointCloud>>(points, KdTreeBuilderOMP(num_threads));
      }

      if (num_threads == 1) {
        estimate_covariances(*points, *tree, num_neighbors);
      } else {
        estimate_covariances_omp(*points, *tree, num_neighbors, num_threads);
      }
    },
    py::arg("points"),
    py::arg("tree") = nullptr,
    py::arg("num_neighbors") = 20,
    py::arg("num_threads") = 1,
    R"pbdoc(
        Estimate point covariances.
        If a sufficient number of neighbor points (5 points) is not found, an identity matrix is set to the point.

        Parameters
        ----------
        points : :class:`PointCloud`
            Input point cloud. Covariances will be estimated in-place. (in/out)
        tree : :class:`KdTree`, optional
            Nearest neighbor search. If None, create a new KdTree (default: None)
        num_neighbors : int, optional
            Number of neighbors. (default: 20)
        num_threads : int, optional
            Number of threads. (default: 1)
        )pbdoc");

  // estimate_normals_covariances
  m.def(
    "estimate_normals_covariances",
    [](PointCloud::Ptr points, std::shared_ptr<KdTree<PointCloud>> tree, int num_neighbors, int num_threads) {
      if (tree == nullptr) {
        tree = std::make_shared<KdTree<PointCloud>>(points, KdTreeBuilderOMP(num_threads));
      }

      if (num_threads == 1) {
        estimate_normals_covariances(*points, *tree, num_neighbors);
      } else {
        estimate_normals_covariances_omp(*points, *tree, num_neighbors, num_threads);
      }
    },
    py::arg("points"),
    py::arg("tree") = nullptr,
    py::arg("num_neighbors") = 20,
    py::arg("num_threads") = 1,
    R"pbdoc(
        Estimate point normals and covariances.
        If a sufficient number of neighbor points (5 points) is not found, a zero vector and an identity matrix is set to the point.

        Parameters
        ----------
        points : :class:`PointCloud`
            Input point cloud. Normals and covariances will be estimated in-place. (in/out)
        tree : :class:`KdTree`, optional
            Nearest neighbor search. If None, create a new KdTree (default: None)
        num_neighbors : int, optional
            Number of neighbors. (default: 20)
        num_threads : int, optional
            Number of threads. (default: 1)
        )pbdoc");

  // preprocess_points (numpy)
  m.def(
    "preprocess_points",
    [](const Eigen::MatrixXd& points_numpy, double downsampling_resolution, int num_neighbors, int num_threads) -> std::pair<PointCloud::Ptr, KdTree<PointCloud>::Ptr> {
      if (points_numpy.cols() != 3 && points_numpy.cols() != 4) {
        std::cerr << "points_numpy must be Nx3 or Nx4" << std::endl;
        return {nullptr, nullptr};
      }

      auto points = std::make_shared<PointCloud>();
      points->resize(points_numpy.rows());
      for (size_t i = 0; i < points_numpy.rows(); i++) {
        if (points_numpy.cols() == 3) {
          points->point(i) << points_numpy.row(i).transpose(), 1.0;
        } else {
          points->point(i) << points_numpy.row(i).transpose();
        }
      }

      auto downsampled = voxelgrid_sampling_omp(*points, downsampling_resolution, num_threads);
      auto kdtree = std::make_shared<KdTree<PointCloud>>(downsampled, KdTreeBuilderOMP(num_threads));
      estimate_normals_covariances_omp(*downsampled, *kdtree, num_neighbors, num_threads);
      return {downsampled, kdtree};
    },
    py::arg("points"),
    py::arg("downsampling_resolution") = 0.25,
    py::arg("num_neighbors") = 10,
    py::arg("num_threads") = 1,
    R"pbdoc(
        Preprocess point cloud (Downsampling and normal/covariance estimation).
        See also: :func:`voxelgrid_sampling`, :func:`estimate_normals_covariances`.

        Parameters
        ----------
        points : [np.float64]
            Input point cloud. Nx3 or Nx4.
        downsampling_resolution : float, optional
            Resolution for downsampling the point clouds. Input points out of 21bit range after discretization are ignored (See also: :func:`voxelgrid_sampling`). (default: 0.25)
        num_neighbors : int, optional
            Number of neighbors used for attribute estimation. (default: 10)
        num_threads : int, optional
            Number of threads. (default: 1)
        
        Returns
        -------
        :class:`PointCloud`
            Downsampled point cloud.
        :class:`KdTree`
            KdTree for the downsampled point cloud.
        )pbdoc");

  // preprocess_points
  m.def(
    "preprocess_points",
    [](const PointCloud& points, double downsampling_resolution, int num_neighbors, int num_threads) -> std::pair<PointCloud::Ptr, KdTree<PointCloud>::Ptr> {
      if (points.empty()) {
        std::cerr << "warning: points is empty" << std::endl;
        return {nullptr, nullptr};
      }

      auto downsampled = voxelgrid_sampling_omp(points, downsampling_resolution, num_threads);
      auto kdtree = std::make_shared<KdTree<PointCloud>>(downsampled, KdTreeBuilderOMP(num_threads));
      estimate_normals_covariances_omp(*downsampled, *kdtree, num_neighbors, num_threads);
      return {downsampled, kdtree};
    },
    py::arg("points"),
    py::arg("downsampling_resolution") = 0.25,
    py::arg("num_neighbors") = 10,
    py::arg("num_threads") = 1,
    R"pbdoc(
        Preprocess point cloud (Downsampling and normal/covariance estimation).
        See also: :func:`voxelgrid_sampling`, :func:`estimate_normals_covariances`.

        Parameters
        ----------
        points : :class:`PointCloud`
            Input point cloud.
        downsampling_resolution : float, optional
            Resolution for downsampling the point clouds. Input points out of 21bit range after discretization are ignored (See also: :func:`voxelgrid_sampling`). (default: 0.25)
        num_neighbors : int, optional
            Number of neighbors used for attribute estimation. (default: 10)
        num_threads : int, optional
            Number of threads. (default: 1)
        
        Returns
        -------
        :class:`PointCloud`
            Downsampled point cloud.
        :class:`KdTree`
            KdTree for the downsampled point cloud.
        )pbdoc");
}