// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/benchmark/read_points.hpp>

namespace py = pybind11;
using namespace small_gicp;

PYBIND11_MODULE(small_gicp, m) {
  m.doc() = "Small GICP";

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
    .def("size", &PointCloud::size)
    .def(
      "points",
      [](const PointCloud& points) -> Eigen::MatrixXd { return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(points.points[0].data(), points.size(), 4); })
    .def(
      "normals",
      [](const PointCloud& points) -> Eigen::MatrixXd { return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(points.normals[0].data(), points.size(), 4); })
    .def("covs", [](const PointCloud& points) { return points.covs; })
    .def("point", [](const PointCloud& points, size_t i) -> Eigen::Vector4d { return points.point(i); })
    .def("normal", [](const PointCloud& points, size_t i) -> Eigen::Vector4d { return points.normal(i); })
    .def("cov", [](const PointCloud& points, size_t i) -> Eigen::Matrix4d { return points.cov(i); });

  // KdTree
  py::class_<KdTreeOMP<PointCloud>, std::shared_ptr<KdTreeOMP<PointCloud>>>(m, "KdTree")  //
    .def(
      py::init([](const PointCloud::ConstPtr& points, int num_threads) { return std::make_shared<KdTreeOMP<PointCloud>>(points, num_threads); }),
      py::arg("points"),
      py::arg("num_threads") = 1)
    .def(
      "nearest_neighbor_search",
      [](const KdTreeOMP<PointCloud>& kdtree, const Eigen::Vector3d& pt) {
        size_t k_index = -1;
        double k_sq_dist = std::numeric_limits<double>::max();
        const size_t found = traits::nearest_neighbor_search(kdtree, Eigen::Vector4d(pt.x(), pt.y(), pt.z(), 1.0), &k_index, &k_sq_dist);
        return std::make_tuple(found, k_index, k_sq_dist);
      })
    .def("knn_search", [](const KdTreeOMP<PointCloud>& kdtree, const Eigen::Vector3d& pt, int k) {
      std::vector<size_t> k_indices(k, -1);
      std::vector<double> k_sq_dists(k, std::numeric_limits<double>::max());
      const size_t found = traits::knn_search(kdtree, Eigen::Vector4d(pt.x(), pt.y(), pt.z(), 1.0), k, k_indices.data(), k_sq_dists.data());
      return std::make_pair(k_indices, k_sq_dists);
    });

  // GaussianVoxelMap
  py::class_<GaussianVoxelMap, std::shared_ptr<GaussianVoxelMap>>(m, "GaussianVoxelMap")  //
    .def(py::init([](double voxel_resolution) { return std::make_shared<GaussianVoxelMap>(voxel_resolution); }), py::arg("voxel_resolution"))
    .def(
      "insert",
      [](GaussianVoxelMap& voxelmap, const PointCloud& points, const Eigen::Matrix4d& T) { voxelmap.insert(points, Eigen::Isometry3d(T)); },
      py::arg("points"),
      py::arg("T") = Eigen::Matrix4d::Identity());

  // RegistrationResult
  py::class_<RegistrationResult>(m, "RegistrationResult")  //
    .def(
      "__repr__",
      [](const RegistrationResult& result) {
        std::stringstream sst;
        sst << "small_gicp.RegistrationResult\n";
        sst << "converted=" << result.converged << "\n";
        sst << "iterations=" << result.iterations << "\n";
        sst << "num_inliers=" << result.num_inliers << "\n";
        sst << "T_target_source=\n" << result.T_target_source.matrix() << "\n";
        sst << "H=\n" << result.H << "\n";
        sst << "b=\n" << result.b.transpose() << "\n";
        sst << "error= " << result.error << "\n";
        return sst.str();
      })
    .def_property_readonly("T_target_source", [](const RegistrationResult& result) -> Eigen::Matrix4d { return result.T_target_source.matrix(); })
    .def_readonly("converged", &RegistrationResult::converged)
    .def_readonly("iterations", &RegistrationResult::iterations)
    .def_readonly("num_inliers", &RegistrationResult::num_inliers)
    .def_readonly("H", &RegistrationResult::H)
    .def_readonly("b", &RegistrationResult::b)
    .def_readonly("error", &RegistrationResult::error);

  // read_ply
  m.def(
    "read_ply",
    [](const std::string& filename) {
      const auto points = read_ply(filename);
      return std::make_shared<PointCloud>(points);
    },
    "Read PLY file",
    py::arg("filename"));

  // voxelgrid_sampling
  m.def(
    "voxelgrid_sampling",
    [](const PointCloud& points, double resolution, int num_threads) {
      if (num_threads == 1) {
        return voxelgrid_sampling(points, resolution);
      }
      return voxelgrid_sampling_omp(points, resolution, num_threads);
    },
    "Voxelgrid sampling",
    py::arg("points"),
    py::arg("downsampling_resolution"),
    py::arg("num_threads") = 1);

  // estimate_normals
  m.def(
    "estimate_normals",
    [](PointCloud::Ptr points, std::shared_ptr<KdTreeOMP<PointCloud>> tree, int num_neighbors, int num_threads) {
      if (tree == nullptr) {
        tree = std::make_shared<KdTreeOMP<PointCloud>>(points, num_threads);
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
    py::arg("num_threads") = 1);

  // estimate_covariances
  m.def(
    "estimate_covariances",
    [](PointCloud::Ptr points, std::shared_ptr<KdTreeOMP<PointCloud>> tree, int num_neighbors, int num_threads) {
      if (tree == nullptr) {
        tree = std::make_shared<KdTreeOMP<PointCloud>>(points, num_threads);
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
    py::arg("num_threads") = 1);

  // estimate_normals_covariances
  m.def(
    "estimate_normals_covariances",
    [](PointCloud::Ptr points, std::shared_ptr<KdTreeOMP<PointCloud>> tree, int num_neighbors, int num_threads) {
      if (tree == nullptr) {
        tree = std::make_shared<KdTreeOMP<PointCloud>>(points, num_threads);
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
    py::arg("num_threads") = 1);

  // preprocess_points
  m.def(
    "preprocess_points",
    [](PointCloud::ConstPtr points, const Eigen::MatrixXd points_numpy, double downsampling_resolution, int num_neighbors, int num_threads)
      -> std::pair<PointCloud::Ptr, KdTreeOMP<PointCloud>::Ptr> {
      if (points == nullptr && points_numpy.size() == 0) {
        std::cerr << "points or points_numpy must be provided" << std::endl;
        return {nullptr, nullptr};
      }

      if (!points) {
        if (points_numpy.cols() != 3 && points_numpy.cols() != 4) {
          std::cerr << "points_numpy must be Nx3 or Nx4" << std::endl;
          return {nullptr, nullptr};
        }

        auto pts = std::make_shared<PointCloud>();
        pts->resize(points_numpy.rows());
        for (size_t i = 0; i < points_numpy.rows(); i++) {
          if (points_numpy.cols() == 3) {
            pts->point(i) << points_numpy.row(i).transpose(), 1.0;
          } else {
            pts->point(i) << points_numpy.row(i).transpose();
          }
        }
        points = pts;
      }

      auto downsampled = voxelgrid_sampling_omp(*points, downsampling_resolution, num_threads);
      auto kdtree = std::make_shared<KdTreeOMP<PointCloud>>(downsampled, num_threads);
      estimate_normals_covariances_omp(*downsampled, *kdtree, num_neighbors, num_threads);
      return {downsampled, kdtree};
    },
    py::arg("points") = nullptr,
    py::arg("points_numpy") = Eigen::MatrixXd(),
    py::arg("downsampling_resolution") = 0.25,
    py::arg("num_neighbors") = 10,
    py::arg("num_threads") = 1);

  // align_points
  m.def(
    "align_points",
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
      PointCloud::ConstPtr target,
      PointCloud::ConstPtr source,
      KdTreeOMP<PointCloud>::ConstPtr target_tree,
      const Eigen::Matrix4d& init_T_target_source,
      GaussianVoxelMap::ConstPtr target_voxelmap,
      double max_correspondence_distance,
      int num_threads) {
      if (target == nullptr && target_voxelmap == nullptr) {
        std::cerr << "target or target_voxelmap must be provided" << std::endl;
        return RegistrationResult(Eigen::Isometry3d::Identity());
      }

      Registration<GICPFactor, ParallelReductionOMP> registration;
      registration.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;
      registration.reduction.num_threads = num_threads;

      if (target) {
        if (target_tree == nullptr) {
          target_tree = std::make_shared<KdTreeOMP<PointCloud>>(target, num_threads);
        }
        auto result = registration.align(*target, *source, *target_tree, Eigen::Isometry3d(init_T_target_source));
        return result;
      } else {
        return registration.align(*target_voxelmap, *source, *target_voxelmap, Eigen::Isometry3d(init_T_target_source));
      }
    },
    py::arg("target") = nullptr,
    py::arg("source") = nullptr,
    py::arg("target_tree") = nullptr,
    py::arg("init_T_target_source") = Eigen::Matrix4d::Identity(),
    py::arg("target_voxelmap") = nullptr,
    py::arg("max_correspondence_distance") = 1.0,
    py::arg("num_threads") = 1);
}