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

namespace py = pybind11;
using namespace small_gicp;

void define_kdtree(py::module& m) {
  // KdTree
  py::class_<KdTreeOMP<PointCloud>, std::shared_ptr<KdTreeOMP<PointCloud>>>(m, "KdTree")  //
    .def(py::init<PointCloud::ConstPtr, int>(), py::arg("points"), py::arg("num_threads") = 1)
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
}