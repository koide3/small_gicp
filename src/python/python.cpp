// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <pybind11/pybind11.h>

namespace py = pybind11;

void define_pointcloud(py::module& m);
void define_kdtree(py::module& m);
void define_voxelmap(py::module& m);
void define_preprocess(py::module& m);
void define_result(py::module& m);
void define_align(py::module& m);
void define_factors(py::module& m);
void define_misc(py::module& m);

PYBIND11_MODULE(small_gicp, m) {
  m.doc() = "Efficient and parallel algorithms for point cloud registration";

  define_pointcloud(m);
  define_kdtree(m);
  define_voxelmap(m);
  define_preprocess(m);
  define_result(m);
  define_align(m);
  define_factors(m);
  define_misc(m);
}