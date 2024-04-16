// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <small_gicp/registration/registration_result.hpp>

namespace py = pybind11;
using namespace small_gicp;

void define_result(py::module& m) {
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
}