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
  py::class_<OptimizationTrialRecord>(m, "OptimizationTrialRecord")
    .def_readonly("trial", &OptimizationTrialRecord::trial)
    .def_readonly("lambda_", &OptimizationTrialRecord::lambda)
    .def_readonly("candidate_error", &OptimizationTrialRecord::candidate_error)
    .def_readonly("comparison_reference_error", &OptimizationTrialRecord::comparison_reference_error)
    .def_readonly("delta", &OptimizationTrialRecord::delta)
    .def_readonly("T_candidate", &OptimizationTrialRecord::T_candidate)
    .def_readonly("accepted", &OptimizationTrialRecord::accepted);

  py::class_<IterationRecord>(m, "IterationRecord")
    .def_readonly("iteration", &IterationRecord::iteration)
    .def_readonly("T_before", &IterationRecord::T_before)
    .def_readonly("T_after", &IterationRecord::T_after)
    .def_readonly("H", &IterationRecord::H)
    .def_readonly("b", &IterationRecord::b)
    .def_readonly("delta", &IterationRecord::delta)
    .def_readonly("linearized_error", &IterationRecord::linearized_error)
    .def_readonly("reported_error", &IterationRecord::reported_error)
    .def_readonly("lambda_before", &IterationRecord::lambda_before)
    .def_readonly("lambda_after", &IterationRecord::lambda_after)
    .def_readonly("num_inliers", &IterationRecord::num_inliers)
    .def_readonly("accepted", &IterationRecord::accepted)
    .def_readonly("trials", &IterationRecord::trials);

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
    .def_property_readonly(
      "T_target_source",
      [](const RegistrationResult& result) -> Eigen::Matrix4d { return result.T_target_source.matrix(); },
      "NDArray[np.float64] : Final transformation matrix (4x4). This transformation brings a point in the source cloud frame to the target cloud frame.")
    .def_readonly("converged", &RegistrationResult::converged, "bool : Convergence flag")
    .def_readonly("iterations", &RegistrationResult::iterations, "int : Number of iterations")
    .def_readonly("num_inliers", &RegistrationResult::num_inliers, "int : Number of inliers")
    .def_readonly("H", &RegistrationResult::H, "NDArray[np.float64] : Final Hessian matrix (6x6)")
    .def_readonly("b", &RegistrationResult::b, "NDArray[np.float64] : Final information vector (6,)")
    .def_readonly("error", &RegistrationResult::error, "float : Final error")
    .def_readonly("history", &RegistrationResult::history, "List[IterationRecord] : optional optimizer history");
}