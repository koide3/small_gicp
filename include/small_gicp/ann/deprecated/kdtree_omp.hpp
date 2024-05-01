// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/nanoflann_omp.hpp>

namespace small_gicp {

/// @brief Unsafe KdTree with multi-thread tree construction with OpenMP backend.
/// @note  This class only parallelizes the tree construction. The search is still single-threaded as in the normal KdTree.
template <class PointCloud>
using UnsafeKdTreeOMP = UnsafeKdTreeGeneric<PointCloud, nanoflann::KDTreeSingleIndexAdaptorOMP>;

/// @brief KdTree with multi-thread tree construction with OpenMP backend.
/// @note  This class only parallelizes the tree construction. The search is still single-threaded as in the normal KdTree.
template <class PointCloud>
using KdTreeOMP = KdTreeGeneric<PointCloud, nanoflann::KDTreeSingleIndexAdaptorOMP>;

}  // namespace small_gicp
