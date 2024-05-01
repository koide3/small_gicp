// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/nanoflann_tbb.hpp>

namespace small_gicp {

/// @brief Unsafe KdTree with multi-thread tree construction with TBB backend.
/// @note  This class only parallelizes the tree construction. The search is still single-threaded as in the normal KdTree.
template <class PointCloud>
using UnsafeKdTreeTBB = UnsafeKdTreeGeneric<PointCloud, nanoflann::KDTreeSingleIndexAdaptorTBB>;

/// @brief KdTree with multi-thread tree construction with TBB backend.
/// @note  This class only parallelizes the tree construction. The search is still single-threaded as in the normal KdTree.
template <class PointCloud>
using KdTreeTBB = KdTreeGeneric<PointCloud, nanoflann::KDTreeSingleIndexAdaptorTBB>;

}  // namespace small_gicp
