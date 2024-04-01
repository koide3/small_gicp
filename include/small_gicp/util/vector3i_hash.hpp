// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>

namespace small_gicp {

/**
 * @brief Spatial hashing function.
 *        Teschner et al., "Optimized Spatial Hashing for Collision Detection of Deformable Objects", VMV2003.
 */
struct XORVector3iHash {
public:
  size_t operator()(const Eigen::Vector3i& x) const {
    const size_t p1 = 73856093;
    const size_t p2 = 19349669;  // 19349663 was not a prime number
    const size_t p3 = 83492791;
    return static_cast<size_t>((x[0] * p1) ^ (x[1] * p2) ^ (x[2] * p3));
  }

  static size_t hash(const Eigen::Vector3i& x) { return XORVector3iHash()(x); }
  static bool equal(const Eigen::Vector3i& x1, const Eigen::Vector3i& x2) { return x1 == x2; }
};

}  // namespace small_gicp
