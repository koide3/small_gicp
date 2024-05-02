#pragma once

#include <array>
#include <limits>
#include <iostream>
#include <algorithm>

namespace small_gicp {

/// @brief K-nearest neighbor search setting.
struct KnnSetting {
public:
  /// @brief Check if the result satisfies the early termination condition.
  template <typename Result>
  bool fulfilled(const Result& result) const {
    return result.worst_distance() < epsilon;
  }

public:
  double epsilon = 0.0;  ///< Early termination threshold
};

/// @brief K-nearest neighbor search result container.
/// @tparam N   Number of neighbors to search. If N == -1, the number of neighbors is dynamicaly determined.
template <int N>
struct KnnResult {
public:
  static constexpr size_t INVALID = std::numeric_limits<size_t>::max();

  /// @brief Constructor
  /// @param indices        Buffer to store indices (must be larger than k=max(N, num_neighbors))
  /// @param distances      Buffer to store distances (must be larger than k=max(N, num_neighbors))
  /// @param num_neighbors  Number of neighbors to search (must be -1 for static case N > 0)
  explicit KnnResult(size_t* indices, double* distances, int num_neighbors = -1) : num_neighbors(num_neighbors), indices(indices), distances(distances) {
    if constexpr (N > 0) {
      if (num_neighbors >= 0) {
        std::cerr << "warning: Specifying dynamic num_neighbors=" << num_neighbors << " for a static KNN result container (N=" << N << ")" << std::endl;
        abort();
      }
    } else {
      if (num_neighbors <= 0) {
        std::cerr << "error: Specifying invalid num_neighbors=" << num_neighbors << " for a dynamic KNN result container" << std::endl;
        abort();
      }
    }

    std::fill(this->indices, this->indices + buffer_size(), INVALID);
    std::fill(this->distances, this->distances + buffer_size(), std::numeric_limits<double>::max());
  }

  /// @brief  Buffer size (i.e., Maximum number of neighbors)
  size_t buffer_size() const {
    if constexpr (N > 0) {
      return N;
    } else {
      return num_neighbors;
    }
  }

  /// @brief Number of found neighbors.
  size_t num_found() const { return std::distance(indices, std::find(indices, indices + buffer_size(), INVALID)); }

  /// @brief Worst distance in the result.
  double worst_distance() const { return distances[buffer_size() - 1]; }

  /// @brief  Push a pair of point index and distance to the result.
  void push(size_t index, double distance) {
    if (distance >= worst_distance()) {
      return;
    }

    if constexpr (N == 1) {
      indices[0] = index;
      distances[0] = distance;
    } else {
      for (int i = buffer_size() - 1; i >= 0; i--) {
        if (i == 0 || distance >= distances[i - 1]) {
          indices[i] = index;
          distances[i] = distance;
          break;
        }

        indices[i] = indices[i - 1];
        distances[i] = distances[i - 1];
      }
    }
  }

public:
  const int num_neighbors;  ///< Maximum number of neighbors to search
  size_t* indices;          ///< Indices of neighbors
  double* distances;        ///< Distances to neighbors
};

}  // namespace small_gicp
