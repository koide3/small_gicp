#pragma once

#include <array>
#include <limits>
#include <iostream>
#include <algorithm>

namespace small_gicp {

struct KnnSetting {
public:
  template <typename Result>
  bool fulfilled(const Result& result) const {
    return result.worst_distance() < epsilon;
  }

public:
  double epsilon = 0.0;
};

/// @brief K-nearest neighbor search result container.
/// @tparam N   Number of neighbors to search. If N == -1, the number of neighbors is dynamicaly determined.
template <int N>
struct KnnResult {
public:
  static constexpr size_t INVALID = std::numeric_limits<size_t>::max();
  using IndexContainer = std::conditional_t<N != -1, std::array<size_t, std::max(N, 1)>, std::vector<size_t>>;
  using DistanceContainer = std::conditional_t<N != -1, std::array<double, std::max(N, 1)>, std::vector<double>>;

  /// @brief  Constructor
  /// @param num_neighbors   Number of neighbors to search. Must be 0 if N != -1.
  /// @param max_distance    Maximum distance to search.
  explicit KnnResult(const int num_neighbors = -1, double max_distance = std::numeric_limits<double>::max()) {
    if constexpr (N > 0) {
      if (num_neighbors >= 0) {
        std::cerr << "warning: Specifying dynamic num_neighbors=" << num_neighbors << " for a static KNN result container (N=" << N << ")" << std::endl;
        abort();
      }
      indices.fill(INVALID);
      distances.fill(max_distance);
    } else {
      if (num_neighbors <= 0) {
        std::cerr << "error: Specifying invalid num_neighbors=" << num_neighbors << " for a dynamic KNN result container" << std::endl;
        abort();
      }

      indices.resize(num_neighbors, INVALID);
      distances.resize(num_neighbors, max_distance);
    }
  }

  /// @brief Number of found neighbors.
  size_t num_found() const { return std::distance(indices.begin(), std::find(indices.begin(), indices.end(), INVALID)); }

  /// @brief Worst distance in the result.
  double worst_distance() const { return distances.back(); }

  /// @brief  Push a pair of point index and distance to the result.
  void push(size_t index, double distance) {
    if (distance >= distances.back()) {
      return;
    }

    if constexpr (N == 1) {
      indices[0] = index;
      distances[0] = distance;
    } else {
      for (int i = distances.size() - 1; i >= 0; i--) {
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
  IndexContainer indices;       ///< Indices of neighbors
  DistanceContainer distances;  ///< Distances to neighbors
};

}  // namespace small_gicp
