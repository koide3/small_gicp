// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <algorithm>
#include <functional>
#include <tbb/tbb.h>

namespace small_gicp {

/// @brief Temporal buffers for radix sort.
template <typename T>
struct RadixSortBuffers {
  std::vector<std::uint64_t> tile_buckets;    //< Tiled buckets
  std::vector<std::uint64_t> global_offsets;  //< Global offsets
  std::vector<T> sorted_buffer;               //< Sorted objects
};

/// @brief Radix sort with TBB parallelization.
/// @note  This function outperforms tbb::parallel_sort only in case with many elements and threads.
///        For usual data size and number of threads, use tbb::parallel_sort.
/// @tparam T          Data type (must be unsigned integral type)
/// @tparam KeyFunc    Key function
/// @tparam bits       Number of bits per step
/// @tparam tile_size  Tile size
/// @param first_      [in/out] First iterator
/// @param last_       [in/out] Last iterator
/// @param key_        Key function (T => uint)
/// @param buffers     Temporal buffers
template <typename T, typename KeyFunc, int bits = 8, int tile_size = 256>
void radix_sort_tbb(T* first_, T* last_, const KeyFunc& key_, RadixSortBuffers<T>& buffers) {
  if (first_ == last_) {
    return;
  }

  auto first = first_;
  auto last = last_;

  using Key = decltype(key_(*first));
  static_assert(std::is_unsigned_v<Key>, "Key must be unsigned integral type");

  // Number of total radix sort steps.
  constexpr int num_steps = (sizeof(Key) * 8 + bits - 1) / bits;

  constexpr int num_bins = 1 << bits;
  const std::uint64_t N = std::distance(first, last);
  const std::uint64_t num_tiles = (N + tile_size - 1) / tile_size;

  // Allocate buffers.
  auto& tile_buckets = buffers.tile_buckets;
  auto& global_offsets = buffers.global_offsets;
  auto& sorted_buffer = buffers.sorted_buffer;
  tile_buckets.resize(num_bins * num_tiles);
  global_offsets.resize(num_bins);
  sorted_buffer.resize(N);

  auto sorted = sorted_buffer.data();

  // Radix sort.
  for (int step = 0; step < num_steps; step++) {
    const auto key = [&](const auto& x) { return ((key_(x) >> (step * bits))) & ((1 << bits) - 1); };

    // Create per-tile histograms.
    std::fill(tile_buckets.begin(), tile_buckets.end(), 0);
    tbb::parallel_for(tbb::blocked_range<std::uint64_t>(0, num_tiles, 4), [&](const tbb::blocked_range<std::uint64_t>& r) {
      for (std::uint64_t tile = r.begin(); tile < r.end(); tile++) {
        std::uint64_t data_begin = tile * tile_size;
        std::uint64_t data_end = std::min<std::uint64_t>((tile + 1) * tile_size, N);

        for (int i = data_begin; i < data_end; ++i) {
          auto buckets = tile_buckets.data() + key(*(first + i)) * num_tiles;
          ++buckets[tile];
        }
      }
    });

    // Store the number of elements of the last tile, which will be overwritten by the next step, in global_offsets.
    std::fill(global_offsets.begin(), global_offsets.end(), 0);
    for (int i = 1; i < num_bins; i++) {
      global_offsets[i] = tile_buckets[i * num_tiles - 1];
    }

    // Calculate per-tile offsets.
    tbb::parallel_for(tbb::blocked_range<std::uint64_t>(0, num_bins, 1), [&](const tbb::blocked_range<std::uint64_t>& r) {
      for (std::uint64_t bin = r.begin(); bin < r.end(); bin++) {
        auto buckets = tile_buckets.data() + bin * num_tiles;
        std::uint64_t last = buckets[0];
        buckets[0] = 0;

        for (std::uint64_t tile = 1; tile < num_tiles; tile++) {
          std::uint64_t tmp = buckets[tile];
          buckets[tile] = buckets[tile - 1] + last;
          last = tmp;
        }
      }
    });

    // Calculate global offsets for each sorting bin.
    for (int i = 1; i < num_bins; i++) {
      global_offsets[i] += global_offsets[i - 1] + tile_buckets[i * num_tiles - 1];
    }

    // Sort elements.
    tbb::parallel_for(tbb::blocked_range<std::uint64_t>(0, num_tiles, 8), [&](const tbb::blocked_range<std::uint64_t>& r) {
      for (std::uint64_t tile = r.begin(); tile < r.end(); ++tile) {
        std::uint64_t data_begin = tile * tile_size;
        std::uint64_t data_end = std::min((tile + 1) * tile_size, static_cast<std::uint64_t>(N));

        for (std::uint64_t i = data_begin; i < data_end; ++i) {
          const T x = *(first + i);
          const int bin = key(x);
          auto offset = tile_buckets.data() + bin * num_tiles + tile;
          sorted[global_offsets[bin] + ((*offset)++)] = x;
        }
      }
    });

    // Swap input and output buffers.
    std::swap(first, sorted);
  }

  // Copy the result to the original buffer.
  if (num_steps % 2 == 1) {
    std::copy(sorted_buffer.begin(), sorted_buffer.end(), first_);
  }
}

/// @brief Radix sort with TBB parallelization.
/// @tparam T           Data type (must be unsigned integral type)
/// @tparam KeyFunc     Key function
/// @tparam bits        Number of bits per step
/// @tparam tile_size   Tile size
/// @param first_       [in/out] First iterator
/// @param last_        [in/out] Last iterator
/// @param key_         Key function (T => uint)
template <typename T, typename KeyFunc, int bits = 4, int tile_size = 256>
void radix_sort_tbb(T* first_, T* last_, const KeyFunc& key_) {
  RadixSortBuffers<T> buffers;
  radix_sort_tbb(first_, last_, key_, buffers);
}

}  // namespace small_gicp