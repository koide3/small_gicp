// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>

#ifdef _MSC_VER
#pragma message("warning: Task-based OpenMP parallelism is not well supported on windows.")
#pragma message("warning: Thus, OpenMP-based downsampling is only partially parallelized on windows.")
#endif

namespace small_gicp {

/// @brief Implementation of merge sort with OpenMP parallelism. Do not call this directly. Use merge_sort_omp instead.
/// @param first  First iterator
/// @param last   Last iterator
/// @param comp   Comparison function
template <typename RandomAccessIterator, typename Compare>
void merge_sort_omp_impl(RandomAccessIterator first, RandomAccessIterator last, const Compare& comp) {
  const size_t n = std::distance(first, last);
  if (n < 1024) {
    std::sort(first, last, comp);
    return;
  }

  auto center = first + n / 2;

#pragma omp task
  merge_sort_omp_impl(first, center, comp);

#pragma omp task
  merge_sort_omp_impl(center, last, comp);

#pragma omp taskwait
  std::inplace_merge(first, center, last, comp);
}

/// @brief Merge sort with OpenMP parallelism.
/// @note  This tends to be slower than quick_sort_omp.
/// @param first        First iterator
/// @param last         Last iterator
/// @param comp         Comparison function
/// @param num_threads  Number of threads
template <typename RandomAccessIterator, typename Compare>
void merge_sort_omp(RandomAccessIterator first, RandomAccessIterator last, const Compare& comp, int num_threads) {
#ifndef _MSC_VER
#pragma omp parallel num_threads(num_threads)
  {
#pragma omp single nowait
    { merge_sort_omp_impl(first, last, comp); }
  }
#else
  std::stable_sort(first, last, comp);
#endif
}

/// @brief Implementation of quick sort with OpenMP parallelism. Do not call this directly. Use quick_sort_omp instead.
/// @param first  First iterator
/// @param last   Last iterator
/// @param comp   Comparison function
template <typename RandomAccessIterator, typename Compare>
void quick_sort_omp_impl(RandomAccessIterator first, RandomAccessIterator last, const Compare& comp) {
  const std::ptrdiff_t n = std::distance(first, last);
  if (n < 1024) {
    std::sort(first, last, comp);
    return;
  }

  const auto median3 = [&](const auto& a, const auto& b, const auto& c, const Compare& comp) {
    return comp(a, b) ? (comp(b, c) ? b : (comp(a, c) ? c : a)) : (comp(a, c) ? a : (comp(b, c) ? c : b));
  };

  const int offset = n / 8;
  const auto m1 = median3(*first, *(first + offset), *(first + offset * 2), comp);
  const auto m2 = median3(*(first + offset * 3), *(first + offset * 4), *(first + offset * 5), comp);
  const auto m3 = median3(*(first + offset * 6), *(first + offset * 7), *(last - 1), comp);

  auto pivot = median3(m1, m2, m3, comp);
  auto middle1 = std::partition(first, last, [&](const auto& val) { return comp(val, pivot); });
  auto middle2 = std::partition(middle1, last, [&](const auto& val) { return !comp(pivot, val); });

#pragma omp task
  quick_sort_omp_impl(first, middle1, comp);

#pragma omp task
  quick_sort_omp_impl(middle2, last, comp);
}

/// @brief Quick sort with OpenMP parallelism.
/// @param first        First iterator
/// @param last         Last iterator
/// @param comp         Comparison function
/// @param num_threads  Number of threads
template <typename RandomAccessIterator, typename Compare>
void quick_sort_omp(RandomAccessIterator first, RandomAccessIterator last, const Compare& comp, int num_threads) {
#ifndef _MSC_VER
#pragma omp parallel num_threads(num_threads)
  {
#pragma omp single nowait
    { quick_sort_omp_impl(first, last, comp); }
  }
#else
  std::sort(first, last, comp);
#endif
}

}  // namespace small_gicp
