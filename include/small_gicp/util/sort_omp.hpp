// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>

namespace small_gicp {

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

template <typename RandomAccessIterator, typename Compare>
void merge_sort_omp(RandomAccessIterator first, RandomAccessIterator last, const Compare& comp, int num_threads) {
#pragma omp parallel num_threads(num_threads)
  {
#pragma omp single nowait
    { merge_sort_omp_impl(first, last, comp); }
  }
}

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

template <typename RandomAccessIterator, typename Compare>
void quick_sort_omp(RandomAccessIterator first, RandomAccessIterator last, const Compare& comp, int num_threads) {
#pragma omp parallel num_threads(num_threads)
  {
#pragma omp single nowait
    { quick_sort_omp_impl(first, last, comp); }
  }
}

}  // namespace small_gicp
