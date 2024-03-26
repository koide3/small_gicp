#include <random>
#include <algorithm>
#include <fmt/format.h>
#include <small_gicp/util/sort_omp.hpp>

#include <gtest/gtest.h>

using namespace small_gicp;

bool identical(const std::vector<int>& arr1, const std::vector<int>& arr2) {
  if (arr1.size() != arr2.size()) {
    return false;
  }

  for (size_t i = 0; i < arr1.size(); i++) {
    if (arr1[i] != arr2[i]) {
      return false;
    }
  }
  return true;
}

TEST(SortOMP, MergeSortTest) {
  std::mt19937 mt;

  std::uniform_int_distribution<> data_dist(-100, 100);
  std::uniform_int_distribution<> size_dist(0, 1024);

  for (int i = 0; i < 100; i++) {
    std::vector<int> data(size_dist(mt));
    std::ranges::generate(data, [&] { return data_dist(mt); });

    std::vector<int> sorted = data;
    std::ranges::sort(sorted);

    std::vector<int> sorted_omp = data;
    merge_sort_omp(sorted_omp.begin(), sorted_omp.end(), std::less<int>(), 4);

    EXPECT_TRUE(identical(sorted, sorted_omp)) << fmt::format("i={} N={}", i, data.size());
  }
}

TEST(SortOMP, QuickSortTest) {
  std::mt19937 mt;

  std::uniform_int_distribution<> data_dist(-100, 100);
  std::uniform_int_distribution<> size_dist(0, 1024);

  for (int i = 0; i < 100; i++) {
    std::vector<int> data(size_dist(mt));
    std::ranges::generate(data, [&] { return data_dist(mt); });

    std::vector<int> sorted = data;
    std::ranges::sort(sorted);

    std::vector<int> sorted_omp = data;
    quick_sort_omp(sorted_omp.begin(), sorted_omp.end(), std::less<int>(), 4);

    EXPECT_TRUE(identical(sorted, sorted_omp)) << fmt::format("i={} N={}", i, data.size());
  }
}
