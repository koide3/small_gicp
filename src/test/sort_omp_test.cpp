#include <random>
#include <algorithm>
#include <fmt/format.h>
#include <small_gicp/util/sort_omp.hpp>

#include <gtest/gtest.h>

using namespace small_gicp;

// Check if two vectors are identical
template <typename T>
bool identical(const std::vector<T>& arr1, const std::vector<T>& arr2) {
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

// Test merge_sort_omp
TEST(SortOMP, MergeSortTest) {
  std::mt19937 mt;

  std::uniform_int_distribution<> size_dist(0, 8192);

  // int
  for (int i = 0; i < 100; i++) {
    std::uniform_int_distribution<> data_dist(-100, 100);
    std::vector<int> data(size_dist(mt));
    std::generate(data.begin(), data.end(), [&] { return data_dist(mt); });

    std::vector<int> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    std::vector<int> sorted_omp = data;
    merge_sort_omp(sorted_omp.begin(), sorted_omp.end(), std::less<int>(), 4);

    EXPECT_TRUE(identical(sorted, sorted_omp)) << fmt::format("i={} N={}", i, data.size());
  }

  // double
  for (int i = 0; i < 100; i++) {
    std::uniform_real_distribution<> data_dist(-100, 100);
    std::vector<double> data(size_dist(mt));
    std::generate(data.begin(), data.end(), [&] { return data_dist(mt); });

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    std::vector<double> sorted_omp = data;
    merge_sort_omp(sorted_omp.begin(), sorted_omp.end(), std::less<double>(), 4);

    EXPECT_TRUE(identical(sorted, sorted_omp)) << fmt::format("i={} N={}", i, data.size());
  }

  // empty
  std::vector<int> empty_vector;
  merge_sort_omp(empty_vector.begin(), empty_vector.end(), std::less<int>(), 4);
  EXPECT_TRUE(empty_vector.empty()) << "Empty vector check";
}

// Test quick_sort_omp
TEST(SortOMP, QuickSortTest) {
  std::mt19937 mt;

  std::uniform_int_distribution<> size_dist(0, 8192);

  // int
  for (int i = 0; i < 100; i++) {
    std::uniform_int_distribution<> data_dist(-100, 100);
    std::vector<int> data(size_dist(mt));
    std::generate(data.begin(), data.end(), [&] { return data_dist(mt); });

    std::vector<int> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    std::vector<int> sorted_omp = data;
    quick_sort_omp(sorted_omp.begin(), sorted_omp.end(), std::less<int>(), 4);

    EXPECT_TRUE(identical(sorted, sorted_omp)) << fmt::format("i={} N={}", i, data.size());
  }

  // double
  for (int i = 0; i < 100; i++) {
    std::uniform_real_distribution<> data_dist(-100, 100);
    std::vector<double> data(size_dist(mt));
    std::generate(data.begin(), data.end(), [&] { return data_dist(mt); });

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    std::vector<double> sorted_omp = data;
    quick_sort_omp(sorted_omp.begin(), sorted_omp.end(), std::less<double>(), 4);

    EXPECT_TRUE(identical(sorted, sorted_omp)) << fmt::format("i={} N={}", i, data.size());
  }

  // empty
  std::vector<int> empty_vector;
  quick_sort_omp(empty_vector.begin(), empty_vector.end(), std::less<int>(), 4);
  EXPECT_TRUE(empty_vector.empty()) << "Empty vector check";
}
