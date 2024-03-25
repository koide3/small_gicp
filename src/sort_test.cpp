#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>

#include <tbb/tbb.h>
#include <fmt/format.h>
#include <small_gicp/util/benchmark.hpp>
#include <small_gicp/util/sort_omp.hpp>

int main(int argc, char** argv) {
  std::vector<std::vector<std::pair<std::uint64_t, size_t>>> dataset(1000);
  std::cout << "read dataset" << std::endl;

#pragma omp parallel for
  for (int i = 0; i < dataset.size(); i++) {
    std::ifstream ifs(fmt::format("/tmp/coord_pt_{:06d}.txt", i));
    if (!ifs) {
      std::cerr << "failed to open " << i << std::endl;
      abort();
    }

    std::vector<std::pair<std::uint64_t, size_t>> coord_pt;
    coord_pt.reserve(200000);

    std::string line;
    while (!ifs.eof() && std::getline(ifs, line) && !line.empty()) {
      std::stringstream sst(line);
      std::uint64_t coord;
      size_t pt;
      sst >> coord >> pt;
      coord_pt.emplace_back(coord, pt);
    }

    dataset[i] = std::move(coord_pt);
  }

  using namespace small_gicp;

  Stopwatch sw;
  const int num_threads = 8;
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, num_threads);

  std::vector<std::vector<std::pair<std::uint64_t, size_t>>> omp_sorted_dataset = dataset;
  Summarizer omp_times;

  sw.start();
  for (auto& coord_pt : omp_sorted_dataset) {
    quick_sort_omp(
      coord_pt.begin(),
      coord_pt.end(),
      [](const auto& a, const auto& b) { return a.first < b.first; },
      num_threads);
    sw.lap();
    omp_times.push(sw.msec());
  }

  std::cout << "omp=" << omp_times.str() << std::endl;

  std::vector<std::vector<std::pair<std::uint64_t, size_t>>> std_sorted_dataset = dataset;
  Summarizer std_times;

  sw.start();
  for (auto& coord_pt : std_sorted_dataset) {
    std::ranges::sort(coord_pt, [](const auto& a, const auto& b) { return a.first < b.first; });
    sw.lap();
    std_times.push(sw.msec());
  }

  std::cout << "std=" << std_times.str() << std::endl;

  std::vector<std::vector<std::pair<std::uint64_t, size_t>>> stable_sorted_dataset = dataset;
  Summarizer stable_times;

  sw.start();
  for (auto& coord_pt : std_sorted_dataset) {
    std::ranges::stable_sort(coord_pt, [](const auto& a, const auto& b) { return a.first < b.first; });
    sw.lap();
    stable_times.push(sw.msec());
  }

  std::cout << "stable=" << stable_times.str() << std::endl;

  std::vector<std::vector<std::pair<std::uint64_t, size_t>>> tbb_sorted_dataset = dataset;
  Summarizer tbb_times;

  sw.start();
  for (auto& coord_pt : tbb_sorted_dataset) {
    tbb::parallel_sort(coord_pt, [](const auto& a, const auto& b) { return a.first < b.first; });
    sw.lap();
    tbb_times.push(sw.msec());
  }

  std::cout << "tbb=" << tbb_times.str() << std::endl;

  std::cout << "validate" << std::endl;
  for (size_t i = 0; i < std_sorted_dataset.size(); i++) {
    for (size_t j = 0; j < std_sorted_dataset[i].size(); j++) {
      if (std_sorted_dataset[i][j].first != tbb_sorted_dataset[i][j].first) {
        std::cerr << "error: " << i << " " << j << std::endl;
        abort();
      }
      if (std_sorted_dataset[i][j].first != omp_sorted_dataset[i][j].first) {
        std::cerr << "error: " << i << " " << j << std::endl;
        abort();
      }
    }
  }

  return 0;
}