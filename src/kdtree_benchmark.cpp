#include <thread>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/ann/kdtree_tbb.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_tbb.hpp>
#include <small_gicp/benchmark/benchmark.hpp>

int main(int argc, char** argv) {
  using namespace small_gicp;

  if (argc < 2) {
    std::cout << "USAGE: kdtree_benchmark <dataset_path>" << std::endl;
    std::cout << "OPTIONS:" << std::endl;
    std::cout << "  --num_threads <value> (default: 4)" << std::endl;
    std::cout << "  --downsampling_resolution <value> (default: 0.25)" << std::endl;
    return 0;
  }

  const std::string dataset_path = argv[1];

  int num_threads = 4;
  int num_trials = 100;

  for (int i = 0; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "--num_threads") {
      num_threads = std::stoi(argv[i + 1]);
    } else if (arg == "--num_trials") {
      num_trials = std::stoi(argv[i + 1]);
    } else if (arg.size() >= 2 && arg.substr(0, 2) == "--") {
      std::cerr << "unknown option: " << arg << std::endl;
      return 1;
    }
  }

  std::cout << "dataset_path=" << dataset_path << std::endl;
  std::cout << "num_threads=" << num_threads << std::endl;
  std::cout << "num_trials=" << num_trials << std::endl;

  tbb::global_control tbb_control(tbb::global_control::max_allowed_parallelism, num_threads);

  KittiDataset kitti(dataset_path, 1);
  auto raw_points = std::make_shared<PointCloud>(kitti.points[0]);
  std::cout << "num_raw_points=" << raw_points->size() << std::endl;

  const auto search_voxel_resolution = [&](size_t target_num_points) {
    std::pair<double, size_t> left = {0.1, voxelgrid_sampling_tbb(*raw_points, 0.1)->size()};
    std::pair<double, size_t> right = {1.0, voxelgrid_sampling_tbb(*raw_points, 1.0)->size()};

    for (int i = 0; i < 20; i++) {
      if (left.second < target_num_points) {
        left.first *= 0.1;
        left.second = voxelgrid_sampling_tbb(*raw_points, left.first)->size();
        continue;
      }
      if (right.second > target_num_points) {
        right.first *= 10.0;
        right.second = voxelgrid_sampling_tbb(*raw_points, right.first)->size();
        continue;
      }

      const double mid = (left.first + right.first) * 0.5;
      const size_t mid_num_points = voxelgrid_sampling_tbb(*raw_points, mid)->size();

      if (std::abs(1.0 - static_cast<double>(mid_num_points) / target_num_points) < 0.001) {
        return mid;
      }

      if (mid_num_points > target_num_points) {
        left = {mid, mid_num_points};
      } else {
        right = {mid, mid_num_points};
      }
    }

    return (left.first + right.first) * 0.5;
  };

  std::cout << "---" << std::endl;

  std::vector<double> downsampling_resolutions;
  std::vector<PointCloud::Ptr> downsampled;
  for (double target = 1.0; target > 0.05; target -= 0.1) {
    const double downsampling_resolution = search_voxel_resolution(raw_points->size() * target);
    downsampling_resolutions.emplace_back(downsampling_resolution);
    downsampled.emplace_back(voxelgrid_sampling_tbb(*raw_points, downsampling_resolution));
    std::cout << "downsampling_resolution=" << downsampling_resolution << std::endl;
    std::cout << "num_points=" << downsampled.back()->size() << std::endl;
  }

  std::cout << "---" << std::endl;

  // warmup
  for (int i = 0; i < 10; i++) {
    auto downsampled = voxelgrid_sampling(*raw_points, 0.1);
    UnsafeKdTree<PointCloud> tree(*downsampled);
    UnsafeKdTreeTBB<PointCloud> tree_tbb(*downsampled);
    UnsafeKdTreeOMP<PointCloud> tree_omp(*downsampled, num_threads);
  }

  Stopwatch sw;
  for (size_t i = 0; i < downsampling_resolutions.size(); i++) {
    if (num_threads != 1) {
      break;
    }

    Summarizer kdtree_times;
    sw.start();
    for (size_t j = 0; j < num_trials; j++) {
      UnsafeKdTree<PointCloud> tree(*downsampled[i]);
      sw.lap();
      kdtree_times.push(sw.msec());
    }
    std::cout << "kdtree_times=" << kdtree_times.str() << " [msec]" << std::endl;
  }

  std::cout << "---" << std::endl;

  for (size_t i = 0; i < downsampling_resolutions.size(); i++) {
    Summarizer kdtree_tbb_times;
    sw.start();
    for (size_t j = 0; j < num_trials; j++) {
      UnsafeKdTreeTBB<PointCloud> tree(*downsampled[i]);
      sw.lap();
      kdtree_tbb_times.push(sw.msec());
    }

    std::cout << "kdtree_tbb_times=" << kdtree_tbb_times.str() << " [msec]" << std::endl;
  }

  std::cout << "---" << std::endl;

  for (size_t i = 0; i < downsampling_resolutions.size(); i++) {
    Summarizer kdtree_omp_times;
    sw.start();
    for (size_t j = 0; j < num_trials; j++) {
      UnsafeKdTreeOMP<PointCloud> tree(*downsampled[i], num_threads);
      sw.lap();
      kdtree_omp_times.push(sw.msec());
    }

    std::cout << "kdtree_omp_times=" << kdtree_omp_times.str() << " [msec]" << std::endl;
  }

  return 0;
}
