#include <fmt/format.h>

#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#ifdef BUILD_WITH_TBB
#include <small_gicp/util/downsampling_tbb.hpp>
#endif
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/benchmark/benchmark.hpp>

#ifdef BUILD_WITH_PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#endif

namespace small_gicp {

template <typename PointCloudPtr, typename Func>
void benchmark(const std::vector<PointCloudPtr>& raw_points, double leaf_size, const Func& downsample) {
  Stopwatch sw;
  Summarizer times;
  Summarizer num_points;

  sw.start();
  for (const auto& points : raw_points) {
    auto downsampled = downsample(points, leaf_size);

    sw.lap();
    times.push(sw.msec());
    num_points.push(downsampled->size());
  }

  std::cout << fmt::format("{} [msec/scan]   {} [points]", times.str(), num_points.str()) << std::endl;
}

#ifdef BUILD_WITH_PCL
template <typename VoxelGrid, typename PointCloudPtr>
auto downsample_pcl(const PointCloudPtr& points, double leaf_size) {
  VoxelGrid voxelgrid;
  voxelgrid.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxelgrid.setInputCloud(points);

  auto downsampled = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  voxelgrid.filter(*downsampled);
  return downsampled;
}
#endif

}  // namespace small_gicp

int main(int argc, char** argv) {
  using namespace small_gicp;

  if (argc < 2) {
    std::cout << "usage: downsampling_benchmark <dataset_path> (--num_threads 4) (--max_num_frames 1000)" << std::endl;
    return 0;
  }

  const std::string dataset_path = argv[1];

  int num_threads = 4;
  size_t max_num_frames = 1000;

  for (int i = 1; i < argc; i++) {
    const std::string arg(argv[i]);
    if (arg == "--num_threads") {
      num_threads = std::stoi(argv[i + 1]);
    } else if (arg == "--max_num_frames") {
      max_num_frames = std::stoul(argv[i + 1]);
    } else if (arg.size() >= 2 && arg.substr(0, 2) == "--") {
      std::cerr << "unknown option: " << arg << std::endl;
      return 1;
    }
  }

  std::cout << "dataset_path=" << dataset_path << std::endl;
  std::cout << "max_num_frames=" << max_num_frames << std::endl;
  std::cout << "num_threads=" << num_threads << std::endl;

#ifdef BUILD_WITH_TBB
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, num_threads);
#endif

  KittiDataset kitti(dataset_path, max_num_frames);
  std::cout << "num_frames=" << kitti.points.size() << std::endl;
  std::cout << fmt::format("num_points={} [points]", summarize(kitti.points, [](const auto& pts) { return pts.size(); })) << std::endl;

#ifdef BUILD_WITH_PCL
  const auto points = kitti.convert<pcl::PointCloud<pcl::PointXYZ>>(true);
#else
  const auto points = kitti.convert<PointCloud>(true);
#endif

  // Warming up
  std::cout << "---" << std::endl;
  std::cout << "leaf_size=0.5(warmup)" << std::endl;
#ifdef BUILD_WITH_PCL
  std::cout << fmt::format("{:25}: ", "pcl_voxelgrid") << std::flush;
  benchmark(points, 0.5, [](const auto& points, double leaf_size) { return downsample_pcl<pcl::VoxelGrid<pcl::PointXYZ>>(points, leaf_size); });
  std::cout << fmt::format("{:25}: ", "pcl_approx_voxelgrid") << std::flush;
  benchmark(points, 0.5, [](const auto& points, double leaf_size) { return downsample_pcl<pcl::ApproximateVoxelGrid<pcl::PointXYZ>>(points, leaf_size); });
#endif
  std::cout << fmt::format("{:25}: ", "small_voxelgrid") << std::flush;
  benchmark(points, 0.5, [](const auto& points, double leaf_size) { return voxelgrid_sampling(*points, leaf_size); });
  std::cout << fmt::format("{:25}: ", "small_voxelgrid_omp") << std::flush;
  benchmark(points, 0.5, [=](const auto& points, double leaf_size) { return voxelgrid_sampling_omp(*points, leaf_size, num_threads); });
#ifdef BUILD_WITH_TBB
  std::cout << fmt::format("{:25}: ", "small_voxelgrid_tbb") << std::flush;
  benchmark(points, 0.5, [](const auto& points, double leaf_size) { return voxelgrid_sampling_tbb(*points, leaf_size); });
#endif

  // Benchmark
  for (double leaf_size = 0.1; leaf_size <= 1.51; leaf_size += 0.1) {
    std::cout << "---" << std::endl;
    std::cout << "leaf_size=" << leaf_size << std::endl;
#ifdef BUILD_WITH_PCL
    std::cout << fmt::format("{:25}: ", "pcl_voxelgrid") << std::flush;
    benchmark(points, leaf_size, [](const auto& points, double leaf_size) { return downsample_pcl<pcl::VoxelGrid<pcl::PointXYZ>>(points, leaf_size); });
    std::cout << fmt::format("{:25}: ", "pcl_approx_voxelgrid") << std::flush;
    benchmark(points, leaf_size, [](const auto& points, double leaf_size) { return downsample_pcl<pcl::ApproximateVoxelGrid<pcl::PointXYZ>>(points, leaf_size); });
#endif
    std::cout << fmt::format("{:25}: ", "small_voxelgrid") << std::flush;
    benchmark(points, leaf_size, [](const auto& points, double leaf_size) { return voxelgrid_sampling(*points, leaf_size); });
    std::cout << fmt::format("{:25}: ", "small_voxelgrid_omp") << std::flush;
    benchmark(points, leaf_size, [=](const auto& points, double leaf_size) { return voxelgrid_sampling_omp(*points, leaf_size, num_threads); });
#ifdef BUILD_WITH_TBB
    std::cout << fmt::format("{:25}: ", "small_voxelgrid_tbb") << std::flush;
    benchmark(points, leaf_size, [](const auto& points, double leaf_size) { return voxelgrid_sampling_tbb(*points, leaf_size); });
#endif
  }

  return 0;
}
