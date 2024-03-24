#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <small_gicp/util/benchmark.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_tbb.hpp>
#include <small_gicp/util/normal_estimation_tbb.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/points/pcl_point.hpp>
#include <small_gicp/points/pcl_point_traits.hpp>

#include <guik/viewer/light_viewer.hpp>

int main(int argc, char** argv) {
  using namespace small_gicp;

  std::cout << "SIMD=" << Eigen::SimdInstructionSetsInUse() << std::endl;

  const std::string dataset_path = "/home/koide/datasets/velodyne";

  std::cout << "Load dataset from " << dataset_path << std::endl;

  Stopwatch sw;
  sw.start();
  KittiDataset kitti(dataset_path, 1000);
  sw.stop();
  std::cout << "load=" << sw.elapsed_sec() << "s" << std::endl;

  size_t sum_num_points = 0;

  const auto points_pcl = kitti.convert<pcl::PointCloud<pcl::PointXYZ>>();

  sw.start();
  sum_num_points = 0;
  for (size_t i = 0; i < points_pcl.size(); i++) {
    auto downsampled = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(0.2, 0.2, 0.2);
    voxelgrid.setInputCloud(points_pcl[i]);

    voxelgrid.filter(*downsampled);
    sum_num_points += downsampled->size();
  }
  sw.stop();
  std::cout << "filter=" << sw.elapsed_sec() << "s avg_num_points=" << static_cast<double>(sum_num_points) / points_pcl.size() << std::endl;

  const auto points = kitti.convert<PointCloud>();

  sw.start();
  sum_num_points = 0;
  for (size_t i = 0; i < points.size(); i++) {
    auto downsampled = voxelgrid_sampling_tbb(*points[i], 0.2);
    sum_num_points += downsampled->size();
  }
  sw.stop();
  std::cout << "filter=" << sw.elapsed_sec() << "s avg_num_points=" << static_cast<double>(sum_num_points) / points.size() << std::endl;

  return 0;
}
