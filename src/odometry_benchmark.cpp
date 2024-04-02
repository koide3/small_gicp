#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/benchmark/benchmark.hpp>
#include <small_gicp/benchmark/benchmark_odom.hpp>

int main(int argc, char** argv) {
  using namespace small_gicp;

  if (argc < 3) {
    std::cout << "USAGE: odometry_benchmark <dataset_path> <output_path> [options]" << std::endl;
    std::cout << "OPTIONS:" << std::endl;
    std::cout << "  --visualize" << std::endl;
    std::cout << "  --num_threads <value> (default: 4)" << std::endl;
    std::cout << "  --num_neighbors <value> (default: 20)" << std::endl;
    std::cout << "  --downsampling_resolution <value> (default: 0.25)" << std::endl;
    std::cout << "  --voxel_resolution <value> (default: 2.0)" << std::endl;

    const auto odom_names = odometry_names();
    std::stringstream sst;
    for (size_t i = 0; i < odom_names.size(); i++) {
      if (i) {
        sst << "|";
      }
      sst << odom_names[i];
    }

    std::cout << "  --engine <" << sst.str() << "> (default: small_gicp)" << std::endl;
    return 0;
  }

  const std::string dataset_path = argv[1];
  const std::string output_path = argv[2];

  OdometryEstimationParams params;
  std::string engine = "small_gicp";

  for (int i = 0; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "--visualize") {
      params.visualize = true;
    } else if (arg == "--num_threads") {
      params.num_threads = std::stoi(argv[i + 1]);
    } else if (arg == "--num_neighbors") {
      params.num_neighbors = std::stoi(argv[i + 1]);
    } else if (arg == "--downsampling_resolution") {
      params.downsampling_resolution = std::stod(argv[i + 1]);
    } else if (arg == "--voxel_resolution") {
      params.voxel_resolution = std::stod(argv[i + 1]);
    } else if (arg == "--engine") {
      engine = argv[i + 1];
    } else if (arg.size() >= 2 && arg.substr(0, 2) == "--") {
      std::cerr << "unknown option: " << arg << std::endl;
      return 1;
    }
  }

  std::cout << "SIMD in use=" << Eigen::SimdInstructionSetsInUse() << std::endl;
  std::cout << "dataset_path=" << dataset_path << std::endl;
  std::cout << "output_path=" << output_path << std::endl;
  std::cout << "registration_engine=" << engine << std::endl;
  std::cout << "num_threads=" << params.num_threads << std::endl;
  std::cout << "num_neighbors=" << params.num_neighbors << std::endl;
  std::cout << "downsampling_resolution=" << params.downsampling_resolution << std::endl;
  std::cout << "voxel_resolution=" << params.voxel_resolution << std::endl;
  std::cout << "visualize=" << params.visualize << std::endl;

  std::shared_ptr<OdometryEstimation> odom = create_odometry(engine, params);
  if (odom == nullptr) {
    return 1;
  }

  KittiDataset kitti(dataset_path);
  std::cout << "num_frames=" << kitti.points.size() << std::endl;
  std::cout << fmt::format("num_points={} [points]", summarize(kitti.points, [](const auto& pts) { return pts.size(); })) << std::endl;

  auto raw_points = kitti.convert<PointCloud>(true);
  std::vector<Eigen::Isometry3d> traj = odom->estimate(raw_points);

  std::cout << "done!" << std::endl;
  odom->report();

  std::ofstream ofs(output_path);
  for (const auto& T : traj) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4; j++) {
        if (i || j) {
          ofs << " ";
        }

        ofs << fmt::format("{:.6f}", T(i, j));
      }
    }
    ofs << std::endl;
  }

  return 0;
}
