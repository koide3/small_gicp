#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/kdtree_mt.hpp>
#include <small_gicp/ann/flat_voxelmap.hpp>
#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_tbb.hpp>
#include <small_gicp/util/normal_estimation.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/util/normal_estimation_tbb.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/benchmark.hpp>
#include <small_gicp/registration/optimizer.hpp>

#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/factors/plane_icp_factor.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/reduction_tbb.hpp>
#include <small_gicp/registration/registration.hpp>

#include <guik/viewer/light_viewer.hpp>
#include <guik/viewer/async_light_viewer.hpp>

int main(int argc, char** argv) {
  using namespace small_gicp;

  // if (argc < 3) {
  //   std::cout << "usage: odometry_benchmark <dataset_path> <output_path> (--engine small|fast|pcl) (--num_threads 4) (--resolution 0.25)" << std::endl;
  //   return 0;
  // }

  const std::string dataset_path = "/home/koide/datasets/kitti/velodyne_filtered";
  const std::string output_path = "/tmp/traj_lidar.txt";

  int num_threads = 128;
  double downsampling_resolution = 0.25;

  /*
  for (auto arg = argv; arg != argv + argc; arg++) {
    if (std::string(*arg) == "--num_threads") {
      num_threads = std::stoi(*(arg + 1));
    } else if (std::string(*arg) == "--resolution") {
      downsampling_resolution = std::stod(*(arg + 1));
    } else if (std::string(*arg) == "--engine") {
      engine = *(arg + 1);
    }
  }
  */

  std::cout << "dataset_path=" << dataset_path << std::endl;
  std::cout << "output_path=" << output_path << std::endl;
  std::cout << "num_threads=" << num_threads << std::endl;
  std::cout << "downsampling_resolution=" << downsampling_resolution << std::endl;

  tbb::global_control control(tbb::global_control::max_allowed_parallelism, num_threads);

  KittiDataset kitti(dataset_path);
  std::cout << "num_frames=" << kitti.points.size() << std::endl;
  std::cout << fmt::format("num_points={} [points]", summarize(kitti.points, [](const auto& pts) { return pts.size(); })) << std::endl;

  auto raw_points = kitti.convert<PointCloud>(true);

  auto async_viewer = guik::async_viewer();
  async_viewer->use_orbit_camera_control(250.0);

  struct InputFrame {
    using Ptr = std::shared_ptr<InputFrame>;
    size_t id;
    PointCloud::Ptr points;
    KdTree<PointCloud>::Ptr kdtree;
    Eigen::Isometry3d T_last_current;
  };
  struct InputFramePair {
    InputFrame::Ptr source;
    InputFrame::Ptr target;
  };

  tbb::flow::graph graph;
  tbb::flow::broadcast_node<InputFrame::Ptr> input(graph);
  tbb::flow::function_node<InputFrame::Ptr, InputFrame::Ptr> preprocess_node(graph, tbb::flow::unlimited, [=](const InputFrame::Ptr& input) {
    input->points = voxelgrid_sampling(*input->points, downsampling_resolution);
    input->kdtree = std::make_shared<KdTree<PointCloud>>(input->points);
    estimate_covariances(*input->points, *input->kdtree, 10);
    return input;
  });
  tbb::flow::sequencer_node<InputFrame::Ptr> postpre_sequencer_node(graph, [](const InputFrame::Ptr& input) { return input->id; });

  tbb::flow::function_node<InputFrame::Ptr, InputFramePair> pairing_node(graph, 1, [&](const InputFrame::Ptr& input) {
    static InputFrame::Ptr last_frame;
    InputFramePair pair;
    pair.source = input;
    pair.target = last_frame;
    last_frame = input;
    return pair;
  });

  tbb::flow::function_node<InputFramePair, InputFrame::Ptr> registration_node(graph, tbb::flow::unlimited, [&](const InputFramePair& pair) {
    if (pair.target == nullptr) {
      pair.source->T_last_current.setIdentity();
      return pair.source;
    }

    Registration<GICPFactor, DistanceRejector, SerialReduction, LevenbergMarquardtOptimizer> registration;
    const auto result = registration.align(*pair.target->points, *pair.source->points, *pair.target->kdtree, Eigen::Isometry3d::Identity());
    pair.source->T_last_current = result.T_target_source;
    return pair.source;
  });

  tbb::flow::sequencer_node<InputFrame::Ptr> postreg_sequencer_node(graph, [](const InputFrame::Ptr& input) { return input->id; });

  tbb::flow::function_node<InputFrame::Ptr> viewer_node(graph, 1, [&](const InputFrame::Ptr& input) {
    static size_t counter = 0;
    static Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T = T * input->T_last_current;

    async_viewer->update_points(fmt::format("{}", counter++), input->points->points, guik::Rainbow(T));
    async_viewer->update_points("points", input->points->points, guik::FlatOrange(T).set_point_scale(2.0f));
  });

  tbb::flow::make_edge(input, preprocess_node);
  tbb::flow::make_edge(preprocess_node, postpre_sequencer_node);
  tbb::flow::make_edge(postpre_sequencer_node, pairing_node);
  tbb::flow::make_edge(pairing_node, registration_node);
  tbb::flow::make_edge(registration_node, postreg_sequencer_node);
  tbb::flow::make_edge(postreg_sequencer_node, viewer_node);

  std::cout << "Run" << std::endl;
  Stopwatch sw;
  sw.start();
  for (size_t i = 0; i < raw_points.size(); i++) {
    auto frame = InputFrame::Ptr(new InputFrame);
    frame->id = i;
    frame->points = raw_points[i];
    if (!input.try_put(frame)) {
      std::cerr << "failed to input!!" << std::endl;
    }
  }

  std::cout << "wait_for_all" << std::endl;
  graph.wait_for_all();

  sw.stop();
  std::cout << "Elapsed time: " << sw.msec() << "[ms]  " << sw.msec() / raw_points.size() << "[msec/scan]" << std::endl;

  guik::async_wait();

  return 0;
}
