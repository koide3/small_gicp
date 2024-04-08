#ifdef BUILD_WITH_TBB

#include <small_gicp/benchmark/benchmark_odom.hpp>

#include <tbb/tbb.h>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/util/normal_estimation.hpp>
#include <small_gicp/registration/registration.hpp>

namespace small_gicp {

class SmallGICPFlowEstimationTBB : public OdometryEstimation {
public:
  struct InputFrame {
    using Ptr = std::shared_ptr<InputFrame>;
    size_t id;                         // Frame sequential ID
    PointCloud::Ptr points;            // Input point cloud
    KdTree<PointCloud>::Ptr kdtree;    // KdTree for the input point cloud
    Eigen::Isometry3d T_last_current;  // Transformation from the last frame to the current frame
    Stopwatch sw;                      // Stopwatch for performance measurement
  };

  struct InputFramePair {
    InputFrame::Ptr target;  // Target frame
    InputFrame::Ptr source;  // Source frame
  };

  explicit SmallGICPFlowEstimationTBB(const OdometryEstimationParams& params)
  : OdometryEstimation(params),
    control(tbb::global_control::max_allowed_parallelism, params.num_threads),
    throughput(0.0) {
#ifdef BUILD_WITH_IRIDESCENCE
    if (params.visualize) {
      auto async_viewer = guik::async_viewer();
      async_viewer->use_orbit_camera_control(250.0);
    }
#endif
  }

  ~SmallGICPFlowEstimationTBB() {
#ifdef BUILD_WITH_IRIDESCENCE
    if (params.visualize) {
      guik::async_destroy();
    }
#endif
  }

  std::vector<Eigen::Isometry3d> estimate(std::vector<PointCloud::Ptr>& points) override {
    // Output trajectory
    std::vector<Eigen::Isometry3d> traj;
    traj.reserve(points.size());

    tbb::flow::graph graph;

    // Input node
    tbb::flow::broadcast_node<InputFrame::Ptr> input_node(graph);

    // Preprocessing node
    tbb::flow::function_node<InputFrame::Ptr, InputFrame::Ptr> preprocess_node(graph, tbb::flow::unlimited, [&](const InputFrame::Ptr& input) {
      input->sw.start();
      input->points = voxelgrid_sampling(*input->points, params.downsampling_resolution);  // Downsampling
      input->kdtree = std::make_shared<KdTree<PointCloud>>(input->points);                 // KdTree construction
      estimate_covariances(*input->points, *input->kdtree, params.num_neighbors);          // Covariance estimation
      return input;
    });

    // Make preprocessed frames sequenced
    tbb::flow::sequencer_node<InputFrame::Ptr> postpre_sequencer_node(graph, [](const InputFrame::Ptr& input) { return input->id; });

    // Make pairs of consecutive frames
    tbb::flow::function_node<InputFrame::Ptr, InputFramePair> pairing_node(graph, 1, [&](const InputFrame::Ptr& input) {
      static InputFrame::Ptr last_frame;
      InputFramePair pair = {last_frame, input};
      last_frame = input;
      return pair;
    });

    // Perform registration between the pair of frames
    tbb::flow::function_node<InputFramePair, InputFrame::Ptr> registration_node(graph, tbb::flow::unlimited, [&](const InputFramePair& pair) {
      if (pair.target == nullptr) {
        // This is the very first frame
        pair.source->T_last_current.setIdentity();
        return pair.source;
      }

      // GICP + Single-thread reduction
      Registration<GICPFactor, SerialReduction> registration;
      registration.rejector.max_dist_sq = params.max_correspondence_distance * params.max_correspondence_distance;

      // Align points
      const auto result = registration.align(*pair.target->points, *pair.source->points, *pair.target->kdtree, Eigen::Isometry3d::Identity());
      pair.source->T_last_current = result.T_target_source;
      return pair.source;
    });

    // Make registered frames sequenced
    tbb::flow::sequencer_node<InputFrame::Ptr> postreg_sequencer_node(graph, [](const InputFrame::Ptr& input) { return input->id; });

    // Output node
    tbb::flow::function_node<InputFrame::Ptr> output_node(graph, 1, [&](const InputFrame::Ptr& input) {
      if (traj.empty()) {
        traj.emplace_back(input->T_last_current);
      } else {
        traj.emplace_back(traj.back() * input->T_last_current);
      }
      const Eigen::Isometry3d T = traj.back();

      input->sw.stop();
      reg_times.push(input->sw.msec());

      static auto t0 = input->sw.t1;
      const double elapsed_msec = std::chrono::duration_cast<std::chrono::nanoseconds>(input->sw.t2 - t0).count() / 1e6;
      throughput = elapsed_msec / (input->id + 1);

      if (input->id && input->id % 256 == 0) {
        report();
      }

#ifdef BUILD_WITH_IRIDESCENCE
      if (params.visualize) {
        static Eigen::Vector2f z_range(0.0f, 0.0f);
        z_range[0] = std::min<double>(z_range[0], T.translation().z() - 5.0f);
        z_range[1] = std::max<double>(z_range[1], T.translation().z() + 5.0f);

        auto async_viewer = guik::async_viewer();
        async_viewer->invoke([=] { guik::viewer()->shader_setting().add("z_range", z_range); });
        async_viewer->update_points(guik::anon(), input->points->points, guik::Rainbow(T));
        async_viewer->update_points("points", input->points->points, guik::FlatOrange(T).set_point_scale(2.0f));
      }
#endif
    });

    // Connect nodes
    tbb::flow::make_edge(input_node, preprocess_node);
    tbb::flow::make_edge(preprocess_node, postpre_sequencer_node);
    tbb::flow::make_edge(postpre_sequencer_node, pairing_node);
    tbb::flow::make_edge(pairing_node, registration_node);
    tbb::flow::make_edge(registration_node, postreg_sequencer_node);
    tbb::flow::make_edge(postreg_sequencer_node, output_node);

    Stopwatch sw;
    sw.start();

    // Feed input frames to the graph
    for (size_t i = 0; i < points.size(); i++) {
      auto frame = InputFrame::Ptr(new InputFrame);
      frame->id = i;
      frame->points = points[i];

      // Feed the input frame to the input node
      if (!input_node.try_put(frame)) {
        std::cerr << "failed to input!!" << std::endl;
      }
    }

    // Wait for the graph to finish
    graph.wait_for_all();

    sw.stop();
    throughput = sw.msec() / points.size();

    return traj;
  }

  void report() override {  //
    std::cout << "registration_time_stats=" << reg_times.str() << " [msec/scan]  total_throughput=" << throughput << " [msec/scan]" << std::endl;
  }

private:
  tbb::global_control control;

  double throughput;
  Summarizer reg_times;
};

static auto small_gicp_tbb_flow_registry =
  register_odometry("small_gicp_tbb_flow", [](const OdometryEstimationParams& params) { return std::make_shared<SmallGICPFlowEstimationTBB>(params); });

}  // namespace small_gicp

#endif