#pragma once

#include <random>
#include <numeric>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/ann/knn_result.hpp>

namespace small_gicp {

struct OctreeNode {
  using NodeIndexType = std::uint32_t;
  static constexpr NodeIndexType INVALID_NODE = std::numeric_limits<NodeIndexType>::max();
  static constexpr NodeIndexType LEAF_NODE = INVALID_NODE - 1;

  union {
    struct Leaf {
      NodeIndexType is_leaf;
      NodeIndexType first;
      NodeIndexType last;
    } lr;
    struct NonLeaf {
      std::array<NodeIndexType, 8> children;
      std::array<double, 4> separator;
    } sub;
  } node_type;
};

template <typename PointCloud>
struct Octree {
public:
  Octree(const PointCloud& points) : points(points) {
    Eigen::Vector4d min_pt = traits::point(points, 0);
    Eigen::Vector4d max_pt = traits::point(points, 0);
    for (size_t i = 1; i < points.size(); ++i) {
      min_pt = min_pt.cwiseMin(traits::point(points, i));
      max_pt = max_pt.cwiseMax(traits::point(points, i));
    }

    origin = min_pt;
    extent = max_pt - min_pt;

    indices.resize(points.size());
    std::iota(indices.begin(), indices.end(), 0);

    size_t node_count = 0;
    nodes.resize(points.size() * 2);
    root = create_node(origin, extent, node_count, indices.data(), indices.data(), indices.data() + points.size());

    nodes.resize(node_count);
  }

  size_t knn_search(const Eigen::Vector4d& query, int k, size_t* k_indices, double* k_sq_dists, const KnnSetting& setting = KnnSetting()) const {
    KnnResult<-1> result(k_indices, k_sq_dists, k);
    knn_search(query, root, result, setting);
    return result.num_found();
  }

public:
  int calc_child_index(const Eigen::Vector4d& separator, const Eigen::Vector4d& query) const {
    const Eigen::Matrix<bool, 4, 1> comp = query.array() > separator.array();
    return comp[0] + comp[1] * 2 + comp[2] * 4;
  }

  Eigen::Vector4d find_best_separator(const Eigen::Vector4d& origin, const Eigen::Vector4d& extent, size_t* first, size_t* last) {
    const auto evaluate = [&](const Eigen::Vector4d& separator, std::array<size_t, 8>& counts) {
      std::fill(counts.begin(), counts.end(), 0);
      for (size_t* it = first; it != last; ++it) {
        const int index = calc_child_index(separator, traits::point(points, *it));
        counts[index]++;
      }

      double entropy = 0.0;
      for (size_t count : counts) {
        if (count != 0) {
          const double p = static_cast<double>(count) / std::distance(first, last);
          entropy -= p * std::log(p);
        }
      }

      return entropy;
    };

    Eigen::Vector4d baseline_separator = origin + extent * 0.5;
    std::array<size_t, 8> baseline_counts;
    double baseline_entropy = evaluate(baseline_separator, baseline_counts);

    Eigen::Vector4d best_separator = baseline_separator;
    std::array<size_t, 8> best_counts = baseline_counts;
    double best_entropy = baseline_entropy;

    std::mt19937 mt(*first + (origin.array() * Eigen::Array4d(10231, 12321, 8412541, 0.0)).sum());
    std::uniform_real_distribution<> udist(0.0, 1.0);

    for (int i = 0; i < 32; i++) {
      Eigen::Vector4d separator = origin + extent.cwiseProduct(Eigen::Vector4d(udist(mt), udist(mt), udist(mt), 0.0));
      std::array<size_t, 8> counts;
      double entropy = evaluate(separator, counts);

      if (entropy > best_entropy) {
        best_separator = separator;
        best_counts = counts;
        best_entropy = entropy;
      }
    }

    return best_separator;
  }

  OctreeNode::NodeIndexType create_node(const Eigen::Vector4d& origin, const Eigen::Vector4d& extent, size_t& node_count, size_t* global_first, size_t* first, size_t* last) {
    const size_t node_id = node_count++;
    OctreeNode& node = nodes[node_id];

    const size_t N = std::distance(first, last);
    if (N < 20) {
      node.node_type.lr.is_leaf = OctreeNode::LEAF_NODE;
      node.node_type.lr.first = std::distance(global_first, first);
      node.node_type.lr.last = std::distance(global_first, last);
      return node_id;
    }

    // const Eigen::Vector4d separator = origin + extent * 0.5;

    const Eigen::Vector4d separator = find_best_separator(origin, extent, first, last);
    Eigen::Map<Eigen::Vector4d>(node.node_type.sub.separator.data()) = separator;
    const Eigen::Vector4d separator_offset = separator - origin;
    const Eigen::Vector4d separator_rest = extent - separator_offset;

    std::vector<std::pair<size_t, int>> indices(N);
    std::transform(first, last, indices.begin(), [&](size_t i) { return std::make_pair(i, calc_child_index(separator, traits::point(points, i))); });

    std::sort(indices.begin(), indices.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    std::transform(indices.begin(), indices.end(), first, [](const auto& p) { return p.first; });

    auto left = indices.begin();
    for (int i = 0; i < 8; i++) {
      const Eigen::Array4d mask = Eigen::Array4d((i & 1), (i & 2) >> 1, (i & 4) >> 2, 0);

      const Eigen::Vector4d offset = separator_offset.array() * mask;
      const Eigen::Vector4d child_origin = origin + offset;
      const Eigen::Vector4d child_extent = separator_offset.array() * (1 - mask) + separator_rest.array() * mask;

      auto right = std::lower_bound(left, indices.end(), i + 1, [](const auto& p, int i) { return p.second < i; });

      auto left_index = first + std::distance(indices.begin(), left);
      auto right_index = first + std::distance(indices.begin(), right);

      if (std::distance(left, right) == 0) {
        node.node_type.sub.children[i] = OctreeNode::INVALID_NODE;
      } else {
        node.node_type.sub.children[i] = create_node(child_origin, child_extent, node_count, global_first, left_index, right_index);
      }

      left = right;
    }

    return node_id;
  }

  /// @brief Find k-nearest neighbors.
  template <typename Result>
  bool knn_search(const Eigen::Vector4d& query, OctreeNode::NodeIndexType node_index, Result& result, const KnnSetting& setting) const {
    const auto& node = nodes[node_index];

    if (node.node_type.lr.is_leaf == OctreeNode::LEAF_NODE) {
      for (size_t i = node.node_type.lr.first; i < node.node_type.lr.last; ++i) {
        const double sq_dist = (traits::point(points, indices[i]) - query).squaredNorm();
        result.push(indices[i], sq_dist);
      }
      return !setting.fulfilled(result);
    }

    const Eigen::Vector4d separator = Eigen::Map<const Eigen::Vector4d>(node.node_type.sub.separator.data());
    const Eigen::Vector4d diff = separator - query;
    const int child_index = calc_child_index(separator, query);

    int num_valid = 0;
    std::array<std::pair<double, int>, 8> lower_bounds;

    for (int i = 0; i < 8; i++) {
      if (node.node_type.sub.children[i] == OctreeNode::INVALID_NODE) {
        continue;
      }

      const int bitmask = i ^ child_index;
      const double lower_bound = (diff.array() * Eigen::Array4d(bitmask & 1, (bitmask & 2) >> 1, (bitmask & 4) >> 2, 0)).matrix().squaredNorm();
      lower_bounds[num_valid++] = std::make_pair(lower_bound, i);
    }

    std::sort(lower_bounds.begin(), lower_bounds.begin() + num_valid, [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    for (int i = 0; i < num_valid; i++) {
      const auto& lb = lower_bounds[i];
      if (result.worst_distance() < lb.first) {
        break;
      }

      if (node.node_type.sub.children[lb.second] != OctreeNode::INVALID_NODE) {
        knn_search(query, node.node_type.sub.children[lb.second], result, setting);
      }
    }

    return true;
  }

public:
  const PointCloud& points;
  std::vector<size_t> indices;

  Eigen::Vector4d origin;
  Eigen::Vector4d extent;

  OctreeNode::NodeIndexType root;
  std::vector<OctreeNode> nodes;
};
}  // namespace small_gicp