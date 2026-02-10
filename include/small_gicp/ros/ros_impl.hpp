// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-FileCopyrightText: Copyright 2025 Ikhyeon Cho
// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <iostream>

#include <small_gicp/points/point_cloud.hpp>

namespace small_gicp {

namespace detail {

/// @brief Parse XYZ field offsets from PointCloud2 fields
/// @tparam PointFieldT  PointField type (sensor_msgs::PointField or sensor_msgs::msg::PointField)
template <typename PointFieldT>
struct FieldOffsets {
  int x = -1;
  int y = -1;
  int z = -1;

  bool has_xyz() const { return x >= 0 && y >= 0 && z >= 0; }

  template <typename FieldsT>
  static FieldOffsets parse(const FieldsT& fields) {
    FieldOffsets offsets;
    for (const auto& field : fields) {
      if (field.name == "x") {
        offsets.x = field.offset;
      } else if (field.name == "y") {
        offsets.y = field.offset;
      } else if (field.name == "z") {
        offsets.z = field.offset;
      }
    }
    return offsets;
  }
};

/// @brief Convert PointCloud2 message to small_gicp::PointCloud (implementation)
/// @tparam PointCloud2T  PointCloud2 message type
/// @tparam PointFieldT   PointField type
template <typename PointCloud2T, typename PointFieldT>
PointCloud::Ptr from_impl(const PointCloud2T& msg) {
  auto cloud = std::make_shared<PointCloud>();

  const size_t num_points = static_cast<size_t>(msg.width) * msg.height;
  if (num_points == 0) {
    return cloud;
  }

  const auto offsets = FieldOffsets<PointFieldT>::parse(msg.fields);
  if (!offsets.has_xyz()) {
    std::cerr << "warning: PointCloud2 message does not have XYZ fields" << std::endl;
    return cloud;
  }

  // Reserve memory without initialization (avoid unnecessary default construction)
  cloud->points.reserve(num_points);

  const uint8_t* data_ptr = msg.data.data();
  const size_t point_step = msg.point_step;

  for (size_t i = 0; i < num_points; i++) {
    const uint8_t* pt = data_ptr + i * point_step;

    const float x = *reinterpret_cast<const float*>(pt + offsets.x);
    const float y = *reinterpret_cast<const float*>(pt + offsets.y);
    const float z = *reinterpret_cast<const float*>(pt + offsets.z);

    // Skip invalid points (NaN or Inf)
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }

    cloud->points.emplace_back(x, y, z, 1.0);
  }

  // Resize normals and covs to match points size
  cloud->normals.resize(cloud->points.size());
  cloud->covs.resize(cloud->points.size());

  return cloud;
}

/// @brief Convert small_gicp::PointCloud to PointCloud2 message (implementation)
/// @tparam PointCloud2T  PointCloud2 message type
/// @tparam PointFieldT   PointField type
/// @tparam TimeT         Timestamp type
template <typename PointCloud2T, typename PointFieldT, typename TimeT>
PointCloud2T to_impl(const PointCloud& cloud, const std::string& frame_id, const TimeT& stamp, bool with_normals) {
  PointCloud2T msg;

  msg.header.frame_id = frame_id;
  msg.header.stamp = stamp;
  msg.height = 1;
  msg.width = static_cast<uint32_t>(cloud.size());
  msg.is_bigendian = false;
  msg.is_dense = true;

  if (cloud.empty()) {
    msg.point_step = 0;
    msg.row_step = 0;
    return msg;
  }

  // Build fields
  PointFieldT field;
  field.count = 1;
  field.datatype = PointFieldT::FLOAT32;

  uint32_t offset = 0;

  field.name = "x";
  field.offset = offset;
  msg.fields.push_back(field);
  offset += sizeof(float);

  field.name = "y";
  field.offset = offset;
  msg.fields.push_back(field);
  offset += sizeof(float);

  field.name = "z";
  field.offset = offset;
  msg.fields.push_back(field);
  offset += sizeof(float);

  if (with_normals) {
    field.name = "normal_x";
    field.offset = offset;
    msg.fields.push_back(field);
    offset += sizeof(float);

    field.name = "normal_y";
    field.offset = offset;
    msg.fields.push_back(field);
    offset += sizeof(float);

    field.name = "normal_z";
    field.offset = offset;
    msg.fields.push_back(field);
    offset += sizeof(float);
  }

  msg.point_step = offset;
  msg.row_step = msg.point_step * msg.width;

  // Fill data
  msg.data.resize(msg.row_step);
  uint8_t* data_ptr = msg.data.data();

  for (size_t i = 0; i < cloud.size(); i++) {
    uint8_t* pt = data_ptr + i * msg.point_step;

    const Eigen::Vector4d& point = cloud.point(i);
    *reinterpret_cast<float*>(pt + 0) = static_cast<float>(point.x());
    *reinterpret_cast<float*>(pt + 4) = static_cast<float>(point.y());
    *reinterpret_cast<float*>(pt + 8) = static_cast<float>(point.z());

    if (with_normals) {
      const Eigen::Vector4d& normal = cloud.normal(i);
      *reinterpret_cast<float*>(pt + 12) = static_cast<float>(normal.x());
      *reinterpret_cast<float*>(pt + 16) = static_cast<float>(normal.y());
      *reinterpret_cast<float*>(pt + 20) = static_cast<float>(normal.z());
    }
  }

  return msg;
}

}  // namespace detail

}  // namespace small_gicp
