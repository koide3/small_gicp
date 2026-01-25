// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-FileCopyrightText: Copyright 2025 Ikhyeon Cho
// SPDX-License-Identifier: MIT
#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <small_gicp/ros/ros_impl.hpp>

namespace small_gicp {

/// @brief Convert ROS2 PointCloud2 message to small_gicp::PointCloud
/// @note  Only XYZ coordinates are extracted. Invalid points (NaN/Inf) are skipped.
/// @warning Normals and covariances are UNINITIALIZED (contain garbage values).
///          You MUST call estimate_normals_covariances() before using this cloud for registration.
/// @param msg  Input ROS2 PointCloud2 message
/// @return     Shared pointer to PointCloud. Returns empty cloud if message is invalid.
inline PointCloud::Ptr from_ros_msg(const sensor_msgs::msg::PointCloud2& msg) {
  return detail::from_impl<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointField>(msg);
}

/// @brief Convert small_gicp::PointCloud to ROS2 PointCloud2 message
/// @note  By default, only XYZ coordinates are exported.
/// @param cloud        Input point cloud
/// @param frame_id     Frame ID for message header
/// @param stamp        Timestamp for message header
/// @param with_normals If true, include normal_x/y/z fields (default: false)
/// @warning If with_normals is true, caller must ensure normals are valid
///          (i.e., estimate_normals_covariances() was called beforehand)
/// @return ROS2 PointCloud2 message
inline sensor_msgs::msg::PointCloud2
to_ros_msg(const PointCloud& cloud, const std::string& frame_id, const builtin_interfaces::msg::Time& stamp = builtin_interfaces::msg::Time(), bool with_normals = false) {
  //
  return detail::to_impl<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointField>(cloud, frame_id, stamp, with_normals);
}

}  // namespace small_gicp
