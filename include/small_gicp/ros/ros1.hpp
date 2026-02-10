// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-FileCopyrightText: Copyright 2025 Ikhyeon Cho
// SPDX-License-Identifier: MIT
#pragma once

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>

#include <small_gicp/ros/ros_impl.hpp>

namespace small_gicp {

/// @brief Convert ROS1 PointCloud2 message to small_gicp::PointCloud
/// @note  Only XYZ coordinates are extracted. Invalid points (NaN/Inf) are skipped.
/// @warning Normals and covariances are UNINITIALIZED (contain garbage values).
///          You MUST call estimate_normals_covariances() before using this cloud for registration.
/// @param msg  Input ROS1 PointCloud2 message
/// @return     Shared pointer to PointCloud. Returns empty cloud if message is invalid.
inline PointCloud::Ptr from_ros_msg(const sensor_msgs::PointCloud2& msg) {
  return detail::from_impl<sensor_msgs::PointCloud2, sensor_msgs::PointField>(msg);
}

/// @brief Convert small_gicp::PointCloud to ROS1 PointCloud2 message
/// @note  By default, only XYZ coordinates are exported.
/// @param cloud        Input point cloud
/// @param frame_id     Frame ID for message header
/// @param stamp        Timestamp for message header (default: ros::Time::now())
/// @param with_normals If true, include normal_x/y/z fields (default: false)
/// @warning If with_normals is true, caller must ensure normals are valid
///          (i.e., estimate_normals_covariances() was called beforehand)
/// @return ROS1 PointCloud2 message
inline sensor_msgs::PointCloud2 to_ros_msg(const PointCloud& cloud, const std::string& frame_id, const ros::Time& stamp = ros::Time::now(), bool with_normals = false) {
  //
  return detail::to_impl<sensor_msgs::PointCloud2, sensor_msgs::PointField>(cloud, frame_id, stamp, with_normals);
}

}  // namespace small_gicp
