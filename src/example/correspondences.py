#!/usr/bin/python3
# SPDX-FileCopyrightText: Copyright 2026 small_gicp contributors
# SPDX-License-Identifier: MIT
"""Extract nearest-neighbor pairs after registration using the existing Python API."""
import numpy

import small_gicp


def find_correspondences(target, source, target_tree, T_target_source,
                         max_correspondence_distance, num_threads=1):
  """Return an (M, 2) int64 array of (source_index, target_index) pairs.

  target_tree must index target. Indices refer to the supplied PointClouds,
  which may be downsampled, rather than the raw arrays used to create them.
  Each source point has at most one match; target indices may be repeated.
  These are nearest neighbors at T_target_source, not necessarily the pairs
  used by the optimizer at its last linearization, so their count may differ
  from result.num_inliers. Extracting them requires an additional nearest-
  neighbor query. This example is for point clouds and a KdTree, not
  GaussianVoxelMap/VGICP correspondences.
  """
  if not numpy.isfinite(max_correspondence_distance) or max_correspondence_distance < 0:
    raise ValueError('max_correspondence_distance must be finite and nonnegative')

  # Avoid accessing points() or querying a KdTree for empty point clouds.
  if len(target) == 0 or len(source) == 0:
    return numpy.empty((0, 2), dtype=numpy.int64)

  source_xyz = source.points()[:, :3]
  transformed = source_xyz @ T_target_source[:3, :3].T + T_target_source[:3, 3]
  target_indices, squared_distances = target_tree.batch_nearest_neighbor_search(
    transformed, num_threads=num_threads)
  target_indices = numpy.asarray(target_indices)
  squared_distances = numpy.asarray(squared_distances)
  # Search indices use size_t, so a missing neighbor can be an unsigned sentinel.
  valid = ((target_indices < len(target)) & numpy.isfinite(squared_distances) &
           (squared_distances <= max_correspondence_distance ** 2))
  return numpy.column_stack((numpy.flatnonzero(valid), target_indices[valid].astype(numpy.int64)))


def main():
  # Synthetic, non-symmetric data: no ROS, external dataset, or viewer required.
  rng = numpy.random.default_rng(128)
  target_raw = rng.uniform(-2.0, 2.0, (300, 3))
  angle = 0.1
  rotation = numpy.array([[numpy.cos(angle), -numpy.sin(angle), 0.0],
                          [numpy.sin(angle), numpy.cos(angle), 0.0],
                          [0.0, 0.0, 1.0]])
  translation = numpy.array([0.1, -0.05, 0.03])
  source_raw = (target_raw - translation) @ rotation

  # Keep the processed clouds so that the returned indices have clear owners.
  target, target_tree = small_gicp.preprocess_points(target_raw, downsampling_resolution=0.05)
  source, _ = small_gicp.preprocess_points(source_raw, downsampling_resolution=0.05)
  max_distance = 0.5
  result = small_gicp.align(target, source, target_tree,
                            max_correspondence_distance=max_distance)
  pairs = find_correspondences(target, source, target_tree, result.T_target_source, max_distance)
  print(result)
  print('Pairs at the final transform (source index, target index):')
  print(pairs[:10])
  print(f'{len(pairs)} pairs in the processed point clouds')


if __name__ == '__main__':
  main()
