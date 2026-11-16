# SPDX-FileCopyrightText: Copyright 2026 small_gicp contributors
# SPDX-License-Identifier: MIT
import numpy
import pytest

import small_gicp
from correspondences import find_correspondences


def cloud(points):
  return small_gicp.PointCloud(numpy.asarray(points, dtype=float).reshape(-1, 3))


@pytest.mark.parametrize('num_threads', [1, 2, 4])
def test_final_transform_and_distance_threshold(num_threads):
  target = cloud([[0, 0, 0], [2, 0, 0], [5, 0, 0]])
  source = cloud([[-1, 0, 0], [1.5, 0, 0], [4.5001, 0, 0], [20, 0, 0]])
  transform = numpy.eye(4)
  transform[0, 3] = 1.0
  pairs = find_correspondences(target, source, small_gicp.KdTree(target), transform, 0.5, num_threads)
  numpy.testing.assert_array_equal(pairs, [[0, 0], [1, 1]])
  assert pairs.dtype == numpy.int64


def test_nearest_neighbor_not_all_pairs_or_one_to_one():
  target = cloud([[0, 0, 0], [0.125, 0, 0]])
  source = cloud([[0.03125, 0, 0], [0.015625, 0, 0]])
  pairs = find_correspondences(target, source, small_gicp.KdTree(target), numpy.eye(4), 0.25)
  numpy.testing.assert_array_equal(pairs, [[0, 0], [1, 0]])


def test_updating_transform_can_change_nearest_neighbor():
  target = cloud([[0, 0, 0], [1, 0, 0]])
  source = cloud([[0.45, 0, 0]])
  tree = small_gicp.KdTree(target)
  transform = numpy.eye(4)
  numpy.testing.assert_array_equal(find_correspondences(target, source, tree, transform, 1.0), [[0, 0]])
  transform[0, 3] = 0.2
  numpy.testing.assert_array_equal(find_correspondences(target, source, tree, transform, 1.0), [[0, 1]])


@pytest.mark.parametrize('target_points,source_points', [([], [[0, 0, 0]]), ([[0, 0, 0]], []), ([], [])])
def test_empty_clouds(target_points, source_points):
  # No tree is constructed or queried for an empty cloud.
  pairs = find_correspondences(cloud(target_points), cloud(source_points), None, numpy.eye(4), 1.0)
  assert pairs.shape == (0, 2)
  assert pairs.dtype == numpy.int64


def test_no_matches_and_zero_threshold():
  target = cloud([[0, 0, 0]])
  source = cloud([[0, 0, 0], [1, 0, 0]])
  tree = small_gicp.KdTree(target)
  numpy.testing.assert_array_equal(find_correspondences(target, source, tree, numpy.eye(4), 0), [[0, 0]])
  assert find_correspondences(target, cloud([[1, 0, 0]]), tree, numpy.eye(4), 0.5).shape == (0, 2)


@pytest.mark.parametrize('distance', [-1, numpy.inf, numpy.nan])
def test_invalid_distance(distance):
  with pytest.raises(ValueError):
    find_correspondences(cloud([]), cloud([]), None, numpy.eye(4), distance)


@pytest.mark.parametrize('num_threads', [1, 2, 4])
def test_gicp_pairs_index_processed_clouds(num_threads):
  rng = numpy.random.default_rng(128)
  target_raw = rng.uniform(-2, 2, (200, 3))
  angle = 0.1
  rotation = numpy.array([[numpy.cos(angle), -numpy.sin(angle), 0],
                          [numpy.sin(angle), numpy.cos(angle), 0], [0, 0, 1]])
  translation = numpy.array([0.1, -0.05, 0.03])
  # Duplicates ensure that preprocessing changes the raw index space.
  target_raw = numpy.repeat(target_raw, 2, axis=0)
  source_raw = (target_raw[::-1] - translation) @ rotation
  target, tree = small_gicp.preprocess_points(target_raw, downsampling_resolution=0.05, num_threads=num_threads)
  source, _ = small_gicp.preprocess_points(source_raw, downsampling_resolution=0.05, num_threads=num_threads)
  assert len(target) < len(target_raw)
  result = small_gicp.align(target, source, tree, max_correspondence_distance=0.5, num_threads=num_threads)
  assert result.converged
  numpy.testing.assert_allclose(result.T_target_source[:3, :3], rotation, atol=1e-3)
  numpy.testing.assert_allclose(result.T_target_source[:3, 3], translation, atol=1e-3)

  pairs = find_correspondences(target, source, tree, result.T_target_source, 0.5, num_threads)
  # Independent brute-force reference on the actual processed arrays.
  transformed = source.points()[:, :3] @ result.T_target_source[:3, :3].T + result.T_target_source[:3, 3]
  squared = numpy.sum((transformed[:, None, :] - target.points()[None, :, :3]) ** 2, axis=2)
  nearest = numpy.argmin(squared, axis=1)
  valid = squared[numpy.arange(len(source)), nearest] <= 0.5 ** 2
  expected = numpy.column_stack((numpy.flatnonzero(valid), nearest[valid]))
  numpy.testing.assert_array_equal(pairs, expected)
  numpy.testing.assert_array_equal(pairs, find_correspondences(target, source, tree, result.T_target_source, 0.5, 1))
  assert len(pairs) == len(source)
