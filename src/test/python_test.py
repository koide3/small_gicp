#!/usr/bin/python3
# SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
# SPDX-License-Identifier: MIT
import numpy
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

import small_gicp


# Basic registation example with small_gicp.PointCloud
def example_small1(target_raw_numpy : numpy.ndarray, source_raw_numpy : numpy.ndarray):
  # Convert numpy arrays (Nx3 or Nx4) to small_gicp.PointCloud
  target_raw = small_gicp.PointCloud(target_raw_numpy)
  source_raw = small_gicp.PointCloud(source_raw_numpy)

  # Preprocess point clouds
  target, target_tree = small_gicp.preprocess_points(target_raw, downsampling_resolution=0.25)
  source, source_tree = small_gicp.preprocess_points(source_raw, downsampling_resolution=0.25)
  
  result = small_gicp.align(target, source, target_tree)
  
  return result.T_target_source
  
# Example to perform each preprocessing and registration separately
def example_small2(target_raw_numpy : numpy.ndarray, source_raw_numpy : numpy.ndarray):
  # Convert numpy arrays (Nx3 or Nx4) to small_gicp.PointCloud
  target_raw = small_gicp.PointCloud(target_raw_numpy)
  source_raw = small_gicp.PointCloud(source_raw_numpy)

  # Downsampling
  target = small_gicp.voxelgrid_sampling(target_raw, 0.25)
  source = small_gicp.voxelgrid_sampling(source_raw, 0.25)
  
  # KdTree construction
  target_tree = small_gicp.KdTree(target)
  source_tree = small_gicp.KdTree(source)
  
  # Estimate covariances
  small_gicp.estimate_covariances(target, target_tree)
  small_gicp.estimate_covariances(source, source_tree)

  # Align point clouds  
  result = small_gicp.align(target, source, target_tree)
  
  return result.T_target_source


### Following functions are for testing ###

# Verity the estimated transformation matrix (for testing)
def verify_result(T_target_source, gt_T_target_source):
  error = numpy.linalg.inv(T_target_source) @ gt_T_target_source
  error_trans = numpy.linalg.norm(error[:3, 3])
  error_rot = Rotation.from_matrix(error[:3, :3]).magnitude()
  
  assert error_trans < 0.05
  assert error_rot < 0.05

import pytest

# Load the point clouds and the ground truth transformation matrix
@pytest.fixture(scope='module', autouse=True)
def load_points():
  gt_T_target_source = numpy.loadtxt('data/T_target_source.txt')  # Load the ground truth transformation matrix
  target_raw = small_gicp.read_ply(('data/target.ply'))  # Read the target point cloud (small_gicp.PointCloud)
  source_raw = small_gicp.read_ply(('data/source.ply'))  # Read the source point cloud (small_gicp.PointCloud)

  target_raw_numpy = target_raw.points()                    # Nx4 numpy array of the target point cloud
  source_raw_numpy = source_raw.points()                    # Nx4 numpy array of the source point cloud
  
  yield (gt_T_target_source, target_raw_numpy, source_raw_numpy)

# Check if the point clouds are loaded correctly
def test_load_points(load_points):
  gt_T_target_source, target_raw_numpy, source_raw_numpy = load_points
  assert gt_T_target_source.shape[0] == 4 and gt_T_target_source.shape[1] == 4
  assert len(target_raw_numpy) > 0 and target_raw_numpy.shape[1] == 4
  assert len(source_raw_numpy) > 0 and source_raw_numpy.shape[1] == 4

# Basic point cloud test
def test_points(load_points):
  _, points_numpy, _ = load_points

  points = small_gicp.PointCloud(points_numpy)
  assert points.size() == points_numpy.shape[0]
  assert numpy.all(numpy.abs(points.points() - points_numpy) < 1e-6)

  points = small_gicp.PointCloud(points_numpy[:, :3])
  assert points.size() == points_numpy.shape[0]
  assert numpy.all(numpy.abs(points.points() - points_numpy) < 1e-6)
  
  for i in range(10):
    assert numpy.all(numpy.abs(points.point(i) - points_numpy[i]) < 1e-6)
  

# Downsampling test
def test_downsampling(load_points):
  _, points_numpy, _ = load_points

  downsampled = small_gicp.voxelgrid_sampling(points_numpy, 0.25)
  assert downsampled.size() > 0
    
  downsampled2 = small_gicp.voxelgrid_sampling(points_numpy, 0.25, num_threads=2)
  assert abs(1.0 - downsampled.size() / downsampled2.size()) < 0.05
  
  downsampled2 = small_gicp.voxelgrid_sampling(small_gicp.PointCloud(points_numpy), 0.25)
  assert downsampled.size() == downsampled2.size()
  
  downsampled2 = small_gicp.voxelgrid_sampling(small_gicp.PointCloud(points_numpy), 0.25, num_threads=2)
  assert abs(1.0 - downsampled.size() / downsampled2.size()) < 0.05

# Preprocess test
def test_preprocess(load_points):
  _, points_numpy, _ = load_points

  downsampled, _ = small_gicp.preprocess_points(points_numpy, downsampling_resolution=0.25)
  assert downsampled.size() > 0

  downsampled2, _ = small_gicp.preprocess_points(points_numpy, downsampling_resolution=0.25, num_threads=2)
  assert abs(1.0 - downsampled.size() / downsampled2.size()) < 0.05
  
  downsampled2, _ = small_gicp.preprocess_points(small_gicp.PointCloud(points_numpy), downsampling_resolution=0.25)
  assert downsampled.size() == downsampled2.size()
  
  downsampled2, _ = small_gicp.preprocess_points(small_gicp.PointCloud(points_numpy), downsampling_resolution=0.25, num_threads=2)
  assert abs(1.0 - downsampled.size() / downsampled2.size()) < 0.05

# Voxelmap test
def test_voxelmap(load_points):
  _, points_numpy, _ = load_points

  downsampled = small_gicp.voxelgrid_sampling(points_numpy, 0.25)
  small_gicp.estimate_covariances(downsampled)

  voxelmap = small_gicp.GaussianVoxelMap(0.5)
  voxelmap.insert(downsampled)
  
  assert voxelmap.size() > 0
  assert voxelmap.size() == len(voxelmap)

# Factor test
def test_factors(load_points):
  gt_T_target_source, target_raw_numpy, source_raw_numpy = load_points

  target, target_tree = small_gicp.preprocess_points(target_raw_numpy, downsampling_resolution=0.25)
  source, source_tree = small_gicp.preprocess_points(source_raw_numpy, downsampling_resolution=0.25)

  result = small_gicp.align(target, source, target_tree, gt_T_target_source)
  result = small_gicp.align(target, source, target_tree, result.T_target_source)

  factors = [small_gicp.GICPFactor()]
  rejector = small_gicp.DistanceRejector()

  sum_H = numpy.zeros((6, 6))
  sum_b = numpy.zeros(6)
  sum_e = 0.0

  for i in range(source.size()):
    succ, H, b, e = factors[0].linearize(target, source, target_tree, result.T_target_source, i, rejector)
    if succ:
      sum_H += H
      sum_b += b
      sum_e += e

  assert numpy.max(numpy.abs(result.H - sum_H) / result.H) < 0.05

# Registration test
def test_registration(load_points):
  gt_T_target_source, target_raw_numpy, source_raw_numpy = load_points

  result = small_gicp.align(target_raw_numpy, source_raw_numpy, downsampling_resolution=0.25)
  verify_result(result.T_target_source, gt_T_target_source)

  result = small_gicp.align(target_raw_numpy, source_raw_numpy, downsampling_resolution=0.25, num_threads=2)
  verify_result(result.T_target_source, gt_T_target_source)

  target, target_tree = small_gicp.preprocess_points(target_raw_numpy, downsampling_resolution=0.25)
  source, source_tree = small_gicp.preprocess_points(source_raw_numpy, downsampling_resolution=0.25)

  result = small_gicp.align(target, source)
  verify_result(result.T_target_source, gt_T_target_source)

  result = small_gicp.align(target, source, target_tree)
  verify_result(result.T_target_source, gt_T_target_source)

  target_voxelmap = small_gicp.GaussianVoxelMap(0.5)
  target_voxelmap.insert(target)
  
  result = small_gicp.align(target_voxelmap, source)
  verify_result(result.T_target_source, gt_T_target_source)

# KdTree test
def test_kdtree(load_points):
  _, target_raw_numpy, source_raw_numpy = load_points

  target, target_tree = small_gicp.preprocess_points(target_raw_numpy, downsampling_resolution=0.5)
  source, source_tree = small_gicp.preprocess_points(source_raw_numpy, downsampling_resolution=0.5)
  
  target_tree_ref = KDTree(target.points())
  source_tree_ref = KDTree(source.points())
  
  def batch_test(points, queries, tree, tree_ref, num_threads):
    # test for batch interface
    k_dists_ref, k_indices_ref = tree_ref.query(queries, k=1)
    k_indices, k_sq_dists = tree.batch_nearest_neighbor_search(queries)
    assert numpy.all(numpy.abs(numpy.square(k_dists_ref) - k_sq_dists) < 1e-6)
    assert numpy.all(numpy.abs(numpy.linalg.norm(points[k_indices] - queries, axis=1) ** 2 - k_sq_dists) < 1e-6)
    
    for k in [2, 10]:
      k_dists_ref, k_indices_ref = tree_ref.query(queries, k=k)
      k_sq_dists_ref, k_indices_ref = numpy.array(k_dists_ref) ** 2, numpy.array(k_indices_ref)
      
      k_indices, k_sq_dists = tree.batch_knn_search(queries, k, num_threads=num_threads)
      k_indices, k_sq_dists = numpy.array(k_indices), numpy.array(k_sq_dists)

      assert(numpy.all(numpy.abs(k_sq_dists_ref - k_sq_dists) < 1e-6))
      for i in range(k):
        diff = numpy.linalg.norm(points[k_indices[:, i]] - queries, axis=1) ** 2 - k_sq_dists[:, i]
        assert(numpy.all(numpy.abs(diff) < 1e-6))

    # test for single query interface
    if num_threads != 1:
      return

    k_dists_ref, k_indices_ref = tree_ref.query(queries, k=1)
    k_indices2, k_sq_dists2 = [], []
    for query in queries:
      found, index, sq_dist = tree.nearest_neighbor_search(query[:3])
      assert found
      k_indices2.append(index)
      k_sq_dists2.append(sq_dist)
    
    assert numpy.all(numpy.abs(numpy.square(k_dists_ref) - k_sq_dists2) < 1e-6)
    assert numpy.all(numpy.abs(numpy.linalg.norm(points[k_indices2] - queries, axis=1) ** 2 - k_sq_dists2) < 1e-6)

    for k in [2, 10]:
      k_dists_ref, k_indices_ref = tree_ref.query(queries, k=k)
      k_sq_dists_ref, k_indices_ref = numpy.array(k_dists_ref) ** 2, numpy.array(k_indices_ref)
      
      k_indices2, k_sq_dists2 = [], []
      for query in queries:
        indices, sq_dists = tree.knn_search(query[:3], k)
        k_indices2.append(indices)
        k_sq_dists2.append(sq_dists)
      k_indices2, k_sq_dists2 = numpy.array(k_indices2), numpy.array(k_sq_dists2)
      
      assert(numpy.all(numpy.abs(k_sq_dists_ref - k_sq_dists2) < 1e-6))
      for i in range(k):
        diff = numpy.linalg.norm(points[k_indices2[:, i]] - queries, axis=1) ** 2 - k_sq_dists2[:, i]
        assert(numpy.all(numpy.abs(diff) < 1e-6))
      

  for num_threads in [1, 2]:
    batch_test(target.points(), target.points(), target_tree, target_tree_ref, num_threads=num_threads)
    batch_test(target.points(), source.points(), target_tree, target_tree_ref, num_threads=num_threads)
    batch_test(source.points(), target.points(), source_tree, source_tree_ref, num_threads=num_threads)
