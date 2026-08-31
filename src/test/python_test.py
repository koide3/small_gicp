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

# Build colored point clouds for colored ICP.
# Colors are a linear function of the position in the target frame; source points are mapped into the
# target frame with the ground truth transformation so that both clouds share the same color field.
def make_colored_clouds(gt_T_target_source, target_raw_numpy, source_raw_numpy, downsampling_resolution=0.25, num_threads=2):
  def color_field(points_xyz):
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    return numpy.stack([0.5 + 0.3 * x, 0.2 * y + 0.1, 0.5 + 0.25 * z], axis=1)

  target_raw = small_gicp.PointCloud(target_raw_numpy)
  target_raw.set_colors(color_field(target_raw_numpy[:, :3]))

  source_in_target = source_raw_numpy[:, :3] @ gt_T_target_source[:3, :3].T + gt_T_target_source[:3, 3]
  source_raw = small_gicp.PointCloud(source_raw_numpy)
  source_raw.set_colors(color_field(source_in_target))

  # Downsampling averages the colors within each voxel together with the points
  target = small_gicp.voxelgrid_sampling(target_raw, downsampling_resolution, num_threads=num_threads)
  source = small_gicp.voxelgrid_sampling(source_raw, downsampling_resolution, num_threads=num_threads)

  target_tree = small_gicp.KdTree(target)
  small_gicp.estimate_normals(target, target_tree, num_threads=num_threads)
  small_gicp.estimate_color_gradients(target, target_tree, num_neighbors=30, num_threads=num_threads, max_radius=2.0)

  return target, source, target_tree

# Colored ICP registration test
def test_colored_icp_registration(load_points):
  gt_T_target_source, target_raw_numpy, source_raw_numpy = load_points

  target, source, target_tree = make_colored_clouds(gt_T_target_source, target_raw_numpy, source_raw_numpy)
  assert target.size() > 0 and source.size() > 0
  assert numpy.any(numpy.abs(target.colors()[:, :3]) > 1e-3)
  assert numpy.any(numpy.abs(target.color_grads()[:, :3]) > 1e-3)

  result = small_gicp.align(target, source, target_tree, registration_type='COLORED_ICP', max_correspondence_distance=1.0, num_threads=2)
  verify_result(result.T_target_source, gt_T_target_source)

  # COLORED_ICP is not supported with numpy inputs and returns the identity transformation
  result = small_gicp.align(target_raw_numpy, source_raw_numpy, registration_type='COLORED_ICP', downsampling_resolution=0.25)
  assert numpy.allclose(result.T_target_source, numpy.eye(4))

# Color gradient estimation test with a linear color field on a plane
def test_estimate_color_gradients_linear(load_points):
  n = 20
  grid = numpy.linspace(-1.0, 1.0, n)
  x, y = numpy.meshgrid(grid, grid)
  points = numpy.stack([x.ravel(), y.ravel(), numpy.zeros(n * n)], axis=1)

  # I(p) = a . p with equal RGB channels; the gradient projected on the plane (normal = z) is (a0, a1, 0)
  a = numpy.array([0.3, -0.2, 0.15])
  intensity = points @ a
  colors = numpy.stack([intensity, intensity, intensity], axis=1)

  cloud = small_gicp.PointCloud(points)
  cloud.set_colors(colors)
  tree = small_gicp.KdTree(cloud)
  small_gicp.estimate_normals(cloud, tree)
  small_gicp.estimate_color_gradients(cloud, tree, num_neighbors=8, max_radius=0.5)

  grads = cloud.color_grads().reshape(n, n, 4)
  interior = grads[2:-2, 2:-2, :].reshape(-1, 4)
  assert numpy.allclose(interior[:, :3], numpy.array([a[0], a[1], 0.0]), atol=1e-5)

# set_colors input validation test
def test_set_colors_validation(load_points):
  _, points_numpy, _ = load_points

  points = small_gicp.PointCloud(points_numpy)
  points.set_colors(numpy.full((points_numpy.shape[0], 3), 0.5))
  assert numpy.allclose(points.colors()[:, :3], 0.5)
  assert numpy.allclose(points.colors()[:, 3], 0.0)
  assert numpy.allclose(points.color(0), [0.5, 0.5, 0.5, 0.0])

  # Invalid shapes are rejected with a warning and leave the colors unchanged
  points.set_colors(numpy.ones((points_numpy.shape[0], 2)))
  points.set_colors(numpy.ones((points_numpy.shape[0] + 3, 3)))
  points.set_colors(numpy.ones((points_numpy.shape[0] + 3, 4)))
  assert numpy.allclose(points.colors()[:, :3], 0.5)

  # Nx4 colors are accepted as-is
  points.set_colors(numpy.concatenate([numpy.full((points_numpy.shape[0], 3), 0.25), numpy.zeros((points_numpy.shape[0], 1))], axis=1))
  assert numpy.allclose(points.color(0), [0.25, 0.25, 0.25, 0.0])
