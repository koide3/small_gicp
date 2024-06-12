#!/usr/bin/python3
# SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
# SPDX-License-Identifier: MIT
import numpy
from scipy.spatial.transform import Rotation

import small_gicp


# Basic registation example with numpy arrays
def example_numpy1(target_raw_numpy : numpy.ndarray, source_raw_numpy : numpy.ndarray):
  print('*** example_numpy1 ***')

  # Example A : Perform registration with numpy arrays
  # Arguments
  # - target_points               : Nx4 or Nx3 numpy array of the target point cloud
  # - source_points               : Nx4 or Nx3 numpy array of the source point cloud
  # Optional arguments
  # - init_T_target_source        : Initial guess of the transformation matrix (4x4 numpy array)
  # - registration_type           : Registration type ("ICP", "PLANE_ICP", "GICP", "VGICP")
  # - voxel_resolution            : Voxel resolution for VGICP
  # - downsampling_resolution     : Downsampling resolution
  # - max_correspondence_distance : Maximum correspondence distance
  # - num_threads                 : Number of threads
  result = small_gicp.align(target_raw_numpy, source_raw_numpy, downsampling_resolution=0.25)

  print('--- registration result ---')
  print(result)

  return result.T_target_source

# Example to perform preprocessing and registration separately
def example_numpy2(target_raw_numpy : numpy.ndarray, source_raw_numpy : numpy.ndarray):
  print('*** example_numpy2 ***')

  # Example B : Perform preprocessing and registration separately

  # Preprocess point clouds
  # Arguments
  # - points                      : Nx4 or Nx3 numpy array of the target point cloud
  # Optional arguments
  # - downsampling_resolution     : Downsampling resolution
  # - num_neighbors               : Number of neighbors for normal and covariance estimation
  # - num_threads                 : Number of threads
  target, target_tree = small_gicp.preprocess_points(target_raw_numpy, downsampling_resolution=0.25)
  source, source_tree = small_gicp.preprocess_points(source_raw_numpy, downsampling_resolution=0.25)

  print('preprocessed target=', target)
  print('preprocessed source=', source)

  # Align point clouds
  # Arguments
  # - target                      : Target point cloud (small_gicp.PointCloud)
  # - source                      : Source point cloud (small_gicp.PointCloud)
  # - target_tree                 : KD-tree of the target point cloud
  # Optional arguments
  # - init_T_target_source        : Initial guess of the transformation matrix (4x4 numpy array)
  # - max_correspondence_distance : Maximum correspondence distance
  # - num_threads                 : Number of threads
  result = small_gicp.align(target, source, target_tree)

  print('--- registration result ---')
  print(result)

  return result.T_target_source


# Basic registation example with small_gicp.PointCloud
def example_small1(target_raw_numpy : numpy.ndarray, source_raw_numpy : numpy.ndarray):
  print('*** example_small1 ***')

  # Convert numpy arrays (Nx3 or Nx4) to small_gicp.PointCloud
  target_raw = small_gicp.PointCloud(target_raw_numpy)
  source_raw = small_gicp.PointCloud(source_raw_numpy)

  # Preprocess point clouds
  target, target_tree = small_gicp.preprocess_points(target_raw, downsampling_resolution=0.25)
  source, source_tree = small_gicp.preprocess_points(source_raw, downsampling_resolution=0.25)

  print('preprocessed target=', target)
  print('preprocessed source=', source)

  result = small_gicp.align(target, source, target_tree)

  print('--- registration result ---')
  print(result)
  
  return result.T_target_source
  
# Example to perform each preprocessing and registration separately
def example_small2(target_raw_numpy : numpy.ndarray, source_raw_numpy : numpy.ndarray):
  print('*** example_small2 ***')

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

  print('preprocessed target=', target)
  print('preprocessed source=', source)

  # Align point clouds  
  result = small_gicp.align(target, source, target_tree)

  print('--- registration result ---')
  print(result)
  
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
  print('--- gt_T_target_source ---')
  print(gt_T_target_source)

  target_raw = small_gicp.read_ply(('data/target.ply'))  # Read the target point cloud (small_gicp.PointCloud)
  source_raw = small_gicp.read_ply(('data/source.ply'))  # Read the source point cloud (small_gicp.PointCloud)

  target_raw_numpy = target_raw.points()                    # Nx4 numpy array of the target point cloud
  source_raw_numpy = source_raw.points()                    # Nx4 numpy array of the source point cloud
  
  yield (gt_T_target_source, target_raw_numpy, source_raw_numpy)

# Check if the point clouds are loaded correctly
def test_load_points(load_points):
  gt_T_target_source, target_raw_numpy, source_raw_numpy = load_points
  assert gt_T_target_source.shape[0] == 4 and gt_T_target_source.shape[1] == 4
  assert len(target_raw_numpy) > 0
  assert len(source_raw_numpy) > 0

def test_example_numpy1(load_points):
  gt_T_target_source, target_raw_numpy, source_raw_numpy = load_points
  T_target_source = example_numpy1(target_raw_numpy, source_raw_numpy)
  verify_result(T_target_source, gt_T_target_source)

def test_example_numpy2(load_points):
  gt_T_target_source, target_raw_numpy, source_raw_numpy = load_points
  T_target_source = example_numpy2(target_raw_numpy, source_raw_numpy)
  verify_result(T_target_source, gt_T_target_source)

def test_example_small1(load_points):
  gt_T_target_source, target_raw_numpy, source_raw_numpy = load_points
  T_target_source = example_small1(target_raw_numpy, source_raw_numpy)
  verify_result(T_target_source, gt_T_target_source)

def test_example_small2(load_points):
  gt_T_target_source, target_raw_numpy, source_raw_numpy = load_points
  T_target_source = example_small2(target_raw_numpy, source_raw_numpy)
  verify_result(T_target_source, gt_T_target_source)

if __name__ == "__main__":
  target_raw = small_gicp.read_ply(('data/target.ply'))  # Read the target point cloud (small_gicp.PointCloud)
  source_raw = small_gicp.read_ply(('data/source.ply'))  # Read the source point cloud (small_gicp.PointCloud)

  target_raw_numpy = target_raw.points()                    # Nx4 numpy array of the target point cloud
  source_raw_numpy = source_raw.points()                    # Nx4 numpy array of the source point cloud

  T_target_source = example_numpy1(target_raw_numpy, source_raw_numpy)
  T_target_source = example_numpy2(target_raw_numpy, source_raw_numpy)
  T_target_source = example_small1(target_raw_numpy, source_raw_numpy)
  T_target_source = example_small2(target_raw_numpy, source_raw_numpy)
