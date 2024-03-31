#!/usr/bin/python3
import numpy
from scipy.spatial.transform import Rotation

import small_gicp
from pyridescence import *

# Verity the estimated transformation matrix (for testing)
def verify_result(T_target_source, gt_T_target_source):
  error = numpy.linalg.inv(T_target_source) @ gt_T_target_source
  error_trans = numpy.linalg.norm(error[:3, 3])
  error_rot = Rotation.from_matrix(error[:3, :3]).magnitude()

  if error_trans > 0.1 or error_rot > 0.1:
    print('error_trans={:.4f}, error_rot={:.4f}'.format(error_trans, error_rot))
    exit(1)


# Basic registation example with numpy arrays
def example1(target_raw_numpy : numpy.ndarray, source_raw_numpy : numpy.ndarray, gt_T_target_source : numpy.ndarray):
  # Example A : Perform registration with numpy arrays
  # Arguments
  # - target_points               : Nx4 or Nx3 numpy array of the target point cloud
  # - source_points               : Nx4 or Nx3 numpy array of the source point cloud
  # Optional arguments
  # - init_T_target_source        : Initial guess of the transformation matrix (4x4 numpy array)
  # - registration_type           : Registration type ("ICP", "PLANE_ICP", "GICP", "VGICP")
  # - voxel_resolution            : Voxel resolution for VGICP
  # - max_correspondence_distance : Maximum correspondence distance
  # - max_iterations              : Maximum number of iterations
  result = small_gicp.align_points(target_raw_numpy, source_raw_numpy)

  # Verity the estimated transformation matrix
  verify_result(result.T_target_source, gt_T_target_source)


  # Example B : Perform preprocessing and registration separately
  # Preprocess point clouds
  # Arguments
  # - points_numpy                : Nx4 or Nx3 numpy array of the target point cloud
  # Optional arguments
  # - downsampling_resolution     : Downsampling resolution
  # - num_neighbors               : Number of neighbors for normal and covariance estimation
  # - num_threads                 : Number of threads
  target, target_tree = small_gicp.preprocess_points(points_numpy=target_raw_numpy, downsampling_resolution=0.25)
  source, source_tree = small_gicp.preprocess_points(points_numpy=source_raw_numpy, downsampling_resolution=0.25)

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
  
  # Verity the estimated transformation matrix
  verify_result(result.T_target_source, gt_T_target_source)


# Basic registation example with small_gicp.PointCloud
def example2(target_raw_numpy : numpy.ndarray, source_raw_numpy : numpy.ndarray, gt_T_target_source : numpy.ndarray):
  # Convert numpy arrays to small_gicp.PointCloud
  target_raw = small_gicp.PointCloud(target_raw_numpy)
  source_raw = small_gicp.PointCloud(source_raw_numpy)
  pass


def main():
  gt_T_target_source = numpy.loadtxt('../data/T_target_source.txt')  # Load the ground truth transformation matrix
  print('--- gt_T_target_source ---')
  print(gt_T_target_source)

  target_raw = small_gicp.read_ply(('../data/target.ply'))  # Read the target point cloud (small_gicp.PointCloud)
  source_raw = small_gicp.read_ply(('../data/source.ply'))  # Read the source point cloud (small_gicp.PointCloud)

  target_raw_numpy = target_raw.points()                    # Nx4 numpy array of the target point cloud
  source_raw_numpy = source_raw.points()                    # Nx4 numpy array of the source point cloud

  example1(target_raw_numpy, source_raw_numpy, gt_T_target_source)
  example2(target_raw_numpy, source_raw_numpy, gt_T_target_source)
  return


if __name__ == "__main__":
  main()
