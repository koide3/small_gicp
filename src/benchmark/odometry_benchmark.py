#!/usr/bin/python3
NUM_THREADS = 4

import os
os.environ['OMP_NUM_THREADS'] = '{}'.format(NUM_THREADS)

import time
import numpy
import pathos
import small_gicp
from tqdm import tqdm
from pyridescence import *


class OdometryEstimation(object):
  def __init__(self):
    self.name = 'N/A'
    self.times = []

  def clear_times(self):
    self.times = []

  def start_timer(self):
    self.t1 = time.time()
  
  def stop_timer(self):
    self.times.append((time.time() - self.t1) * 1e3)
  
  def method_name(self):
    return self.name
  
  def report(self):
    if not len(self.times):
      return 'N/A'
    
    mean = numpy.mean(self.times)
    std = numpy.std(self.times)
    median = numpy.median(self.times)
    return 'mean={:.3f} std={:.3f} median={:.3f} last={:.3f} [msec]'.format(mean, std, median, self.times[-1])

  def estimate(self, points):
    self.start_timer()
    T = self.estimate_impl(points)
    self.stop_timer()
    
    return T
  

def create_small_gicp(num_threads):
  class OdometryEstimationSmallGICP(OdometryEstimation):
    def __init__(self, num_threads):
      super().__init__()
      self.name = 'small_gicp'
      self.num_threads = num_threads
      
      self.last_frame = None
      self.T_world_lidar = numpy.identity(4)
      
    def estimate_impl(self, points):
      current_frame = small_gicp.PointCloud(points)
      tree = small_gicp.KdTree(current_frame, self.num_threads)
      small_gicp.estimate_covariances(current_frame, tree, num_threads=self.num_threads)
      
      if not self.last_frame:
        self.last_frame = (current_frame, tree)
        return self.T_world_lidar

      result = small_gicp.align(self.last_frame[0], current_frame, self.last_frame[1], num_threads=self.num_threads)
      
      self.T_world_lidar = self.T_world_lidar @ result.T_target_source
      self.last_frame = (current_frame, tree)

      return self.T_world_lidar
  
  return OdometryEstimationSmallGICP(num_threads)


def create_open3d_gicp():
  import open3d
 
  class OdometryEstimationOpen3dGICP(OdometryEstimation):
    def __init__(self):
      super().__init__()
      self.name = 'open3d_gicp'
      self.last_frame = None
      self.T_world_lidar = numpy.identity(4)
      
    def estimate_impl(self, points):
      current_frame = open3d.geometry.PointCloud()
      current_frame.points = open3d.utility.Vector3dVector(points)
      current_frame.estimate_covariances()
          
      if not self.last_frame:
        self.last_frame = current_frame
        return self.T_world_lidar

      result = open3d.pipelines.registration.registration_generalized_icp(current_frame, self.last_frame, 1.0)
      # result = open3d.pipelines.registration.registration_icp(
      #   current_frame, self.last_frame, 1.0, numpy.identity(4),
      #   open3d.pipelines.registration.TransformationEstimationPointToPoint()
      # )
      
      self.T_world_lidar = self.T_world_lidar @ result.transformation
      self.last_frame = current_frame

      return self.T_world_lidar
  
  return OdometryEstimationOpen3dGICP()


def create_probreg_cpd():
  import open3d
  import probreg as cpd
 
  class OdometryEstimationProbreg(OdometryEstimation):
    def __init__(self):
      super().__init__()
      self.name = 'probreg'
      self.last_frame = None
      self.T_world_lidar = numpy.identity(4)
      
    def estimate_impl(self, points):
      current_frame = open3d.geometry.PointCloud()
      current_frame.points = open3d.utility.Vector3dVector(points)
          
      if not self.last_frame:
        self.last_frame = current_frame
        return self.T_world_lidar

      result, _, _ = cpd.registration_cpd(current_frame, self.last_frame)
      print(result)
      # result = open3d.pipelines.registration.registration_generalized_icp(current_frame, self.last_frame, 1.0)
      # result = open3d.pipelines.registration.registration_icp(
      #   current_frame, self.last_frame, 1.0, numpy.identity(4),
      #   open3d.pipelines.registration.TransformationEstimationPointToPoint()
      # )
      
      self.T_world_lidar = self.T_world_lidar @ result.transformation
      self.last_frame = current_frame

      return self.T_world_lidar
  
  return OdometryEstimationProbreg()


def create_simpleicp():
  import pandas
  import simpleicp
 
  class OdometryEstimationSimpleICP(OdometryEstimation):
    def __init__(self):
      super().__init__()
      self.name = 'simple_icp'
      self.last_frame = None
      self.T_world_lidar = numpy.identity(4)
      
    def estimate_impl(self, points):
      points = pandas.DataFrame(points, columns=['x', 'y', 'z'])
      current_frame = simpleicp.PointCloud(points)
          
      if self.last_frame is None:
        self.last_frame = current_frame
        return self.T_world_lidar

      icp = simpleicp.SimpleICP()
      icp.add_point_clouds(self.last_frame, current_frame)
      H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1)
      
      self.T_world_lidar = self.T_world_lidar @ H
      self.last_frame = current_frame

      return self.T_world_lidar
  
  return OdometryEstimationSimpleICP()


def main():
  dataset_path = '/home/koide/datasets/kitti/velodyne_filtered'
  filenames = sorted([dataset_path + '/' + x for x in os.listdir(dataset_path) if x.endswith('.bin')])[:500]
 
  print('Loading and downsampling points')
  def preprocess(filename):
    points = numpy.fromfile(filename, dtype=numpy.float32).reshape(-1, 4)
    downsampled = small_gicp.voxelgrid_sampling(small_gicp.PointCloud(points[:, :3]), 0.25)
    return downsampled.points()[:, :3]

  with pathos.multiprocessing.ProcessingPool() as p:
    all_points = p.map(preprocess, filenames)
  
  odom = create_open3d_gicp()
  # odom = create_small_gicp(num_threads=NUM_THREADS)
  # odom = create_probreg_cpd()
  
  viewer = guik.viewer()
  viewer.spin_until_click()
  
  viewer.append_text('method={} ({} threads)'.format(odom.method_name(), NUM_THREADS))
  viewer.disable_vsync()
  viewer.use_orbit_camera_control(150.0)
  for i, points in enumerate(all_points):
    if i == 50:
      print('Warminup done')
      odom.clear_times()
    elif i % 10 == 0:
      report = odom.report()
      print(report)
      viewer.append_text(report)

    T_world_lidar = odom.estimate(points)
    
    cloud_buffer = glk.create_pointcloud_buffer(points)
    viewer.update_drawable('current', cloud_buffer, guik.FlatOrange(T_world_lidar).add('point_scale', 2.0))
    viewer.update_drawable('coord', glk.primitives.coordinate_system(), guik.VertexColor(T_world_lidar))
    viewer.update_drawable(guik.anon(), cloud_buffer, guik.Rainbow(T_world_lidar))
    viewer.lookat(T_world_lidar[:3, 3])
    if not viewer.spin_once():
      break
       
  
  

if __name__ == '__main__':
  main()
