#!/usr/bin/python3
import os
import time
import argparse
import collections
import numpy
import small_gicp
from pyridescence import *

# Odometry estimation based on scan-to-scan matching
class ScanToScanMatchingOdometry(object):
  def __init__(self, num_threads):
    self.num_threads = num_threads
    self.T_last_current = numpy.identity(4)
    self.T_world_lidar = numpy.identity(4)
    self.target = None
  
  def estimate(self, raw_points):
    downsampled, tree = small_gicp.preprocess_points(raw_points, 0.25, num_threads=self.num_threads)
    
    if self.target is None:
      self.target = (downsampled, tree)
      return self.T_world_lidar

    result = small_gicp.align(self.target[0], downsampled, self.target[1], self.T_last_current, num_threads=self.num_threads)
    
    self.T_last_current = result.T_target_source
    self.T_world_lidar = self.T_world_lidar @ result.T_target_source
    self.target = (downsampled, tree)
    
    return self.T_world_lidar
    
# Odometry estimation based on scan-to-model matching
class ScanToModelMatchingOdometry(object):
  def __init__(self, num_threads):
    self.num_threads = num_threads
    self.T_last_current = numpy.identity(4)
    self.T_world_lidar = numpy.identity(4)
    self.target = small_gicp.GaussianVoxelMap(1.0)
    self.target.set_lru(horizon=100, clear_cycle=10)
  
  def estimate(self, raw_points):
    downsampled, tree = small_gicp.preprocess_points(raw_points, 0.25, num_threads=self.num_threads)
    
    if self.target.size() == 0:
      self.target.insert(downsampled)
      return self.T_world_lidar
    
    result = small_gicp.align(self.target, downsampled, self.T_world_lidar @ self.T_last_current, num_threads=self.num_threads)

    self.T_last_current = numpy.linalg.inv(self.T_world_lidar) @ result.T_target_source
    self.T_world_lidar = result.T_target_source
    self.target.insert(downsampled, self.T_world_lidar)
    
    guik.viewer().update_drawable('target', glk.create_pointcloud_buffer(self.target.voxel_points()[:, :3]), guik.Rainbow())
    
    return self.T_world_lidar
  

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset_path', help='/path/to/kitti/velodyne')
  parser.add_argument('--num_threads', help='Number of threads', type=int, default=4)
  parser.add_argument('-m', '--model', help='Use scan-to-model matching odometry', action='store_true')
  args = parser.parse_args()
  
  dataset_path = args.dataset_path
  filenames = sorted([dataset_path + '/' + x for x in os.listdir(dataset_path) if x.endswith('.bin')])
  
  if not args.model:
    odom = ScanToScanMatchingOdometry(args.num_threads)
  else:
    odom = ScanToModelMatchingOdometry(args.num_threads)
  
  viewer = guik.viewer()
  viewer.disable_vsync()
  time_queue = collections.deque(maxlen=500)

  for i, filename in enumerate(filenames):
    raw_points = numpy.fromfile(filename, dtype=numpy.float32).reshape(-1, 4)[:, :3]
    
    t1 = time.time()
    T = odom.estimate(raw_points)
    t2 = time.time()
   
    time_queue.append(t2 - t1)
    viewer.lookat(T[:3, 3])
    viewer.update_drawable('points', glk.create_pointcloud_buffer(raw_points), guik.FlatOrange(T).add('point_scale', 2.0))
    
    if i % 10 == 0:
      viewer.update_drawable('pos_{}'.format(i), glk.primitives.coordinate_system(), guik.VertexColor(T))
      viewer.append_text('avg={:.3f} msec/scan  last={:.3f} msec/scan'.format(1000 * numpy.mean(time_queue), 1000 * time_queue[-1]))
    
    if not viewer.spin_once():
      break    

if __name__ == '__main__':
  main()
