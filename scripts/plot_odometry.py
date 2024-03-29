#!/usr/bin/python3
import os
import re
import numpy
from collections import namedtuple
from matplotlib import pyplot

Result = namedtuple('Result', ['reg_mean', 'reg_std', 'tp_mean', 'tp_std'])

def parse_result(filename):
  reg_mean = None
  reg_std = None
  throughput_mean = None
  throughput_std = None
  with open(filename, 'r') as f:
    for line in f.readlines():
      found = re.findall(r'([^=]+)\s*\+\-\s*(\S+)', line)
      if not found or len(found) != 2:
        found = re.findall(r'total_throughput=(\S+)', line)
        if found:
          throughput_mean = float(found[0])
        continue
      
      reg_mean = float(found[0][0].strip())
      reg_std = float(found[0][1].strip())
      throughput_mean = float(found[1][0].strip())
      throughput_std = float(found[1][1].strip())
    
  return Result(reg_mean, reg_std, throughput_mean, throughput_std)

def main():
  results_path = os.path.dirname(__file__) + '/results'

  results = {}
  for filename in os.listdir(results_path):
    found = re.findall(r'odometry_benchmark_(\S+)_(\d+).txt', filename)
    if not found:
      continue    
   
    rets = parse_result(results_path + '/' + filename)
    results['{}_{}'.format(found[0][0], found[0][1])] = rets
  
  fig, axes = pyplot.subplots(2, 2, figsize=(24, 12))
  
  num_threads = [1, 2, 4, 8, 16, 32, 64, 128]

  pcl_reg = results['pcl_1'].reg_mean
  pcl_tp = results['pcl_1'].tp_mean
  axes[0, 0].plot([num_threads[0], num_threads[-1]], [pcl_reg, pcl_reg], label='pcl_gicp', linestyle='--')
  axes[0, 1].plot([num_threads[0], num_threads[-1]], [pcl_tp, pcl_tp], label='pcl_gicp', linestyle='--')
  axes[1, 0].plot([num_threads[0], num_threads[-1]], [1.0, 1.0], label='pcl_gicp', linestyle='--')
  axes[1, 1].plot([num_threads[0], num_threads[-1]], [1.0, 1.0], label='pcl_gicp', linestyle='--')

  methods = ['fast_gicp', 'fast_vgicp', 'small_gicp_omp', 'small_gicp_tbb', 'small_vgicp_tbb', 'small_vgicp_omp']
  markers = ['o', 'o', '^', '^', 's', 's']

  for method, marker in zip(methods, markers):
    reg_means = [results['{}_{}'.format(method, N)].reg_mean for N in num_threads]
    axes[0, 0].plot(num_threads, reg_means, label=method, marker=marker)
    axes[1, 0].plot(num_threads, pcl_reg / numpy.array(reg_means), label=method, marker=marker)

  for method, marker in zip(methods, markers):
    tp_means = [results['{}_{}'.format(method, N)].tp_mean for N in num_threads]
    axes[0, 1].plot(num_threads, tp_means, label=method, marker=marker)
    axes[1, 1].plot(num_threads, pcl_tp / numpy.array(tp_means), label=method, marker=marker)
  flow_tp_means = [results['small_gicp_tbb_flow_{}'.format(N)].tp_mean for N in num_threads]
  axes[0, 1].plot(num_threads, flow_tp_means, label='small_gicp_tbb_flow', marker='*')
  axes[1, 1].plot(num_threads, pcl_tp / numpy.array(flow_tp_means), label='small_gicp_tbb_flow', marker='*')

  axes[0, 0].set_title('Net registration time (KdTree construction + cov estimation + pose estimation)')
  axes[1, 0].set_title('Net registration time (KdTree construction + cov estimation + pose estimation)')
  axes[0, 1].set_title('Total throughput (Downsampling + registration)')
  axes[1, 1].set_title('Total throughput (Downsampling + registration)')
  axes[0, 0].set_ylabel('Time [msec/scan]')
  axes[0, 1].set_ylabel('Time [msec/scan]')
  axes[1, 0].set_ylabel('Processing speed ratio (pcl_gicp=1.0)')
  axes[1, 1].set_ylabel('Processing speed ratio (pcl_gicp=1.0)')
  for i in range(2):
    for j in range(2):
      axes[i, j].set_xlabel('Number of threads = [1, 2, 4, ..., 128]')
      axes[i, j].set_xscale('log')
      axes[i, j].legend()
      axes[i, j].grid()
  
  fig.savefig('odometry_time.svg')
  pyplot.show()

if __name__ == "__main__":
  main()