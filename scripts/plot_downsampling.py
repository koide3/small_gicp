#!/usr/bin/python3
import re
import os
import numpy
from collections import namedtuple
from matplotlib import pyplot

Result = namedtuple('Result', ['time_mean', 'time_std', 'points_mean', 'points_std'])

def parse_result(filename):
  leaf_sizes = []
  results = {}
  
  with open(filename, 'r') as f:
    for line in f.readlines():
      if '(warmup)' in line:
        continue
      
      found = re.findall(r'leaf_size=(\S+)', line)
      if found:
        leaf_sizes.append(float(found[0]))
        
      if len(leaf_sizes) == 0:
        continue
      
      stat = r'\s*(\S+)\s*\+\-\s*(\S+)\s*'
      found = re.findall(stat, line)
      if len(found) != 2:
        continue
      
      name = line.split(':')[0].strip()
      if name not in results:
        results[name] = []
      
      results[name].append(Result(float(found[0][0]), float(found[0][1]), float(found[1][0]), float(found[1][1])))
  
  return leaf_sizes, results
        

def main():
  results_path = os.path.dirname(__file__) + '/results'
  
  leaf_sizes = None
  raw_results = []
  for filename in os.listdir(results_path):
    found = re.findall(r'downsampling_benchmark_(\d+).txt', filename)    
    if not found:
      continue
    
    leaf_sizes, rets = parse_result(results_path + '/' + filename)
    raw_results.append((int(found[0]), rets))
  
  raw_results = sorted(raw_results, key=lambda x: x[0])

  def summarize(rets):
    time_mean = numpy.array([x.time_mean for x in rets])
    time_std = numpy.array([x.time_std for x in rets])
    points_mean = numpy.array([x.points_mean for x in rets])
    points_std = numpy.array([x.points_std for x in rets])
    return Result(time_mean, time_std, points_mean, points_std)

  results = {}
  results['pcl_voxelgrid'] = summarize(raw_results[0][1]['pcl_voxelgrid'])
  results['pcl_approx_voxelgrid'] = summarize(raw_results[0][1]['pcl_approx_voxelgrid'])
  results['small_voxelgrid'] = summarize(raw_results[0][1]['small_voxelgrid'])
  for num_threads, rets in raw_results:
    results['small_voxelgrid_tbb ({} threads)'.format(num_threads)] = summarize(rets['small_voxelgrid_tbb'])
    results['small_voxelgrid_omp ({} threads)'.format(num_threads)] = summarize(rets['small_voxelgrid_omp'])
  
  fig, axes = pyplot.subplots(1, 5)
  fig.set_size_inches(18, 3)

  leaf_size_indices = [0, 1, 4, 9, -1]
  for i, leaf_size_index in enumerate(leaf_size_indices):
    leaf_size = leaf_sizes[leaf_size_index]

    num_threads = [x[0] for x in raw_results]
    time_tbb = []
    time_omp = []
    for N in [x[0] for x in raw_results]:
      time_tbb.append(results['small_voxelgrid_tbb ({} threads)'.format(N)].time_mean[leaf_size_index])
      time_omp.append(results['small_voxelgrid_omp ({} threads)'.format(N)].time_mean[leaf_size_index])
    
    baseline = results['pcl_voxelgrid'].time_mean[leaf_size_index]
    axes[i].plot([num_threads[0], num_threads[-1]], [baseline, baseline], label='pcl_voxelgrid', linestyle='--')
    axes[i].scatter(num_threads, time_tbb, label='small_voxelgrid_tbb', marker='^')
    axes[i].scatter(num_threads, time_omp, label='small_voxelgrid_omp', marker='^')
    
    axes[i].grid()
    axes[i].set_xscale('log')
    axes[i].set_xlabel('Num threads')
    axes[i].set_title('Leaf size = {} m'.format(leaf_size))
  
  axes[0].set_ylabel('Processing time [msec/scan]')
  axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=3)

  fig.savefig('downsampling_threads.svg')

  
  fig, axes = pyplot.subplots(1, 2)
  fig.set_size_inches(18, 3)

  methods = ['small_voxelgrid', 'small_voxelgrid_tbb (2 threads)', 'small_voxelgrid_tbb (3 threads)', 'small_voxelgrid_tbb (4 threads)', 'small_voxelgrid_tbb (6 threads)', 'pcl_voxelgrid', 'pcl_approx_voxelgrid']
  labels = ['small_voxelgrid', 'small_voxelgrid_tbb (2 threads)', 'small_voxelgrid_tbb (3 threads)', 'small_voxelgrid_tbb (4 threads)', 'small_voxelgrid_tbb (6 threads)', 'pcl_voxelgrid', 'pcl_approx_voxelgrid']
  markers = ['^', '^', '^', '^', '^', 'o', 'o']
  for method, label, marker in zip(methods, labels, markers):
    axes[0].plot(leaf_sizes, results[method].time_mean, label=label, linestyle='--' if 'pcl' in method else '-', marker=marker)
    
    if 'threads' not in method or '2 threads' in method:
      axes[1].plot(leaf_sizes, results[method].points_mean / results['pcl_voxelgrid'].points_mean, label=label, linestyle='--' if 'pcl' in method else '-', marker=marker)
  
  axes[0].set_xlabel('Leaf size [m]')
  axes[0].set_ylabel('Processing time [msec/scan]')
  axes[1].set_xlabel('Leaf size [m]')
  axes[1].set_ylabel('Num points ratio to pcl_voxelgrid')
  axes[0].legend()
  axes[1].legend()
  axes[0].grid()
  axes[1].grid()
  
  fig.savefig('downsampling_comp.svg')
  
  pyplot.show()


if __name__ == '__main__':
  main()
