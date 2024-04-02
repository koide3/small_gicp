#!/usr/bin/python3
import re
import os
import numpy
from matplotlib import pyplot
from collections import namedtuple

def parse_result(filename):
  num_points = []
  results = {}
  
  with open(filename, 'r') as f:
    for line in f.readlines():
      matched = re.match(r'num_threads=(\d+)', line)
      if matched:
        num_threads = int(matched.group(1))
        continue
      
      matched = re.match(r'num_points=(\d+)', line)
      if matched:
        num_points.append(int(matched.group(1)))
        continue
      
      matched = re.match(r'(\S+)_times=(\S+) \+\- (\S+) \(median=(\S+)\)', line)
      if matched:
        method = matched.group(1)
        mean = float(matched.group(2))
        std = float(matched.group(3))
        med = float(matched.group(4))

        if method not in results:
          results[method] = []
        
        results[method].append(med)
        continue

  return (num_threads, num_points, results)

def main():
  results_path = os.path.dirname(__file__) + '/results'

  results = {}
  for filename in os.listdir(results_path):
    matched = re.match(r'kdtree_benchmark_(\S+)_(\d+).txt', filename)
    if not matched:
      continue

    method = '{}_{}'.format(matched.group(1), matched.group(2))
    print(method)

    num_threads, num_points, rets = parse_result(results_path + '/' + filename)
    results[method] = rets[list(rets.keys())[0]] 

  fig, axes = pyplot.subplots(1, 2, figsize=(12, 3))
  
  num_threads = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128]
  axes[0].plot(num_points, results['small_1'], label='kdtree (nanoflann)', marker='o', linestyle='--')
  for idx in [1, 3, 5, 7, 8]:
    N = num_threads[idx]
    axes[0].plot(num_points, results['omp_{}'.format(N)], label='kdtree_omp (%d threads)' % N, marker='s')
    axes[0].plot(num_points, results['tbb_{}'.format(N)], label='kdtree_tbb (%d threads)' % N, marker='^')

  baseline = numpy.array(results['small_1'])
  axes[1].plot([num_threads[0], num_threads[-1]], [1.0, 1.0], label='kdtree (nanoflann)', linestyle='--')
  for idx in [5]:
    threads = num_threads[idx]
    N = num_points[idx]
    
    axes[1].plot(num_threads, baseline[idx] / [results['omp_{}'.format(threads)][idx] for threads in num_threads], label='omp (num_points=%d)' % N, marker='s')
    axes[1].plot(num_threads, baseline[idx] / [results['tbb_{}'.format(threads)][idx] for threads in num_threads], label='tbb (num_points=%d)' % N, marker='^')

  axes[1].set_xscale('log')
   
  axes[0].set_xlabel('Number of points')
  axes[0].set_ylabel('KdTree construction time [msec/scan]')
  axes[1].set_xlabel('Number of threads = [1, 2, 4, ..., 128]')
  axes[1].set_ylabel('Processing speed ratio (single-thread = 1.0)')

  axes[0].grid()
  axes[1].grid()
  
  axes[0].legend()
  axes[1].legend()
  
  fig.savefig('kdtree_time.svg')
  
  pyplot.show()
  
  

if __name__ == '__main__':
  main()