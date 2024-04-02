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
    found = re.findall(r'odometry_benchmark_(\S+)_(native|nonnative)_(\d+).txt', filename)
    if not found:
      continue    
   
    rets = parse_result(results_path + '/' + filename)
    results['{}_{}_{}'.format(found[0][0], found[0][1], found[0][2])] = rets
  
  fig, axes = pyplot.subplots(1, 1, figsize=(12, 2))
  axes = [axes]

  num_threads = [1, 2, 4, 8, 16, 32, 64, 128]

  print(results['small_gicp_native_1'], results['small_gicp_tbb_native_1'])
  print(results['small_gicp_nonnative_1'], results['small_gicp_tbb_nonnative_1'])
  
  native = [results['small_gicp_tbb_native_{}'.format(N)].reg_mean for N in num_threads]
  nonnative = [results['small_gicp_tbb_nonnative_{}'.format(N)].reg_mean for N in num_threads]

  axes[0].plot(num_threads, native, label='small_gicp_tbb (-march=native)', marker='o')
  axes[0].plot(num_threads, nonnative, label='small_gicp_tbb (nonnative)', marker='o')
  axes[0].set_xlabel('Number of threads [1, 2, ..., 128]')
  axes[0].set_ylabel('Time [msec/scan]')
  axes[0].set_xscale('log')
  axes[0].grid()
  axes[0].legend()
  
  pyplot.show()    
    

if __name__ == "__main__":
  main()