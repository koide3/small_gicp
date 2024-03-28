#!/usr/bin/python3
import re
import os
import pathos
import subprocess
from collections import namedtuple

def run_evo(commands):
  p = subprocess.Popen(commands, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  p.wait()
  stdout, stderr = p.communicate()

  if len(stderr):
    print(stderr.decode('utf-8'))

  result = stdout.decode('utf-8')
  results = {}
  for item in re.findall(r'([a-z]+)\s+([0-9]+\.[0-9]+)', result):
    results[item[0]] = float(item[1])

  return results


def eval_ape(gt_filename, traj_filename, t_offset=0.0):
  ret = run_evo(['evo_ape', 'kitti', gt_filename, traj_filename, '-a'])
  return ret


def eval_rpe(gt_filename, traj_filename, delta_unit='m', delta=100, all_pairs=True, t_offset=0.0):
  commands = ['evo_rpe', 'kitti', gt_filename, traj_filename, '-a', '--delta_unit', str(delta_unit), '--delta', str(delta)]
  if all_pairs:
    commands += ['--all_pairs']

  ret = run_evo(commands)
  return ret


def main():
  gt_path = '/home/koide/datasets/ssd/kitti/poses/00_lidar.txt'

  results_path = os.path.dirname(__file__) + '/results'
  
  filenames = []
  for filename in os.listdir(results_path):
    found = re.findall(r'traj_lidar_(\S+)_(\d+).txt', filename)
    if not found:
      continue
    
    method = found[0][0] + '_' + found[0][1]
    filenames.append((method, results_path + '/' + filename))
  
  Result = namedtuple('Result', ['ape', 'rpe'])
  def evaluate(filename):
    ape = eval_ape(gt_path, filename)
    rpe = eval_rpe(gt_path, filename, delta=100)
    return Result(ape, rpe)

  print('evaluating')
  with pathos.multiprocessing.ProcessingPool() as p:
    errors = p.map(evaluate, [filename for method, filename in filenames])
  
  results = {}
  for (method, filename), error in zip(filenames, errors):
    results[method] = error
  
  methods = ['pcl_1', 'fast_gicp_1', 'fast_vgicp_1', 'small_gicp_1', 'small_gicp_tbb_1', 'small_gicp_omp_1', 'small_vgicp_tbb_1']
  labels = ['pcl_gicp', 'fast_gicp', 'fast_vgicp', 'small_gicp', 'small_gicp (tbb)', 'small_gicp (omp)', 'small_vgicp']
  
  for method, label in zip(methods, labels):
    ape, rpe = results[method]
    print('{:20s} : APE {:.3f} +- {:.3f}  RPE {:.3f} +- {:.3f}'.format(label, ape['rmse'], ape['std'], rpe['rmse'], rpe['std']))
  
    
  
  

if __name__ == '__main__':
  main()
