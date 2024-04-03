#!/usr/bin/python3
import re
import os
import pathos
import subprocess
from collections import namedtuple
from matplotlib import pyplot

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
  
  methods = ['pcl_1', 'fast_gicp_128', 'fast_vgicp_128', 'small_gicp_1', 'small_gicp_tbb_128', 'small_gicp_omp_128', 'small_vgicp_tbb_128']
  labels = ['pcl_gicp', 'fast_gicp', 'fast_vgicp', 'small_gicp', 'small_gicp (tbb)', 'small_gicp (omp)', 'small_vgicp']
  
  Result = namedtuple('Result', ['ape', 'rpe100', 'rpe400', 'rpe800'])
  def evaluate(inputs):
    method, filename = inputs
    print('.', end='', flush=True)
    ape = eval_ape(gt_path, filename)
    rpe100 = eval_rpe(gt_path, filename, delta=100)
    rpe400 = eval_rpe(gt_path, filename, delta=400)
    rpe800 = eval_rpe(gt_path, filename, delta=800)
    return method, Result(ape, rpe100, rpe400, rpe800)

  print('evaluating')
  with pathos.multiprocessing.ProcessingPool() as p:
    errors = p.map(evaluate, [(method, filename) for method, filename in filenames if method in methods])
  print()
  
  results = {}
  for method, error in errors:
    results[method] = error
    
  for method, label in zip(methods, labels):
    ape, rpe100, rpe400, rpe800 = results[method]
    print('{:20s} : APE={:.3f} +- {:.3f}  RPE(100)={:.3f} +- {:.3f}  RPE(400)={:.3f} +- {:.3f}  RPE(800)={:.3f} +- {:.3f}'
          .format(label, ape['rmse'], ape['std'], rpe100['rmse'], rpe100['std'], rpe400['rmse'], rpe400['std'], rpe800['rmse'], rpe800['std']))
  
  fig, axes = pyplot.subplots(1, 3, figsize=(15, 5))
  apes = [results[method].ape['rmse'] for method in methods]
  axes[0].bar(labels, apes)
  
  pyplot.show()
  
    
  
  

if __name__ == '__main__':
  main()
