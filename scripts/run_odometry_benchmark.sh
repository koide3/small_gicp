#!/bin/bash
dataset_path=$1
exe_path=../build/odometry_benchmark

mkdir results

engines=(pcl fast_gicp fast_vgicp small_gicp small_gicp_tbb small_gicp_omp)
for engine in ${engines[@]}; do
  N=1
  $exe_path $dataset_path $(printf "results/traj_lidar_%s_%d.txt" $engine $N) --num_threads $N --engine $engine | tee $(printf "results/odometry_benchmark_%s_%d.txt" $engine $N)
done

engines=(fast_gicp fast_vgicp small_gicp_omp small_gicp_tbb small_vgicp_omp small_vgicp_tbb small_gicp_tbb_flow)
num_threads=(128 96 64 32 16 8 4 2)

for N in ${num_threads[@]}; do
  for engine in ${engines[@]}; do
    $exe_path $dataset_path $(printf "results/traj_lidar_%s_%d.txt" $engine $N) --num_threads $N --engine $engine | tee $(printf "results/odometry_benchmark_%s_%d.txt" $engine $N)
  done
done
