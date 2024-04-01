#!/bin/bash
dataset_path=$1
exe_path=../build/kdtree_benchmark

mkdir results
num_threads=(1 2 3 4 5 6 7 8 16 32 64 128)

$exe_path $dataset_path --num_threads 1 --num_trials 1000 --method small | tee results/kdtree_benchmark_small_$N.txt

for N in ${num_threads[@]}; do
    sleep 1
    $exe_path $dataset_path --num_threads $N --num_trials 1000 --method tbb | tee results/kdtree_benchmark_tbb_$N.txt
done

for N in ${num_threads[@]}; do
    sleep 1
    $exe_path $dataset_path --num_threads $N --num_trials 1000 --method omp | tee results/kdtree_benchmark_omp_$N.txt
done
