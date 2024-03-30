#!/bin/bash
dataset_path=$1
exe_path=../build/kdtree_benchmark

mkdir results
num_threads=(1 2 3 4 5 6 7 8 16 32 64 128)

for N in ${num_threads[@]}; do
    sleep 1
    echo $exe_path $dataset_path --num_threads $N | tee results/kdtree_benchmark_$N.txt
    $exe_path $dataset_path --num_threads $N --num_trials 1000 | tee results/kdtree_benchmark_$N.txt
done
