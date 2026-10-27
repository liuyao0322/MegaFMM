#!/bin/bash
set -euo pipefail
for variant in 0 1 2 3; do
 nvcc -O3 -DNDEBUG --expt-relaxed-constexpr -std=c++17 -gencode arch=compute_90a,code=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include -DPROBE_VARIANT=$variant results/probe368.cu -lcuda -o results/probe368_$variant > results/probe368_${variant}.log 2>&1
 timeout -k 5 45 ./results/probe368_$variant >> results/probe368_${variant}.log 2>&1
 tail -4 results/probe368_${variant}.log
done
