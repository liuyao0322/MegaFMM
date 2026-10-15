#!/bin/bash
set -euo pipefail
for variant in 0 1; do
 nvcc -O3 -DNDEBUG --expt-relaxed-constexpr -std=c++17 -gencode arch=compute_90a,code=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include -DPROBE_VARIANT=$variant results/probe366.cu -lcuda -o results/probe366_$variant > results/probe366_${variant}.log 2>&1
 timeout -k 5 45 ./results/probe366_$variant >> results/probe366_${variant}.log 2>&1
 tail -4 results/probe366_${variant}.log
done
