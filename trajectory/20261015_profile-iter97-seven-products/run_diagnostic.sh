#!/bin/bash
set -euo pipefail
nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_90a,code=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/profile_iter97/diagnostics/source/measure.cu -lcuda -o results/profile_iter97/diagnostics/measure -Xptxas=-v > results/profile_iter97/diagnostics/build.log 2>&1
./results/profile_iter97/diagnostics/measure results/profile_iter97/diagnostics 0 > results/profile_iter97/diagnostics/run.log 2>&1
