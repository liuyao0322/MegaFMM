#!/bin/bash
set -euo pipefail
nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_90a,code=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/opt_seven_round/bitwise100/check.cu -lcuda -o results/opt_seven_round/bitwise100/check -Xptxas=-v > results/opt_seven_round/bitwise100/build.log 2>&1
./results/opt_seven_round/bitwise100/check > results/opt_seven_round/bitwise100/run.log 2>&1
