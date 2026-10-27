#!/bin/bash
set -euo pipefail
export TMPDIR="$PWD/results/opt_seven_round/profile100/ncu/tmp"
/share/app/cuda/cuda-12.8/bin/ncu --target-processes all --profile-from-start off --set full --kernel-name regex:sw_cooperative_hilo --launch-count 2 --clock-control none --cache-control none --force-overwrite -o results/opt_seven_round/profile100/ncu/sw_full ./results/opt_seven_round/iter100_bench --algorithm sw --shapes 1024,1024,1024 --dtype both --profile > results/opt_seven_round/profile100/ncu/sw_full.log 2>&1
/share/app/cuda/cuda-12.8/bin/ncu --import results/opt_seven_round/profile100/ncu/sw_full.ncu-rep --page raw --csv > results/opt_seven_round/profile100/ncu/sw_raw.csv
/share/app/cuda/cuda-12.8/bin/ncu --import results/opt_seven_round/profile100/ncu/sw_full.ncu-rep --page source --print-source sass --csv > results/opt_seven_round/profile100/ncu/sw_sass.csv
nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_90a,code=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/opt_seven_round/profile100/diagnostics/source/measure.cu -lcuda -o results/opt_seven_round/profile100/diagnostics/measure -Xptxas=-v > results/opt_seven_round/profile100/diagnostics/build.log 2>&1
./results/opt_seven_round/profile100/diagnostics/measure results/opt_seven_round/profile100/diagnostics 0 > results/opt_seven_round/profile100/diagnostics/run.log 2>&1
