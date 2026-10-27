#!/bin/bash
set -euo pipefail
export TMPDIR="$PWD/results/profile_iter97/ncu/tmp"
/share/app/cuda/cuda-12.8/bin/ncu --target-processes all --profile-from-start off --set full --kernel-name regex:sw_cooperative_hilo --launch-count 2 --clock-control none --cache-control none --force-overwrite -o results/profile_iter97/ncu/sw_full ./build/megafmm_bench --algorithm sw --shapes 1024,1024,1024 --dtype both --profile > results/profile_iter97/ncu/sw_full.log 2>&1
/share/app/cuda/cuda-12.8/bin/ncu --import results/profile_iter97/ncu/sw_full.ncu-rep --page raw --csv > results/profile_iter97/ncu/sw_raw.csv
/share/app/cuda/cuda-12.8/bin/ncu --import results/profile_iter97/ncu/sw_full.ncu-rep --page source --print-source sass --csv > results/profile_iter97/ncu/sw_sass.csv
