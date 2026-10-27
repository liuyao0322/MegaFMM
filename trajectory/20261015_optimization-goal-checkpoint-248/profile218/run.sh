#!/bin/bash
set -eu
export TMPDIR="$PWD/results/opt_goal/profile218/tmp"
NCU=/share/app/cuda/cuda-12.8/bin/ncu
$NCU --target-processes all --profile-from-start off --set full --kernel-name regex:sw_cooperative_hilo --launch-count 2 --clock-control none --cache-control none --force-overwrite -o results/opt_goal/profile218/full results/opt_seven_round/iter218_bench --algorithm sw --shapes 1024,1024,1024 --dtype both --profile > results/opt_goal/profile218/full.log 2>&1
$NCU --import results/opt_goal/profile218/full.ncu-rep --page raw --csv > results/opt_goal/profile218/raw.csv
$NCU --import results/opt_goal/profile218/full.ncu-rep --page source --print-source sass --csv > results/opt_goal/profile218/sass.csv
