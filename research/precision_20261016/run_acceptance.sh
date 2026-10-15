#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
root=research/precision_20261016
# Serial runs only: never overlap analysis, profiling, compilation or timings.
"$root/compare_b128_w0" > "$root/direct_b128_w0_launch.log" 2>&1
"$root/validate_b128_w1" > "$root/extended_validation.log" 2>&1
for repeat in 0 1 2; do
  "$root/compare_b128_w1" > "$root/acceptance_launch_${repeat}.log" 2>&1
  "$root/compare_b128_w1" --graph > "$root/acceptance_graph_${repeat}.log" 2>&1
done
env -u LD_LIBRARY_PATH /share/home/liuyao/miniconda3/envs/kernelagent/bin/python "$root/scaling.py" > "$root/scaling.log" 2>&1
