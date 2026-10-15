#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
root=research/precision_20261016
for tool in racecheck synccheck; do
 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 \
  --kernel-name kns=direct_quadrants "$root/validate_b128_w1" > "$root/${tool}_scoped.log" 2>&1
done
env -u LD_LIBRARY_PATH /share/home/liuyao/miniconda3/envs/kernelagent/bin/python "$root/int8_model.py" > "$root/int8_model.log" 2>&1
