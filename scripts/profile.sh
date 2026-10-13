#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Probe ncu once after baseline. Profiler replay times are never benchmark scores.
bash scripts/on_gpu.sh ncu --target-processes all --set basic \
  --kernel-name 'regex:pack_|combine_' --launch-count 3 \
  --force-overwrite -o results/baseline \
  ./build/megafmm_bench --shapes '4096,4096,4096' --dtype bf16 \
  --algorithm lt --profile 2>&1 | tee results/profile_baseline.txt
