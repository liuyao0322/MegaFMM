#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
MEGAFMM_PROFILE_ALGORITHM="${MEGAFMM_PROFILE_ALGORITHM:-lt}"
MEGAFMM_PROFILE_SIZE="${MEGAFMM_PROFILE_SIZE:-8192}"
bash scripts/on_gpu.sh nsys profile --trace=cuda --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --stats=true \
  --force-overwrite=true -o "results/timeline_${MEGAFMM_PROFILE_ALGORITHM}" \
  ./build/megafmm_bench \
  --shapes "$MEGAFMM_PROFILE_SIZE,$MEGAFMM_PROFILE_SIZE,$MEGAFMM_PROFILE_SIZE" \
  --dtype bf16 --algorithm "$MEGAFMM_PROFILE_ALGORITHM" --profile \
  2>&1 | tee "results/timeline_${MEGAFMM_PROFILE_ALGORITHM}.txt"
