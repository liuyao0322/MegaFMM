#!/bin/bash
set -euo pipefail
PROFILE_OUTPUT="$1"
nvcc -O3 -std=c++17 -lineinfo -gencode=arch=compute_90a,code=sm_90a \
  -Iinclude -Isolution -Ithird_party/cutlass/include -Xptxas=-v \
  "$PROFILE_OUTPUT/source/measure.cu" -lcuda -o "$PROFILE_OUTPUT/measure" \
  > "$PROFILE_OUTPUT/build.log" 2>&1
"$PROFILE_OUTPUT/measure" "$PROFILE_OUTPUT" > "$PROFILE_OUTPUT/run.log" 2>&1
cuobjdump --dump-sass "$PROFILE_OUTPUT/measure" > "$PROFILE_OUTPUT/sass.txt"
