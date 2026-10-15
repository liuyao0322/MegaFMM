#!/bin/bash
set -euo pipefail
work=research/falcon_layout_fusion/check332
nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include "$work/check.cu" -lcuda -o "$work/check" -Xptxas=-v > "$work/build.log" 2>&1
"$work/check" > "$work/stress.log" 2>&1
