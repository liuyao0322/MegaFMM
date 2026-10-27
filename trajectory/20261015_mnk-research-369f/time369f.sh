#!/bin/bash
set -euo pipefail
nvcc -DRESEARCH_MASK="${RESEARCH_MASK:-63}" -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -O3 -DNDEBUG --expt-relaxed-constexpr -std=c++17 -gencode arch=compute_90a,code=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include results/time369f.cu build/libmegafmm.a -lcublasLt -lcublas -lcuda -o results/time369f > results/time369f_compile.log 2>&1
for m in 4096 8192; do
 for dtype in bf16 fp16; do
  timeout -k 5 180 ./results/time369f "$m" "$m" "$m" "$dtype" > "results/time369f_${m}_${dtype}.log" 2>&1
  tail -2 "results/time369f_${m}_${dtype}.log"
 done
done
