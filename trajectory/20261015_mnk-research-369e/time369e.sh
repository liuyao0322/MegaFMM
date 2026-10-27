#!/bin/bash
set -euo pipefail
nvcc -O3 -DNDEBUG --expt-relaxed-constexpr -std=c++17 -gencode arch=compute_90a,code=sm_90a -Iinclude -Isolution results/time369e.cu build/libmegafmm.a -lcublasLt -lcublas -lcuda -o results/time369e > results/time369e_compile.log 2>&1
timeout -k 5 120 ./results/time369e --check > results/time369e_check.log 2>&1
cat results/time369e_check.log
for m in 4096 8192; do
 for dtype in bf16 fp16; do
  timeout -k 5 240 ./results/time369e "$m" "$dtype" > "results/time369e_${m}_${dtype}.log" 2>&1
  tail -2 "results/time369e_${m}_${dtype}.log"
 done
done
