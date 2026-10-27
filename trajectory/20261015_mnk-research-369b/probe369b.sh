#!/bin/bash
set -euo pipefail
probe_failed=0
for variant in 0 1 2 3; do
 if nvcc -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -O3 -DNDEBUG --expt-relaxed-constexpr -std=c++17 -gencode arch=compute_90a,code=sm_90a -Iinclude -Isolution -Ithird_party/cutlass/include -DPROBE_VARIANT=$variant results/probe369b.cu -lcuda -o results/probe369b_$variant > results/probe369b_${variant}.log 2>&1; then
 if timeout -k 5 45 ./results/probe369b_$variant >> results/probe369b_${variant}.log 2>&1; then
  :
 else
  probe_status=$?; probe_failed=1; echo "variant=$variant preflight_exit=$probe_status" >> results/probe369b_${variant}.log
 fi
 else
  probe_status=$?; probe_failed=1; echo "variant=$variant compile_exit=$probe_status" >> results/probe369b_${variant}.log
 fi
 tail -4 results/probe369b_${variant}.log
done
exit "$probe_failed"
