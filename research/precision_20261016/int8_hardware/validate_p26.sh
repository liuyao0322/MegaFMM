#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/int8_hardware
# Frozen P26 binaries. Ordinary/Graph compare uses the production tuned Dense gate.
"$r/bin/compare_p26" --graph > "$r/p26_graph.log" 2>&1
# Independent fresh seeds,7extended patterns,84cases/672interleaved calls.
# Stress keeps the historical cuBLAS GemmEx control; original FP64 AB unchanged.
"$r/bin/stress_p26" > "$r/stress_p26.log" 2>&1
# Scoped checks:28cases/56calls per tool. Instrumented timings are not benchmark data.
for tool in memcheck racecheck synccheck; do
 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 \
  --kernel-name kns=int8_research "$r/bin/stress_p26" --sanitizer > "$r/p26_${tool}.log" 2>&1
done
