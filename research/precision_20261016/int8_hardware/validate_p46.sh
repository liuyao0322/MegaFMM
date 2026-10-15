#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/int8_hardware
# Frozen P46 binaries. Ordinary/Graph compare uses the production tuned Dense gate.
cp "$r/p46_graph_abba_b0.log" "$r/p46_graph.log"
# Independent fresh seeds,9extended patterns,108cases/864interleaved calls.
# Stress keeps the historical cuBLAS GemmEx control; original FP64 AB unchanged.
"$r/bin/stress_p46" > "$r/stress_p46.log" 2>&1
# Scoped checks:36cases/72calls per tool. Instrumented timings are not benchmark data.
for tool in memcheck racecheck synccheck; do
 /share/app/cuda/cuda-12.8/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 \
  --kernel-name kns=int8_research "$r/bin/stress_p46" --sanitizer > "$r/p46_${tool}.log" 2>&1
done
