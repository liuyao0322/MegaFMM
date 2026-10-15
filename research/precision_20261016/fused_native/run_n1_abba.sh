#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/fused_native
h=research/precision_20261016/int8_hardware
# Two independent complete-call ABBA modes; every process runs all78 gates.
for mode in ordinary graph; do
 args=()
 if [[ "$mode" == graph ]]; then args=(--graph); fi
 "$h/bin/compare_p47" "${args[@]}" > "$r/n1_${mode}_abba_a0.log" 2>&1
 "$r/bin/compare" "${args[@]}" > "$r/n1_${mode}_abba_b0.log" 2>&1
 "$r/bin/compare" "${args[@]}" > "$r/n1_${mode}_abba_b1.log" 2>&1
 "$h/bin/compare_p47" "${args[@]}" > "$r/n1_${mode}_abba_a1.log" 2>&1
done
