#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/int8_hardware
# Four fresh processes, Graph replay calls and unchanged complete correctness.
"$r/bin/compare_p53" --graph > "$r/p54_shape_graph_abba_a0.log" 2>&1
"$r/bin/compare_p54" --graph > "$r/p54_shape_graph_abba_b0.log" 2>&1
"$r/bin/compare_p54" --graph > "$r/p54_shape_graph_abba_b1.log" 2>&1
"$r/bin/compare_p53" --graph > "$r/p54_shape_graph_abba_a1.log" 2>&1
