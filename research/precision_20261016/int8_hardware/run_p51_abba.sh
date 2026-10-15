#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/int8_hardware
# Four fresh processes, ordinary full calls and unchanged complete correctness.
"$r/bin/compare_p50" > "$r/p51_abba_a0.log" 2>&1
"$r/bin/compare_p51" > "$r/p51_abba_b0.log" 2>&1
"$r/bin/compare_p51" > "$r/p51_abba_b1.log" 2>&1
"$r/bin/compare_p50" > "$r/p51_abba_a1.log" 2>&1
