#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/int8_hardware
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
 -Ithird_party/cutlass/include -Xptxas=-v "$r/pow2_compact_test.cu" -o "$r/bin/pow2_compact_p14"
"$r/bin/pow2_compact_p14"
