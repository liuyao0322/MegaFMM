#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/fused_int8
mkdir -p "$r/bin"
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
 -Iinclude -Isolution -Ithird_party/cutlass/include -Xptxas=-v "$r/profile_compare.cu" \
 build/libmegafmm.a -lcublas -lcublasLt -lcuda -o "$r/bin/profile_compare"
