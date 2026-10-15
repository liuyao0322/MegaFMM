#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/fused_int8
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
 -Iinclude -Ithird_party/cutlass/include -Xptxas=-v "$r/probe.cu" -lcuda -o "$r/bin/probe"
