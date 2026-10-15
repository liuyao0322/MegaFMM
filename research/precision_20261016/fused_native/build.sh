#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
root=research/precision_20261016/fused_native
mkdir -p "$root/bin"
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
 -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -Iinclude -Isolution -Ithird_party/cutlass/include \
 -Xptxas=-v "$root/compare.cu" build/libmegafmm.a -lcublas -lcublasLt -lcuda -o "$root/bin/compare"
