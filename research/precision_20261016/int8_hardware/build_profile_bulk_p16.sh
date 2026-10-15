#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/int8_hardware
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
 -Iinclude -Isolution -Ithird_party/cutlass/include -Xptxas=-v "$r/profile_bulk_p16.cu" \
 build/libmegafmm.a -lcublas -lcublasLt -lcuda -o "$r/bin/profile_bulk_p16"
