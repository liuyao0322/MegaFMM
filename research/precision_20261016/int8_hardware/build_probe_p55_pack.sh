#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/int8_hardware
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
 -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -Iinclude -Isolution -Ithird_party/cutlass/include \
 -Xcompiler=-ffp-contract=off -Xptxas=-v "$r/probe_p55_pack.cu" \
 build/libmegafmm.a -lcublas -lcublasLt -lcuda -o "$r/bin/probe_p55_pack"
