#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
common=(-O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -Iinclude -Isolution -Ithird_party/cutlass/include)
nvcc "${common[@]}" '-DDIRECT_HEADER="direct_b128_w1.cuh"' research/precision_20261016/compare_direct.cu build/libmegafmm.a -lcublas -lcublasLt -lcuda -o research/precision_20261016/compare_b128_w1
nvcc "${common[@]}" '-DDIRECT_HEADER="direct_b128_w0.cuh"' research/precision_20261016/compare_direct.cu build/libmegafmm.a -lcublas -lcublasLt -lcuda -o research/precision_20261016/compare_b128_w0
nvcc "${common[@]}" '-DDIRECT_HEADER="direct_b128_w1.cuh"' research/precision_20261016/validate_direct.cu build/libmegafmm.a -lcublas -lcublasLt -lcuda -o research/precision_20261016/validate_b128_w1
