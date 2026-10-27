#!/bin/bash
set -euo pipefail
nvcc -O3 -DNDEBUG --expt-relaxed-constexpr -std=c++17 -gencode arch=compute_90a,code=sm_90a -Iinclude -Isolution results/profile_pair369d.cu build/libmegafmm.a -lcublasLt -lcublas -lcuda -o results/profile_pair369d
mkdir -p results/mnk_ncu_tmp
TMPDIR="$PWD/results/mnk_ncu_tmp" /share/app/cuda/cuda-12.8/bin/ncu --set full --profile-from-start off --force-overwrite -o results/mnk369d_pair8192 ./results/profile_pair369d 61 8192 8192 8192 fp16 > results/mnk369d_pair8192_ncu.log 2>&1
