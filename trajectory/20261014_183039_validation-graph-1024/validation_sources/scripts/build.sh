#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
# Shared-filesystem timestamp caching can hide newer login-node source from make.
# Rebuild all objects so every recorded benchmark uses the current source.
bash scripts/on_gpu.sh bash -c 'sha256sum solution/* include/megafmm/gemm.h bench/benchmark.cu && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --clean-first -j "${SLURM_CPUS_PER_TASK:-4}"' 2>&1 | tee results/build.log
