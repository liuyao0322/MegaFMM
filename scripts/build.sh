#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
bash scripts/on_gpu.sh bash -c 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j "${SLURM_CPUS_PER_TASK:-4}"' 2>&1 | tee results/build.log
