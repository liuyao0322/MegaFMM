#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
MEGAFMM_BUILD_DIR="${MEGAFMM_BUILD_DIR:-build}"
MEGAFMM_RESULT_DIR="${MEGAFMM_RESULT_DIR:-results}"
mkdir -p "$MEGAFMM_RESULT_DIR"
# Shared-filesystem timestamp caching can hide newer login-node source from make.
# Rebuild all objects so every recorded benchmark uses the current source.
bash scripts/on_gpu.sh bash -c '
  set -euo pipefail
  find solution -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum include/megafmm/gemm.h bench/benchmark.cu
  build_flags=(--clean-first)
  if [[ -n "${MEGAFMM_SW_REUSE_COMMIT:-}" ]]; then
    # A preceding full build must match this commit. Reuse only when every
    # common source/header is unchanged; force SW object removal to avoid NFS
    # timestamp ambiguity. All targets still build and link normally.
    git diff --quiet "$MEGAFMM_SW_REUSE_COMMIT" -- solution include bench tests CMakeLists.txt ":!solution/sw/hopper/sw_hopper.cu" ":!solution/sw/hopper/sw_inline_*.cuh" ":!solution/sw/hopper/sw_async_*.cuh"
    test -f "$1/CMakeFiles/megafmm.dir/solution/sw/hopper/sw_hopper.cu.o"
    cmake -E rm -f "$1/CMakeFiles/megafmm.dir/solution/sw/hopper/sw_hopper.cu.o"
    build_flags=()
  elif [[ -n "${MEGAFMM_TEST_REUSE_COMMIT:-}" ]]; then
    # Rebuild tests against an unchanged, identified full build.
    git diff --quiet "$MEGAFMM_TEST_REUSE_COMMIT" -- solution include bench CMakeLists.txt
    python3 - "$1" "$MEGAFMM_TEST_REUSE_COMMIT" <<VERIFY_BUILD
from pathlib import Path
import hashlib,json,sys
manifest=json.loads((Path(sys.argv[1])/"verified_build.json").read_text())
assert manifest["commit"]==sys.argv[2]
for group in ["source_sha256","artifact_sha256"]:
    for name,digest in manifest[group].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
VERIFY_BUILD
    cmake -E rm -f "$1"/CMakeFiles/test_*.dir/tests/*.o
    build_flags=()
  fi
  cmake -S . -B "$1" -DCMAKE_BUILD_TYPE=Release \
    -DMEGAFMM_FP32_CONFIG_HEADER="${MEGAFMM_FP32_CONFIG_HEADER:-}"
  cmake --build "$1" "${build_flags[@]}" -j "${SLURM_CPUS_PER_TASK:-4}"
' _ "$MEGAFMM_BUILD_DIR" 2>&1 | tee "$MEGAFMM_RESULT_DIR/build.log"
