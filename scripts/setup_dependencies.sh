#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${SLURM_STEP_ID:-}" ]]; then
  echo 'Stage dependencies on the login node before GPU work.' >&2
  exit 1
fi
MEGAFMM_CUTLASS_COMMIT=f7b19de32c5d1f3cedfc735c2849f12b537522ee
if [[ ! -d third_party/cutlass/.git ]]; then
  mkdir -p third_party
  GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch v3.5.1 \
    https://gitcode.com/GitHub_Trending/cu/cutlass.git third_party/cutlass
fi
test "$(git -C third_party/cutlass rev-parse HEAD)" = "$MEGAFMM_CUTLASS_COMMIT"
git -C third_party/cutlass diff --exit-code
printf 'CUTLASS v3.5.1 %s\n' "$MEGAFMM_CUTLASS_COMMIT"
