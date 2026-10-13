#!/usr/bin/env bash
# Full post-campaign verification against the already completed current build.
# Existing final_* evidence belongs to the earlier campaign and is untouched.
# Only strict numerical certification may return 2; structural/sanitizer/runtime
# failures abort with 1. A strict-gate miss remains explicit in its log and exit.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf 'Usage: bash scripts/verify_small.sh\n'
  printf 'Build first; run DAG, full structural/API tests, memcheck, all-candidate\n'
  printf 'racecheck/synccheck, and frozen strict numerics in the existing allocation.\n'
  printf 'Writes results/small_*.txt/.exit and small_verify_runs.csv.\n'
  printf 'MEGAFMM_SANITIZER may select a newer standalone compute-sanitizer.\n'
  exit 0
fi
if (( $# != 0 )); then
  printf 'verify_small.sh takes no selection arguments; it always runs the full verification.\n' >&2
  exit 1
fi
if [[ ! -x build/test_correctness ]]; then
  printf 'Missing executable build/test_correctness; build before verification.\n' >&2
  exit 1
fi
source_commit=$(git rev-parse HEAD)
binary_sha256=$(sha256sum build/test_correctness)
binary_sha256=${binary_sha256%% *}
mkdir -p results
metadata=results/small_verify_runs.csv
printf 'label,source_commit,binary_sha256,exit_code\n' > "$metadata"
for label in dag correctness memcheck racecheck synccheck strict_numerics; do
  : > "results/small_${label}.txt"
  : > "results/small_${label}.exit"
done
numeric_failure=0
sanitizer=${MEGAFMM_SANITIZER:-compute-sanitizer}
run_sanitizer() {
  local tool=$1
  shift
  printf 'SANITIZER_EXECUTABLE=%s\n' "$sanitizer"
  bash scripts/on_gpu.sh "$sanitizer" --version || return
  bash scripts/on_gpu.sh "$sanitizer" --tool "$tool" --error-exitcode 1 \
    ./build/test_correctness "$@"
}
run_check() {
  local label=$1 allow_numeric=$2
  shift 2
  local status current_sha
  current_sha=$(sha256sum build/test_correctness)
  current_sha=${current_sha%% *}
  if [[ "$current_sha" != "$binary_sha256" ]]; then
    printf 'Correctness binary changed during verification; abort before %s.\n' "$label" >&2
    exit 1
  fi
  set +e
  "$@" > "results/small_${label}.txt" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "results/small_${label}.exit"
  printf '%s,%s,%s,%s\n' "$label" "$source_commit" "$binary_sha256" "$status" >> "$metadata"
  current_sha=$(sha256sum build/test_correctness)
  current_sha=${current_sha%% *}
  if [[ "$current_sha" != "$binary_sha256" ]]; then
    printf 'Correctness binary changed while %s ran; invalidate this check and abort.\n' "$label" >&2
    exit 1
  fi
  printf '%s finished: exit=%s; results/small_%s.txt\n' "$label" "$status" "$label"
  if [[ "$status" == 2 && "$allow_numeric" == 1 ]]; then
    numeric_failure=2
    printf 'Frozen strict numerical gate failed; structural checks remain separate.\n'
  elif [[ "$status" != 0 ]]; then
    printf 'Verification/runtime failure for %s (actual exit=%s); see its log.\n' "$label" "$status" >&2
    exit 1
  fi
}

run_check dag 0 python3 tests/test_dag.py
run_check correctness 0 bash scripts/on_gpu.sh ./build/test_correctness
run_check memcheck 0 run_sanitizer memcheck
# No candidate filters: coverage follows the complete current test candidate list.
run_check racecheck 0 run_sanitizer racecheck --smoke-candidates
run_check synccheck 0 run_sanitizer synccheck --smoke-candidates
run_check strict_numerics 1 bash scripts/on_gpu.sh ./build/test_correctness --strict-numerics
exit "$numeric_failure"
