#!/usr/bin/env bash
# Build first with scripts/build.sh. Numerical gate failures remain explicit.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
python3 tests/test_dag.py > results/final_dag.txt 2>&1
python3 tests/mixed_precision_floor.py > results/mixed_precision_witness.txt
bash scripts/on_gpu.sh ./build/test_correctness > results/final_correctness.txt 2>&1
printf 'Structural/API checks passed.\n'
bash scripts/on_gpu.sh compute-sanitizer --tool memcheck --error-exitcode 1 \
  ./build/test_correctness > results/final_memcheck.txt 2>&1
printf 'Full structural/API memcheck passed.\n'
for tool in racecheck synccheck; do
  bash scripts/on_gpu.sh compute-sanitizer --tool "$tool" --error-exitcode 1 \
    ./build/test_correctness --smoke-candidates > "results/final_${tool}.txt" 2>&1
  printf '%s on all kernel candidates passed.\n' "$tool"
done
set +e
bash scripts/on_gpu.sh ./build/test_correctness --strict-numerics \
  > results/final_strict_numerics.txt 2>&1
numeric_status=$?
set -e
printf '%s\n' "$numeric_status" > results/final_strict_numerics.exit
if [[ "$numeric_status" != 0 && "$numeric_status" != 2 ]]; then
  printf 'Strict numerical test encountered a runtime/test failure: exit %s\n' "$numeric_status" >&2
  exit "$numeric_status"
fi
printf 'Strict numerical gate exit=%s (2 means the frozen gate failed).\n' "$numeric_status"
exit "$numeric_status"
