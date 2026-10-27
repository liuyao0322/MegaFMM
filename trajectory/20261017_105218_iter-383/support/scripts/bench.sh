#!/bin/bash
# Benchmark wrapper with trajectory tracking
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."
RESULTS_DIR="${MEGAFMM_RESULT_DIR:-results}"
BUILD_DIR="${MEGAFMM_BUILD_DIR:-build}"
mkdir -p "$RESULTS_DIR"

MODE_ARGS=()
[[ "${MEGAFMM_GRAPH:-0}" == 1 ]] && MODE_ARGS+=(--graph)
LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

sw_precheck() {
  if [[ "${MEGAFMM_SW_PRECHECK:-0}" == 1 ]]; then
    bash scripts/on_gpu.sh "$BUILD_DIR/test_sw_tuner" --candidates &&
    bash scripts/on_gpu.sh "$BUILD_DIR/test_sw_seven_batch" &&
    bash scripts/on_gpu.sh "$BUILD_DIR/test_sw_small_native"
  fi
}

# --- Bench command ---
# Run benchmark without exiting on failure — we need trajectory even for failed runs
: > "$RESULTS_DIR/current.csv"
set +e
{ bash scripts/build.sh && sw_precheck && bash scripts/on_gpu.sh "$BUILD_DIR/megafmm_bench" \
  --algorithm "${MEGAFMM_ALGORITHM:-dense}" \
  --shapes "${MEGAFMM_SHAPES:-1024,1024,1024}" \
  --dtype "${MEGAFMM_DTYPE:-both}" \
  --rotate-inputs "${MEGAFMM_ROTATE_INPUTS:-1}" \
  --trials "${MEGAFMM_TRIALS:-100}" --warmup "${MEGAFMM_WARMUP:-20}" \
  --groups "${MEGAFMM_GROUPS:-5}" --correct-trials "${MEGAFMM_CORRECT_TRIALS:-3}" \
  --output "$RESULTS_DIR/current.csv" "${MODE_ARGS[@]}"; } 2>&1 | tee "$RESULTS_DIR/bench_output.txt"
BENCH_EXIT=$?
set -e
# --- End bench command ---

# --- Trajectory ---
if [ -n "$LABEL" ]; then
    TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"
else
    TRAJ_DIR="trajectory/${TIMESTAMP}"
fi
mkdir -p "$TRAJ_DIR"
cp -r solution/* "$TRAJ_DIR/"
[ -f "$RESULTS_DIR/bench_output.txt" ] && mv "$RESULTS_DIR/bench_output.txt" "$TRAJ_DIR/output.txt"
[ -f "$RESULTS_DIR/current.csv" ] && cp "$RESULTS_DIR/current.csv" "$TRAJ_DIR/results.csv"
[ -f "$RESULTS_DIR/current.csv.sw_tuning.csv" ] && cp "$RESULTS_DIR/current.csv.sw_tuning.csv" "$TRAJ_DIR/results.csv.sw_tuning.csv"
echo "Trajectory saved to: $TRAJ_DIR"

exit $BENCH_EXIT
