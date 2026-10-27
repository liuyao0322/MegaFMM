#!/bin/bash
# Benchmark wrapper with trajectory tracking
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."
RESULTS_DIR="${MEGAFMM_RESULT_DIR:-results}"
BUILD_DIR="${MEGAFMM_BUILD_DIR:-build}"
mkdir -p "$RESULTS_DIR"

AFFINITY_ARGS=()
[[ -n "${MEGAFMM_BENCH_CPUSET:-}" ]] && AFFINITY_ARGS=(taskset -c "$MEGAFMM_BENCH_CPUSET")
MODE_ARGS=()
[[ "${MEGAFMM_GRAPH:-0}" == 1 ]] && MODE_ARGS+=(--graph)
SHAPE_ARGS=()
[[ -n "${MEGAFMM_SHAPES:-}" ]] && SHAPE_ARGS=(--shapes "$MEGAFMM_SHAPES")
CONFIG_OVERRIDES=()
for setting in MEGAFMM_DTYPE:--dtype MEGAFMM_TRIALS:--trials MEGAFMM_WARMUP:--warmup \
               MEGAFMM_GROUPS:--groups MEGAFMM_CORRECT_TRIALS:--correct-trials; do
  variable="${setting%%:*}"
  [[ -z "${!variable:-}" ]] || CONFIG_OVERRIDES+=("${setting#*:}" "${!variable}")
done
LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

sw_precheck() {
  if [[ "${MEGAFMM_SW_PRECHECK:-0}" == 1 ]]; then
    bash scripts/on_gpu.sh "$BUILD_DIR/test_sw_paths"
  fi
}

# --- Bench command ---
BUILD_SCRIPT=scripts/build.sh
BENCH_BINARY="$BUILD_DIR/megafmm_bench"
if [[ "${MEGAFMM_PACKED_COMPARE:-0}" == 1 ]]; then
  BUILD_SCRIPT=scripts/build_packed_compare.sh
  BENCH_BINARY="$BUILD_DIR/packed_fp32_compare"
  [[ -z "${MEGAFMM_PACKED_CONFIG:-}" ]] || CONFIG_OVERRIDES+=(--packed-config "$MEGAFMM_PACKED_CONFIG")
fi
# Run benchmark without exiting on failure — we need trajectory even for failed runs
: > "$RESULTS_DIR/current.csv"
set +e
{ bash "$BUILD_SCRIPT" && sw_precheck && bash scripts/on_gpu.sh "${AFFINITY_ARGS[@]}" python3 scripts/run_benchmark.py \
  --config "${MEGAFMM_SHAPES_FILE:-bench/shapes.json}" --binary "$BENCH_BINARY" \
  --algorithm "${MEGAFMM_ALGORITHM:-dense}" \
  "${SHAPE_ARGS[@]}" \
  --rotate-inputs "${MEGAFMM_ROTATE_INPUTS:-1}" \
  "${CONFIG_OVERRIDES[@]}" \
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
