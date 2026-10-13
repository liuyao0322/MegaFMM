#!/bin/bash
# Benchmark wrapper with trajectory tracking
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."

MODE_ARGS=()
[[ "${MEGAFMM_GRAPH:-0}" == 1 ]] && MODE_ARGS+=(--graph)
LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Bench command ---
# Run benchmark without exiting on failure — we need trajectory even for failed runs
: > results/current.csv
set +e
{ bash scripts/build.sh && bash scripts/on_gpu.sh ./build/megafmm_bench \
  --algorithm "${MEGAFMM_ALGORITHM:-lt}" \
  --shapes "${MEGAFMM_SHAPES:-4096,4096,4096;8192,8192,8192;16384,16384,16384}" \
  --dtype "${MEGAFMM_DTYPE:-both}" \
  --mega-variant "${MEGAFMM_MEGA_VARIANT:-0}" \
  --hopper-variant "${MEGAFMM_HOPPER_VARIANT:-0}" \
  --rotate-inputs "${MEGAFMM_ROTATE_INPUTS:-1}" \
  --trials "${MEGAFMM_TRIALS:-100}" --warmup "${MEGAFMM_WARMUP:-20}" \
  --groups "${MEGAFMM_GROUPS:-5}" --correct-trials "${MEGAFMM_CORRECT_TRIALS:-3}" \
  --output results/current.csv "${MODE_ARGS[@]}"; } 2>&1 | tee _bench_output.txt
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
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
[ -f results/current.csv ] && cp results/current.csv "$TRAJ_DIR/results.csv"
echo "Trajectory saved to: $TRAJ_DIR"

exit $BENCH_EXIT
