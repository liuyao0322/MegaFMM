#!/bin/bash
set -euo pipefail
PROFILE_ARCHIVE=$(cd "$(dirname "$0")" && pwd)
cd "$(git -C "$PROFILE_ARCHIVE" rev-parse --show-toplevel)"
PROFILE_OUTPUT="${MEGAFMM_PROFILE_DIR:-results/production_chain_replay}"
mkdir -p "$PROFILE_OUTPUT/source"
python3 "$PROFILE_ARCHIVE/generate.py" "$PROFILE_OUTPUT/source"
cp "$PROFILE_ARCHIVE/source/measure.cu" "$PROFILE_OUTPUT/source/measure.cu"
bash scripts/on_gpu.sh bash "$PROFILE_ARCHIVE/build_and_run.sh" "$PROFILE_OUTPUT"
python3 "$PROFILE_ARCHIVE/analyze.py" "$PROFILE_OUTPUT" > "$PROFILE_OUTPUT/analysis.log"
