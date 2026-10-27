#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 tests/test_dag.py
bash scripts/on_gpu.sh ./build/test_correctness "$@"
