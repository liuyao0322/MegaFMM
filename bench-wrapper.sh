#!/usr/bin/env bash
# Root entry point; the implementation owns benchmark and trajectory handling.
set -euo pipefail
exec bash "$(dirname "$0")/scripts/bench.sh" "$@"
