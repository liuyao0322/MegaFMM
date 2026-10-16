#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec research/precision_20261016/int8_hardware/bin/probe_p57_layout
