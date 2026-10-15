#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
r=research/precision_20261016/int8_hardware
bash "$r/build_probe_p53_layout.sh" > "$r/probe_p53_layout_build.log" 2>&1
"$r/bin/probe_p53_layout" > "$r/probe_p53_layout.log" 2>&1
