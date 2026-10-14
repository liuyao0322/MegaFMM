#!/bin/bash
set -euo pipefail
cuobjdump --dump-sass build/megafmm_bench > results/profile_iter97/production.sass
cuobjdump --dump-sass results/profile_iter97/diagnostics/measure > results/profile_iter97/diagnostic.sass
