#!/usr/bin/env bash
set -euo pipefail
D=${1:?usage: run_cerium.sh EXPERIMENT_ROOT}
export LD_LIBRARY_PATH="$D/cerium/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for rep in 1 2 3; do
  name="cerium-q32-r$rep"
  python3 "$D/src/experiments/gpu_keyload/run_matrix.py" \
    --binary "$D/src/bench/cerium_keyload/build/cerium_keyload" \
    --results "$D/results" --case "$name" --threads 1 --wait PASSIVE --affinity 0-6 \
    --timeout 240 -- --q 32 --p 16 --k 8 --hwt 192 --ephemeral-hwt 192 \
    --device 0 --tmp-root /tmp
 done
