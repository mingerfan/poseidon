#!/usr/bin/env bash
set -euo pipefail
D=${1:?usage: run_poseidon.sh EXPERIMENT_ROOT}
RUN="$D/src/experiments/gpu_keyload/run_matrix.py"
BIN="$D/poseidon_gpu_keyload"
export POSEIDON_GPU_KEYLOAD_TRACE=1
run() {
  local name=$1 threads=$2 wait=$3 affinity=$4 q=$5 p=$6 params=$7
  python3 "$RUN" --binary "$BIN" --results "$D/results" --case "$name" \
    --threads "$threads" --wait "$wait" --affinity "$affinity" --timeout 240 -- \
    16 "$q" "$p" 8 "$D/results/$name/probe.json" --check full --parameter-mode "$params"
}
for rep in 1 2 3; do
  run "poseidon-q32-omp1-r$rep" 1 PASSIVE 0-6 32 16 none
done
run poseidon-q32-omp7-passive 7 PASSIVE 0-6 32 16 none
run poseidon-q32-omp7-active 7 ACTIVE 0-6 32 16 none
run poseidon-q32-default64 unset DEFAULT all 32 16 none || true
# Deliberately over-subscribed stress diagnostic; not a recommended setting.
run poseidon-q32-omp64-active 64 ACTIVE 0-6 32 16 none || true
run poseidon-q50-full 1 PASSIVE 0-6 50 25 full
run poseidon-q50-sparse 1 PASSIVE 0-6 50 25 sparse
python3 "$D/src/bench/gpu_keyload/compare_parameters.py" \
  "$D/results/poseidon-q50-full/probe.json" "$D/results/poseidon-q50-sparse/probe.json" \
  > "$D/results/parameter-comparison.json"
