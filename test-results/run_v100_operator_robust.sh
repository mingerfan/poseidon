#!/usr/bin/env bash
set -Eeuo pipefail

repo=/home/xuming/poseidon-gpu-worker-experiment
for repeat in 1 2 3 4 5; do
    bash "$repo/test-results/run_v100_operator_levels.sh" "robust${repeat}" 10
done
