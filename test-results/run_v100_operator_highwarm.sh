#!/usr/bin/env bash
set -Eeuo pipefail

repo=/home/xuming/poseidon-gpu-worker-experiment
for repeat in 1 2 3; do
    bash "$repo/test-results/run_v100_operator_levels.sh" "highwarm${repeat}" 10 500
done
