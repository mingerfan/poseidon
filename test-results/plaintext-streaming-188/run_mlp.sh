#!/usr/bin/env bash
set -euo pipefail
models=/home/xuming/poseidon-plaintext-streaming-models
fixtures=/home/xuming/poseidon-memory-step3-20261008/memory-v1-review
runner=/home/xuming/poseidon/build-runtime-gpu-rmm2412/bin/poseidon_runtime_gpu_mlp_e2e
export OMP_NUM_THREADS=2 POSEIDON_GPU_MLP_ALLOW_NO_BOOT=1 POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB=64
mkdir -p "$models/results"
for topology in 1gpu 4gpu; do
 for variant in stream prefetch; do
  for mode in sequential workers; do
   if [[ $mode == workers ]]; then export POSEIDON_RUNTIME_DEVICE_WORKERS=${topology%gpu}; else unset POSEIDON_RUNTIME_DEVICE_WORKERS; fi
   tag=$topology-$variant-$mode
   /usr/bin/time -v -o "$models/results/$tag.time" timeout 180 "$runner" \
    "$models/$topology-$variant._hecate_MLP.runtime-plan.json" "$fixtures/operator-spec.json" \
    "$models/$topology-$variant._hecate_MLP.bundle" "$fixtures/input.json" "$fixtures/mock-result.json" \
    "$models/results/$tag.json" > "$models/results/$tag.log" 2>&1
  done
 done
done
