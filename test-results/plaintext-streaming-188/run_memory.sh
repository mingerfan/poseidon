#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/xuming/poseidon-tools/openmpi/bin:/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/home/xuming/poseidon-tools/openmpi/lib:/home/xuming/poseidon-tools/nccl/nvidia/nccl/lib:/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=2 POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB=64
models=/home/xuming/poseidon-plaintext-streaming-models
fixtures=/home/xuming/poseidon-memory-step3-20261008/memory-v1-review
runner=/home/xuming/poseidon-memory-step3-20261008/build/bin/poseidon_gpu_mpi_runtime_e2e
mkdir -p "$models/memory"
for topology in 1gpu 4gpu; do
 for variant in stream prefetch; do
  for mode in sequential per_device_workers; do
   tag=$topology-$variant-$mode
   /usr/bin/time -v -o "$models/memory/$tag.time" timeout 180 "$runner" \
    "$models/$topology-$variant._hecate_MLP.runtime-plan.json" "$fixtures/operator-spec.json" \
    "$models/$topology-$variant._hecate_MLP.bundle" "$models/memory/$tag.json" \
    --local --warmups 1 --iterations 3 --measure-memory --execution-mode "$mode" \
    > "$models/memory/$tag.log" 2>&1
  done
 done
done
