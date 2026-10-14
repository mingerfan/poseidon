#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/xuming/poseidon-tools/openmpi/bin:/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/home/xuming/poseidon-tools/openmpi/lib:/home/xuming/poseidon-tools/nccl/nvidia/nccl/lib:/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=2 POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB=64
models=/home/xuming/poseidon-plaintext-streaming-models
fixtures=/home/xuming/poseidon-memory-step3-20261008/memory-v1-review
runner=/home/xuming/poseidon-memory-step3-20261008/build/bin/poseidon_gpu_mpi_runtime_e2e
mkdir -p "$models/trace"
for variant in stream prefetch; do
 /usr/local/cuda-12.2/bin/nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx \
  --force-overwrite=true -o "$models/trace/$variant" \
  "$runner" "$models/4gpu-$variant._hecate_MLP.runtime-plan.json" "$fixtures/operator-spec.json" \
  "$models/4gpu-$variant._hecate_MLP.bundle" "$models/trace/$variant.json" \
  --local --warmups 1 --iterations 3 --execution-mode per_device_workers \
  > "$models/trace/$variant.log" 2>&1
 /usr/local/cuda-12.2/bin/nsys export --type sqlite --force-overwrite=true \
  --output "$models/trace/$variant.sqlite" "$models/trace/$variant.nsys-rep" \
  > "$models/trace/$variant.export.log" 2>&1
done
