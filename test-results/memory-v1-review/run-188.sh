#!/usr/bin/env bash
set -euo pipefail
cd /home/xuming/poseidon-memory-step3-20261008
export PATH=/home/xuming/poseidon-tools/openmpi/bin:/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/home/xuming/poseidon-tools/openmpi/lib:/home/xuming/poseidon-tools/nccl/nvidia/nccl/lib:/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}
export POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB=64
export OMP_NUM_THREADS=2
export POSEIDON_GPU_MLP_ALLOW_NO_BOOT=1
export NCCL_DEBUG=WARN
out=memory-v1-review/results
mkdir -p "$out"
run() {
 local name=$1; shift
 if timeout 180s "$@" >"$out/$name.log" 2>&1; then
  printf '%s: PASS\n' "$name"; tail -2 "$out/$name.log"
 else local status=$?; printf '%s: FAIL (%s)\n' "$name" "$status"; tail -12 "$out/$name.log"; return "$status"; fi
}
mpi=(mpirun --bind-to none --oversubscribe --mca btl ^openib)
run cpu-api build/bin/poseidon_runtime_cpu_api_tests
run nccl-2x2 "${mpi[@]}" -n 2 build/bin/poseidon_nccl_mpi_smoke --device-counts 2x2
run transfer-2x2 "${mpi[@]}" -n 2 build/bin/poseidon_gpu_mpi_transfer_test --device-counts 2x2 --rank-to-node 0x0
for topology in 1gpu 4gpu 2x2; do
 if [[ $topology == 2x2 ]]; then launcher=("${mpi[@]}" -n 2); mode=(--rank-to-node 0x0); else launcher=(); mode=(--local); fi
 for variant in baseline release reuse; do
  for execution in sequential per_device_workers; do
  run "smoke-$topology-$variant-$execution" "${launcher[@]}" build/bin/poseidon_gpu_mpi_runtime_e2e \
   "memory-v1-review/smoke/$topology.$variant.memory_smoke.runtime-plan.json" memory-v1-review/smoke/operator-spec.json - \
   "$out/smoke-$topology-$variant-$execution.json" "${mode[@]}" --warmups 1 --iterations 10 --measure-memory \
   --expected-output "memory-v1-review/smoke/$topology.expected.json" --execution-mode "$execution"
  done
 done
 for variant in baseline release reuse; do
  run "mlp-$topology-$variant-repeat" "${launcher[@]}" build/bin/poseidon_gpu_mpi_runtime_e2e \
   "memory-v1-review/$topology/mlp.$variant._hecate_MLP.runtime-plan.json" memory-v1-review/operator-spec.json \
   "memory-v1-review/$topology/mlp.$variant._hecate_MLP.bundle" \
   "$out/mlp-$topology-$variant-repeat.json" "${mode[@]}" --warmups 1 --iterations 10 --measure-memory
  if [[ $topology != 2x2 ]]; then
   run "mlp-$topology-$variant-numerical" build/bin/poseidon_runtime_gpu_mlp_e2e \
    "memory-v1-review/$topology/mlp.$variant._hecate_MLP.runtime-plan.json" memory-v1-review/operator-spec.json \
    "memory-v1-review/$topology/mlp.$variant._hecate_MLP.bundle" memory-v1-review/input.json memory-v1-review/mock-result.json \
    "$out/mlp-$topology-$variant-numerical.json"
  fi
 done
done
