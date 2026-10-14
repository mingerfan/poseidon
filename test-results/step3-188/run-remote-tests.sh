#!/usr/bin/env bash
set -euo pipefail
cd /home/xuming/poseidon-memory-step3-20261008
export PATH=/home/xuming/poseidon-tools/openmpi/bin:/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/home/xuming/poseidon-tools/openmpi/lib:/home/xuming/poseidon-tools/nccl/nvidia/nccl/lib:/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}
export POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB=64
export OMP_NUM_THREADS=2
export NCCL_DEBUG=WARN
mkdir -p results
run() {
    local name=$1
    shift
    if timeout 180s "$@" >"results/$name.log" 2>&1; then
        printf '%s: PASS\n' "$name"
        tail -2 "results/$name.log"
    else
        local result=$?
        printf '%s: FAIL (%s)\n' "$name" "$result"
        tail -35 "results/$name.log"
        return "$result"
    fi
}
mpi=(mpirun --bind-to none --oversubscribe --mca btl ^openib)
if [[ ${1:-all} == all ]]; then
    run cpu-api build/bin/poseidon_runtime_cpu_api_tests
    run gpu-api build/bin/poseidon_runtime_gpu_api_tests
    run nccl-2x2 "${mpi[@]}" -n 2 build/bin/poseidon_nccl_mpi_smoke --device-counts 2x2
    run transfer-2x2 "${mpi[@]}" -n 2 build/bin/poseidon_gpu_mpi_transfer_test --device-counts 2x2 --rank-to-node 0x0
fi
for topology in 1x4 2x2; do
    if [[ $topology == 1x4 ]]; then ranks=1; nodes=0; else ranks=2; nodes=0x0; fi
    for variant in baseline release; do
        for mode in sequential per_device_workers; do
            name="$topology-$variant-$mode"
            export POSEIDON_RUNTIME_TRACE="$PWD/results/$name"
            run "$name" "${mpi[@]}" -n "$ranks" build/bin/poseidon_gpu_mpi_plan_e2e \
                "plans/$topology.$variant.json" plans/operator-spec.json "results/$name.json" \
                --rank-to-node "$nodes" --execution-mode "$mode"
        done
    done
    unset POSEIDON_RUNTIME_TRACE
    if [[ $topology == 1x4 ]]; then
        run "$topology-release-repeat" build/bin/poseidon_gpu_mpi_runtime_e2e \
            "plans/$topology.release.json" plans/operator-spec.json - \
            "results/$topology-release-repeat.json" --local --warmups 1 --iterations 10
    else
        run "$topology-release-repeat" "${mpi[@]}" -n 2 build/bin/poseidon_gpu_mpi_runtime_e2e \
            "plans/$topology.release.json" plans/operator-spec.json - \
            "results/$topology-release-repeat.json" --rank-to-node 0x0 --warmups 1 --iterations 10
    fi
done
