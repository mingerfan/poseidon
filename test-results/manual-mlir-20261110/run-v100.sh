#!/usr/bin/env bash
set -euo pipefail
R=${POSEIDON_MANUAL_MLIR_RUN_DIR:-/home/xuming/poseidon-manual-mlir-20261110}
exec 9>/home/xuming/.poseidon-v100-gpu0.lock
flock 9
trap 'rc=$?; echo "failed rc=$rc $(date -Is)" > "$R/network-state.txt"' ERR
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MALLOC_ARENA_MAX=2
export POSEIDON_GPU_KEY_VIEW_CACHE=1 POSEIDON_GPU_LAZY_BASECONV=1
export POSEIDON_GPU_MODDOWN_FOURSTEP=1 POSEIDON_GPU_QP_FOURSTEP_ANY_P=1
export POSEIDON_DOUBLE_HOIST_QP_C0=1 POSEIDON_DOUBLE_HOIST_DIRECT_GIANT_ACCUMULATE=1
export POSEIDON_S2C_LIBRARY_BOOT=1
unset LD_PRELOAD CUDA_INJECTION64_PATH NVTX_INJECTION64_PATH CUDA_LAUNCH_BLOCKING
FIXTURE="$R/src/Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt"
BINARY="$R/build/bin/poseidon_resnet20_s2c_first"
env | grep -E '^(CUDA_|OMP_|OPENBLAS_|MKL_|MALLOC_|POSEIDON_)' | sort > "$R/network-environment.txt"
for format in bin json; do
    echo "running $format $(date -Is)" > "$R/network-state.txt"
    test "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" = 0
    nvidia-smi -i 0 --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv > "$R/network-$format-gpu-before.txt"
    export POSEIDON_S2C_CUPTI=0
    if [[ $format == bin ]]; then export POSEIDON_S2C_CUPTI=1; fi
    /usr/bin/time -p -o "$R/network-$format-process-time.txt" \
        "$BINARY" --unsafe-runtime-plan "$FIXTURE" 0 \
        "$R/artifacts/compiled-$format/scheduled.resnet20_manual.runtime-plan.$format" \
        "$R/artifacts/compiled-$format/operator-spec.json" > "$R/network-$format.log" 2>&1
    grep 'PREPARED_RUNTIME_RESULT' "$R/network-$format.log"
    nvidia-smi -i 0 --query-gpu=index,memory.used,utilization.gpu --format=csv > "$R/network-$format-gpu-after.txt"
done
echo "complete $(date -Is)" > "$R/network-state.txt"
