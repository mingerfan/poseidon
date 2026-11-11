#!/usr/bin/env bash
set -euo pipefail
R=${POSEIDON_TRACE_RUN_DIR:-/home/xuming/poseidon-api-trace-20261111}
GPU=${POSEIDON_TRACE_PHYSICAL_GPU:-0}
D=${POSEIDON_TRACE_PLAN_DIR:-$R/trace-run}
exec 9>"/home/xuming/.poseidon-v100-gpu${GPU}.lock"
flock -n 9
trap 'rc=$?; echo "failed rc=$rc $(date -Is)" > "$R/run.state"' ERR
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MALLOC_ARENA_MAX=2
export POSEIDON_GPU_KEY_VIEW_CACHE=1 POSEIDON_GPU_LAZY_BASECONV=1
export POSEIDON_GPU_MODDOWN_FOURSTEP=1 POSEIDON_GPU_QP_FOURSTEP_ANY_P=1
export POSEIDON_DOUBLE_HOIST_QP_C0=1 POSEIDON_DOUBLE_HOIST_DIRECT_GIANT_ACCUMULATE=1
export POSEIDON_S2C_LIBRARY_BOOT=1 POSEIDON_S2C_CUPTI=1
unset LD_PRELOAD CUDA_INJECTION64_PATH NVTX_INJECTION64_PATH CUDA_LAUNCH_BLOCKING
unset POSEIDON_S2C_CONV_MERGED_MASK POSEIDON_S2C_SHORT_CHAIN POSEIDON_S2C_SECRET_H
unset POSEIDON_S2C_BOOT_LEVEL_P POSEIDON_S2C_APP_DNUM1 POSEIDON_S2C_APP_DNUM1_MAX_Q
unset POSEIDON_S2C_REPLAY_SKIP_SLOTS POSEIDON_GPU_SMALL_P_FOURSTEP
if test -e "$D"; then echo "refusing to overwrite $D" >&2; exit 1; fi
mkdir "$D"
test "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" = 0
env | grep -E '^(CUDA_|OMP_|OPENBLAS_|MKL_|MALLOC_|POSEIDON_)' | sort > "$R/environment.txt"
nvidia-smi -q -i "$GPU" > "$R/gpu.txt"
echo "running unit GPU tests $(date -Is)" > "$R/run.state"
for test in gpu_key_view_cache_test gpu_uploader_stream_order_test gpu_selected_prerotated_crop_test gpu_trace_plan_validation_test; do
  "$R/build/bin/$test" > "$R/$test.log" 2>&1
done
"$R/build/bin/test_gpu_split_hoist" --small-only > "$R/hoist-small-fixed.log" 2>&1
echo "tracing full Q49 ResNet20 $(date -Is)" > "$R/run.state"
/usr/bin/time -p -o "$R/network-process-time.txt" stdbuf -oL -eL \
  "$R/build/bin/poseidon_resnet20_s2c_first" --unsafe-trace-plan \
  "$R/src/Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt" 0 "$D" \
  > "$R/network.log" 2>&1
printf 'complete %s\n' "$(date -Is)" > "$R/run.state"
