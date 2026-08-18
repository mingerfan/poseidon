#!/usr/bin/env bash
set -Eeuo pipefail

repo=/home/xuming/poseidon-gpu-worker-experiment
run_name=${1:-repeat}
timing_iterations=${2:-50}
warmup_iterations=${3:-20}
result_dir="$repo/test-results/v100-profile-e63ec18/levels-$run_name"
benchmark="$repo/build-v100-profile-sm70/demo_gpu_ciphertext_add_handler"

mkdir -p "$result_dir"
for q_count in 5 6 7 8 9 10 11 12 13 14 15 16 17; do
    printf 'START q_count=%s timestamp=%(%FT%T%z)T\n' "$q_count" -1
    env \
        CUDA_VISIBLE_DEVICES=0 \
        LD_LIBRARY_PATH="$repo/build-v100-profile-sm70/poseidon_build:/usr/local/cuda-12.2/lib64" \
        POSEIDON_DEMO_DEGREE=8192 \
        POSEIDON_DEMO_Q_COUNT="$q_count" \
        POSEIDON_DEMO_P_COUNT=2 \
        POSEIDON_DEMO_WARMUP="$warmup_iterations" \
        POSEIDON_DEMO_ITERATIONS="$timing_iterations" \
        "$benchmark" > "$result_dir/q${q_count}.log" 2>&1
    printf 'DONE q_count=%s timestamp=%(%FT%T%z)T\n' "$q_count" -1
done
