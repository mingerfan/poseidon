#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${POSEIDON_GPU_QWEN_BUILD_DIR:-${script_dir}/build}"
output_dir="${POSEIDON_GPU_QWEN_OUTPUT_DIR:-${script_dir}/output}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${POSEIDON_GPU_QWEN_LOG_FILE:-${output_dir}/qwen_gpu_${timestamp}.log}"

mkdir -p "${output_dir}"
start_epoch="$(date +%s)"

{
    echo "run_started_at=$(date --iso-8601=seconds)"
    echo "command=$0 $*"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "POSEIDON_NTT_ALGO=${POSEIDON_NTT_ALGO:-unset}"
    echo "POSEIDON_GPU_QWEN_ENCODE_THREADS=${POSEIDON_GPU_QWEN_ENCODE_THREADS:-8}"
    echo "POSEIDON_GPU_QWEN_PROFILE_LINEAR=${POSEIDON_GPU_QWEN_PROFILE_LINEAR:-unset}"

    cmake -S "${script_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${build_dir}" --target poseidon_gpu_qwen -j"${POSEIDON_GPU_QWEN_BUILD_JOBS:-2}"
    "${build_dir}/poseidon_gpu_qwen" "$@"
    status=$?

    end_epoch="$(date +%s)"
    elapsed="$((end_epoch - start_epoch))"
    printf 'run_finished_at=%s status=%d elapsed_seconds=%d elapsed_hms=%02dh%02dm%02ds\n' \
        "$(date --iso-8601=seconds)" "${status}" "${elapsed}" \
        "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
    exit "${status}"
} 2>&1 | tee "${log_file}"

exit "${PIPESTATUS[0]}"
