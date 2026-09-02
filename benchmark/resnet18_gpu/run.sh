#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${POSEIDON_GPU_RESNET18_BUILD_DIR:-${SCRIPT_DIR}/build}"
CMAKE_BIN="${CMAKE_BIN:-$(command -v cmake)}"
OUTPUT_DIR="${POSEIDON_GPU_RESNET18_OUTPUT_DIR:-${SCRIPT_DIR}/output}"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_STARTED_AT="$(date --iso-8601=seconds)"
LOG_FILE="${POSEIDON_GPU_RESNET18_LOG_FILE:-${OUTPUT_DIR}/resnet18_gpu_${RUN_TIMESTAMP}.log}"
mkdir -p -- "$(dirname -- "${LOG_FILE}")"

{
    printf '===== Poseidon GPU ResNet18 run started: %s =====\n' "${RUN_STARTED_AT}"
    printf 'command:'
    printf ' %q' "$0" "$@"
    printf '\nlog_file=%s\n' "${LOG_FILE}"

    "${CMAKE_BIN}" -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=70
    "${CMAKE_BIN}" --build "${BUILD_DIR}" \
        --target poseidon_gpu_resnet18 -j "${BUILD_JOBS:-2}"

    set +e
    "${BUILD_DIR}/poseidon_gpu_resnet18" "$@"
    RUN_STATUS=$?
    set -e
    printf '===== Poseidon GPU ResNet18 run finished: %s status=%d =====\n' \
        "$(date --iso-8601=seconds)" "${RUN_STATUS}"
    exit "${RUN_STATUS}"
} 2>&1 | tee "${LOG_FILE}"
