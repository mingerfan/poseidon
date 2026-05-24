#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
BUILD_TYPE="${BUILD_TYPE:-Debug}"
ENABLE_GPU_TESTS="${ENABLE_GPU_TESTS:-ON}"

CMAKE_ARGS=(
    -S "${SCRIPT_DIR}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DPOSEIDON_GPU_DATA_STRUCT_ENABLE_GPU_TESTS="${ENABLE_GPU_TESTS}"
)

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=${CONDA_PREFIX}")
fi

echo "=== Configure ==="
cmake "${CMAKE_ARGS[@]}"

echo "=== Build ==="
if [[ "${ENABLE_GPU_TESTS}" != "ON" ]]; then
    echo "GPU demo is disabled because ENABLE_GPU_TESTS=${ENABLE_GPU_TESTS}"
    exit 0
fi

cmake --build "${BUILD_DIR}" --target demo_gpu_ciphertext_add_handler -j"$(nproc)"

DEMO_BIN="${BUILD_DIR}/demo_gpu_ciphertext_add_handler"
if [[ ! -x "${DEMO_BIN}" ]]; then
    echo "Demo binary was not produced: ${DEMO_BIN}" >&2
    exit 1
fi

echo "=== Run GPU elementwise demo ==="
set +e
"${DEMO_BIN}"
DEMO_STATUS=$?
set -e

if [[ "${DEMO_STATUS}" == "77" ]]; then
    echo "GPU elementwise demo skipped"
elif [[ "${DEMO_STATUS}" != "0" ]]; then
    exit "${DEMO_STATUS}"
fi
