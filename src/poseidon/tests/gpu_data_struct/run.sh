#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ENABLE_GPU_TESTS="${ENABLE_GPU_TESTS:-ON}"
RUN_NSYS_MULTIPLY_PLAIN="${RUN_NSYS_MULTIPLY_PLAIN:-0}"
RUN_NSYS_MUL_RELIN_RESCALE="${RUN_NSYS_MUL_RELIN_RESCALE:-0}"
POSEIDON_NSYS_DEGREE="${POSEIDON_NSYS_DEGREE:-16384}"
POSEIDON_NSYS_Q_COUNT="${POSEIDON_NSYS_Q_COUNT:-16}"
POSEIDON_NSYS_P_COUNT="${POSEIDON_NSYS_P_COUNT:-0}"
POSEIDON_NSYS_Q_SWEEP="${POSEIDON_NSYS_Q_SWEEP:-}"
POSEIDON_NSYS_ITERATIONS="${POSEIDON_NSYS_ITERATIONS:-200}"
POSEIDON_NSYS_WARMUP="${POSEIDON_NSYS_WARMUP:-5}"
NSYS_OUTPUT="${NSYS_OUTPUT:-${BUILD_DIR}/nsys_mp_q${POSEIDON_NSYS_Q_COUNT}}"
POSEIDON_NSYS_CHAIN_DEGREE="${POSEIDON_NSYS_CHAIN_DEGREE:-65536}"
POSEIDON_NSYS_CHAIN_Q_COUNT="${POSEIDON_NSYS_CHAIN_Q_COUNT:-32}"
POSEIDON_NSYS_CHAIN_P_COUNT="${POSEIDON_NSYS_CHAIN_P_COUNT:-6}"
POSEIDON_NSYS_CHAIN_ITERATIONS="${POSEIDON_NSYS_CHAIN_ITERATIONS:-10}"
POSEIDON_NSYS_CHAIN_WARMUP="${POSEIDON_NSYS_CHAIN_WARMUP:-1}"
NSYS_CHAIN_OUTPUT="${NSYS_CHAIN_OUTPUT:-${BUILD_DIR}/nsys_mul_relin_rescale_N${POSEIDON_NSYS_CHAIN_DEGREE}_q${POSEIDON_NSYS_CHAIN_Q_COUNT}_p${POSEIDON_NSYS_CHAIN_P_COUNT}}"
DEFAULT_NSYS_BIN="${HOME}/tools/nsight-systems/nsight_systems-linux-x86_64-2025.6.3.541-archive/target-linux-x64/nsys"
if [[ -z "${NSYS_BIN:-}" && -x "${DEFAULT_NSYS_BIN}" ]]; then
    NSYS_BIN="${DEFAULT_NSYS_BIN}"
else
    NSYS_BIN="${NSYS_BIN:-nsys}"
fi
NSYS_TRACE="${NSYS_TRACE:-cuda-sw,nvtx}"
NSYS_CAPTURE_RANGE="${NSYS_CAPTURE_RANGE:-cudaProfilerApi}"
NSYS_CAPTURE_RANGE_END="${NSYS_CAPTURE_RANGE_END:-stop}"

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
if [[ "${RUN_NSYS_MUL_RELIN_RESCALE}" != "0" ]]; then
    echo "Using nsys: ${NSYS_BIN}"
    "${NSYS_BIN}" --version
    POSEIDON_NSYS_MUL_RELIN_RESCALE=1 \
    POSEIDON_NSYS_DEGREE="${POSEIDON_NSYS_CHAIN_DEGREE}" \
    POSEIDON_NSYS_Q_COUNT="${POSEIDON_NSYS_CHAIN_Q_COUNT}" \
    POSEIDON_NSYS_P_COUNT="${POSEIDON_NSYS_CHAIN_P_COUNT}" \
    POSEIDON_NSYS_ITERATIONS="${POSEIDON_NSYS_CHAIN_ITERATIONS}" \
    POSEIDON_NSYS_WARMUP="${POSEIDON_NSYS_CHAIN_WARMUP}" \
    "${NSYS_BIN}" profile \
        --trace="${NSYS_TRACE}" \
        --capture-range="${NSYS_CAPTURE_RANGE}" \
        --capture-range-end="${NSYS_CAPTURE_RANGE_END}" \
        --cuda-um-cpu-page-faults=false \
        --cuda-um-gpu-page-faults=false \
        --sample=none \
        --cpuctxsw=none \
        --export=sqlite \
        --force-overwrite=true \
        --stats=true \
        -o "${NSYS_CHAIN_OUTPUT}" \
        "${DEMO_BIN}"
elif [[ "${RUN_NSYS_MULTIPLY_PLAIN}" != "0" ]]; then
    echo "Using nsys: ${NSYS_BIN}"
    "${NSYS_BIN}" --version
    POSEIDON_NSYS_MULTIPLY_PLAIN=1 \
    POSEIDON_NSYS_DEGREE="${POSEIDON_NSYS_DEGREE}" \
    POSEIDON_NSYS_Q_COUNT="${POSEIDON_NSYS_Q_COUNT}" \
    POSEIDON_NSYS_P_COUNT="${POSEIDON_NSYS_P_COUNT}" \
    POSEIDON_NSYS_Q_SWEEP="${POSEIDON_NSYS_Q_SWEEP}" \
    POSEIDON_NSYS_ITERATIONS="${POSEIDON_NSYS_ITERATIONS}" \
    POSEIDON_NSYS_WARMUP="${POSEIDON_NSYS_WARMUP}" \
    "${NSYS_BIN}" profile \
        --trace="${NSYS_TRACE}" \
        --capture-range="${NSYS_CAPTURE_RANGE}" \
        --capture-range-end="${NSYS_CAPTURE_RANGE_END}" \
        --cuda-um-cpu-page-faults=false \
        --cuda-um-gpu-page-faults=false \
        --sample=none \
        --cpuctxsw=none \
        --export=sqlite \
        --force-overwrite=true \
        --stats=true \
        -o "${NSYS_OUTPUT}" \
        "${DEMO_BIN}"
else
    "${DEMO_BIN}"
fi
DEMO_STATUS=$?
set -e

if [[ "${DEMO_STATUS}" == "77" ]]; then
    echo "GPU elementwise demo skipped"
elif [[ "${DEMO_STATUS}" != "0" ]]; then
    exit "${DEMO_STATUS}"
fi
