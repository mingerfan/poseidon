#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CMAKE_BIN="${CMAKE_BIN:-cmake}"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-mgpu-memcpy-bench}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM:-3.5}"

SOURCE_DEVICE="${SOURCE_DEVICE:-0}"
DESTINATION_DEVICE="${DESTINATION_DEVICE:-1}"
ITERATIONS="${ITERATIONS:-50}"
WARMUP="${WARMUP:-5}"
DEGREE="${DEGREE:-32768}"
COMPONENTS="${COMPONENTS:-2}"
P_COUNT="${P_COUNT:-0}"
ALLOW_SAME_DEVICE="${ALLOW_SAME_DEVICE:-0}"
MODES="${MODES:-}"

CMAKE_ARGS=(
    -S "${REPO_ROOT}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DPOSEIDON_BUILD_MGPU=ON
    -DPOSEIDON_BUILD_MGPU_TESTS=OFF
    -DPOSEIDON_BUILD_MGPU_BENCH=ON
    -DPOSEIDON_BUILD_MGPU_CUDA_COMM=ON
    -DPOSEIDON_BUILD_MGPU_GPU_OBJECTS=ON
    -DPOSEIDON_BUILD_EXAMPLES=OFF
    -DPOSEIDON_BUILD_BENCH=OFF
    -DPOSEIDON_BUILD_DEPS=ON
    -DCMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM}"
)

RUN_ARGS=(
    --source-device "${SOURCE_DEVICE}"
    --destination-device "${DESTINATION_DEVICE}"
    --iterations "${ITERATIONS}"
    --warmup "${WARMUP}"
    --degree "${DEGREE}"
    --components "${COMPONENTS}"
    --p-count "${P_COUNT}"
)

if [[ "${ALLOW_SAME_DEVICE}" != "0" ]]; then
    RUN_ARGS+=(--allow-same-device)
fi

if [[ -n "${MODES}" ]]; then
    RUN_ARGS+=(--modes "${MODES}")
fi

RUN_ARGS+=("$@")

echo "=== Configure cudaMemcpyPeer benchmark ==="
"${CMAKE_BIN}" "${CMAKE_ARGS[@]}"

echo "=== Build cudaMemcpyPeer benchmark ==="
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target poseidon_mgpu_ckks_transfer_bench -j"${JOBS:-2}"

BENCH_BIN="${BUILD_DIR}/bin/poseidon_mgpu_ckks_transfer_bench"
if [[ ! -x "${BENCH_BIN}" ]]; then
    echo "Benchmark binary was not produced: ${BENCH_BIN}" >&2
    exit 1
fi

echo "=== Run cudaMemcpyPeer benchmark ==="
echo "${BENCH_BIN} ${RUN_ARGS[*]}"
"${BENCH_BIN}" "${RUN_ARGS[@]}"
