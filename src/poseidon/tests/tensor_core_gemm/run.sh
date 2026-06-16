#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"

"${CMAKE_BIN}" -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
"${CMAKE_BIN}" --build "${BUILD_DIR}" -j"$(nproc)"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target test
"${BUILD_DIR}/tensor_core_gemm_bench" --size "${POSEIDON_TENSOR_CORE_BENCH_SIZE:-256}" --mode all --warmup 1 --repeat 3
