#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_DIR="${BUILD_DIR:-${DEFAULT_BUILD_DIR}}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ENABLE_GPU_TESTS="${ENABLE_GPU_TESTS:-ON}"
BUILD_CPU_TEST="${BUILD_CPU_TEST:-OFF}"
BUILD_GPU_STORAGE_TEST="${BUILD_GPU_STORAGE_TEST:-OFF}"
CMAKE_BIN="${CMAKE_BIN:-}"
POSEIDON_CLEAN_STALE_BUILD="${POSEIDON_CLEAN_STALE_BUILD:-ON}"
RUN_NSYS_MULTIPLY_PLAIN="${RUN_NSYS_MULTIPLY_PLAIN:-0}"
RUN_NSYS_RELINEARIZE="${RUN_NSYS_RELINEARIZE:-0}"
RUN_NSYS_MUL_RELIN_RESCALE="${RUN_NSYS_MUL_RELIN_RESCALE:-0}"
RUN_NCU_RELINEARIZE="${RUN_NCU_RELINEARIZE:-0}"
POSEIDON_NTT_ALGO="${POSEIDON_NTT_ALGO:-fourstep}"
POSEIDON_KEYSWITCH_FOURSTEP_ALL_NTT="${POSEIDON_KEYSWITCH_FOURSTEP_ALL_NTT:-1}"
POSEIDON_KEYSWITCH_FOURSTEP_PHASE2_MAC="${POSEIDON_KEYSWITCH_FOURSTEP_PHASE2_MAC:-1}"
POSEIDON_KEYSWITCH_FOURSTEP_FINALIZE_FUSED="${POSEIDON_KEYSWITCH_FOURSTEP_FINALIZE_FUSED:-1}"
export POSEIDON_KEYSWITCH_FOURSTEP_ALL_NTT
export POSEIDON_KEYSWITCH_FOURSTEP_PHASE2_MAC
export POSEIDON_KEYSWITCH_FOURSTEP_FINALIZE_FUSED
export POSEIDON_NTT_ALGO
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
POSEIDON_NSYS_RELIN_DEGREE="${POSEIDON_NSYS_RELIN_DEGREE:-65536}"
POSEIDON_NSYS_RELIN_Q_COUNT="${POSEIDON_NSYS_RELIN_Q_COUNT:-8}"
POSEIDON_NSYS_RELIN_P_COUNT="${POSEIDON_NSYS_RELIN_P_COUNT:-2}"
POSEIDON_NSYS_RELIN_ITERATIONS="${POSEIDON_NSYS_RELIN_ITERATIONS:-1}"
POSEIDON_NSYS_RELIN_WARMUP="${POSEIDON_NSYS_RELIN_WARMUP:-0}"
NSYS_RELIN_OUTPUT="${NSYS_RELIN_OUTPUT:-${BUILD_DIR}/nsys_relinearize_N${POSEIDON_NSYS_RELIN_DEGREE}_q${POSEIDON_NSYS_RELIN_Q_COUNT}_p${POSEIDON_NSYS_RELIN_P_COUNT}}"
DEFAULT_NSYS_BIN="${HOME}/tools/nsight-systems/nsight_systems-linux-x86_64-2025.6.3.541-archive/target-linux-x64/nsys"
if [[ -z "${NSYS_BIN:-}" && -x "${DEFAULT_NSYS_BIN}" ]]; then
    NSYS_BIN="${DEFAULT_NSYS_BIN}"
else
    NSYS_BIN="${NSYS_BIN:-nsys}"
fi
NSYS_TRACE="${NSYS_TRACE:-cuda-sw,nvtx}"
NSYS_CAPTURE_RANGE="${NSYS_CAPTURE_RANGE:-cudaProfilerApi}"
NSYS_CAPTURE_RANGE_END="${NSYS_CAPTURE_RANGE_END:-stop}"
DEFAULT_NCU_BIN="/usr/local/cuda/bin/ncu"
if [[ -z "${NCU_BIN:-}" && -x "${DEFAULT_NCU_BIN}" ]]; then
    NCU_BIN="${DEFAULT_NCU_BIN}"
else
    NCU_BIN="${NCU_BIN:-ncu}"
fi
NCU_SET="${NCU_SET:-full}"
NCU_TARGET_PROCESSES="${NCU_TARGET_PROCESSES:-all}"
NCU_PROFILE_FROM_START="${NCU_PROFILE_FROM_START:-off}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-20}"
NCU_KERNEL_NAME="${NCU_KERNEL_NAME:-regex:.*forward_ntt_cheddar_qp_active_phase2_mul_accumulate_65536_kernel.*}"
NCU_RELIN_OUTPUT="${NCU_RELIN_OUTPUT:-${BUILD_DIR}/ncu_relinearize_fourstep_fused_N${POSEIDON_NSYS_RELIN_DEGREE}_q${POSEIDON_NSYS_RELIN_Q_COUNT}_p${POSEIDON_NSYS_RELIN_P_COUNT}}"

if [[ -z "${CMAKE_BIN}" ]]; then
    if command -v cmake >/dev/null 2>&1; then
        CMAKE_BIN="$(command -v cmake)"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/cmake" ]]; then
        CMAKE_BIN="${CONDA_PREFIX}/bin/cmake"
    else
        echo "CMake was not found in PATH." >&2
        echo "Install CMake on the offline machine, activate the environment that provides it," >&2
        echo "or run with CMAKE_BIN=/absolute/path/to/cmake ./run.sh" >&2
        exit 127
    fi
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    CACHE_SOURCE_DIR="$(
        sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' \
            "${BUILD_DIR}/CMakeCache.txt" | tail -n 1
    )"
    if [[ -n "${CACHE_SOURCE_DIR}" && "${CACHE_SOURCE_DIR}" != "${SCRIPT_DIR}" ]]; then
        if [[ "${POSEIDON_CLEAN_STALE_BUILD}" != "0" && "${BUILD_DIR}" == "${DEFAULT_BUILD_DIR}" ]]; then
            echo "Removing stale copied CMake build directory: ${BUILD_DIR}"
            echo "Cached source directory was: ${CACHE_SOURCE_DIR}"
            "${CMAKE_BIN}" -E rm -rf "${BUILD_DIR}"
        else
            echo "Stale CMake cache detected in ${BUILD_DIR}" >&2
            echo "Cached source directory was: ${CACHE_SOURCE_DIR}" >&2
            echo "Current source directory is: ${SCRIPT_DIR}" >&2
            echo "Remove the build directory, set BUILD_DIR to a fresh path, or enable" >&2
            echo "POSEIDON_CLEAN_STALE_BUILD=ON for the default build directory." >&2
            exit 2
        fi
    fi
fi

CMAKE_ARGS=(
    -S "${SCRIPT_DIR}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DPOSEIDON_GPU_DATA_STRUCT_ENABLE_GPU_TESTS="${ENABLE_GPU_TESTS}"
    -DPOSEIDON_GPU_DATA_STRUCT_BUILD_CPU_TEST="${BUILD_CPU_TEST}"
    -DPOSEIDON_GPU_DATA_STRUCT_BUILD_GPU_STORAGE_TEST="${BUILD_GPU_STORAGE_TEST}"
)

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=${CONDA_PREFIX}")
fi

echo "=== Configure ==="
echo "Using cmake: ${CMAKE_BIN}"
"${CMAKE_BIN}" --version
"${CMAKE_BIN}" "${CMAKE_ARGS[@]}"

echo "=== Build ==="
if [[ "${ENABLE_GPU_TESTS}" != "ON" ]]; then
    echo "GPU demo is disabled because ENABLE_GPU_TESTS=${ENABLE_GPU_TESTS}"
    exit 0
fi

"${CMAKE_BIN}" --build "${BUILD_DIR}" --target demo_gpu_ciphertext_add_handler -j"$(nproc)"

DEMO_BIN="${BUILD_DIR}/demo_gpu_ciphertext_add_handler"
if [[ ! -x "${DEMO_BIN}" ]]; then
    echo "Demo binary was not produced: ${DEMO_BIN}" >&2
    exit 1
fi

echo "=== Run GPU elementwise demo ==="
set +e
if [[ "${RUN_NCU_RELINEARIZE}" != "0" ]]; then
    echo "Using ncu: ${NCU_BIN}"
    "${NCU_BIN}" --version
    POSEIDON_NSYS_RELINEARIZE=1 \
    POSEIDON_NSYS_DEGREE="${POSEIDON_NSYS_RELIN_DEGREE}" \
    POSEIDON_NSYS_Q_COUNT="${POSEIDON_NSYS_RELIN_Q_COUNT}" \
    POSEIDON_NSYS_P_COUNT="${POSEIDON_NSYS_RELIN_P_COUNT}" \
    POSEIDON_NSYS_ITERATIONS="${POSEIDON_NSYS_RELIN_ITERATIONS}" \
    POSEIDON_NSYS_WARMUP="${POSEIDON_NSYS_RELIN_WARMUP}" \
    POSEIDON_NTT_ALGO="${POSEIDON_NTT_ALGO}" \
    POSEIDON_NTT_FUSION_STAGES="${POSEIDON_NTT_FUSION_STAGES:-3}" \
    "${NCU_BIN}" \
        --target-processes "${NCU_TARGET_PROCESSES}" \
        --set "${NCU_SET}" \
        --kernel-name "${NCU_KERNEL_NAME}" \
        --launch-count "${NCU_LAUNCH_COUNT}" \
        --profile-from-start "${NCU_PROFILE_FROM_START}" \
        --force-overwrite \
        --export "${NCU_RELIN_OUTPUT}" \
        "${DEMO_BIN}"
elif [[ "${RUN_NSYS_MUL_RELIN_RESCALE}" != "0" ]]; then
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
elif [[ "${RUN_NSYS_RELINEARIZE}" != "0" ]]; then
    echo "Using nsys: ${NSYS_BIN}"
    "${NSYS_BIN}" --version
    POSEIDON_NSYS_RELINEARIZE=1 \
    POSEIDON_NSYS_DEGREE="${POSEIDON_NSYS_RELIN_DEGREE}" \
    POSEIDON_NSYS_Q_COUNT="${POSEIDON_NSYS_RELIN_Q_COUNT}" \
    POSEIDON_NSYS_P_COUNT="${POSEIDON_NSYS_RELIN_P_COUNT}" \
    POSEIDON_NSYS_ITERATIONS="${POSEIDON_NSYS_RELIN_ITERATIONS}" \
    POSEIDON_NSYS_WARMUP="${POSEIDON_NSYS_RELIN_WARMUP}" \
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
        -o "${NSYS_RELIN_OUTPUT}" \
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
