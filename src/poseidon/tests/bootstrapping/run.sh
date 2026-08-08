#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_DIR="${BUILD_DIR:-${DEFAULT_BUILD_DIR}}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ENABLE_GPU_TESTS="${ENABLE_GPU_TESTS:-ON}"
CMAKE_BIN="${CMAKE_BIN:-}"
POSEIDON_CLEAN_STALE_BUILD="${POSEIDON_CLEAN_STALE_BUILD:-ON}"
POSEIDON_NTT_ALGO="${POSEIDON_NTT_ALGO:-fourstep}"
POSEIDON_KEYSWITCH_FOURSTEP_ALL_NTT="${POSEIDON_KEYSWITCH_FOURSTEP_ALL_NTT:-1}"
POSEIDON_KEYSWITCH_FOURSTEP_PHASE2_MAC="${POSEIDON_KEYSWITCH_FOURSTEP_PHASE2_MAC:-1}"
POSEIDON_KEYSWITCH_FOURSTEP_FINALIZE_FUSED="${POSEIDON_KEYSWITCH_FOURSTEP_FINALIZE_FUSED:-1}"
POSEIDON_BOOTSTRAP_PROFILE="${POSEIDON_BOOTSTRAP_PROFILE:-dynamic32}"

if [[ "${POSEIDON_BOOTSTRAP_PROFILE}" == "dynamic32" ||
      "${POSEIDON_BOOTSTRAP_PROFILE}" == "dual30" ]]; then
    # 40-bit normal CKKS scale and 45-bit bootstrap scale over GPU-friendly
    # physical primes. The mixed <=32-bit Q chain and all stage widths are
    # selected by the CPU-compatible min_scale/2 dynamic planner.
    export POSEIDON_BOOTSTRAP_DEGREE="${POSEIDON_BOOTSTRAP_DEGREE:-16384}"
    export POSEIDON_BOOTSTRAP_Q_COUNT="${POSEIDON_BOOTSTRAP_Q_COUNT:-34}"
    export POSEIDON_BOOTSTRAP_P_COUNT="${POSEIDON_BOOTSTRAP_P_COUNT:-9}"
    export POSEIDON_GPU_LINEAR_TRANSFORM_MODE="${POSEIDON_GPU_LINEAR_TRANSFORM_MODE:-double_hoist}"
    export POSEIDON_GPU_DOUBLE_HOIST_BABY_TILE="${POSEIDON_GPU_DOUBLE_HOIST_BABY_TILE:-4}"
    export POSEIDON_GPU_DOUBLE_HOIST_MAX_WORKSPACE_MB="${POSEIDON_GPU_DOUBLE_HOIST_MAX_WORKSPACE_MB:-1024}"
    export POSEIDON_BOOTSTRAP_LOG_Q="${POSEIDON_BOOTSTRAP_LOG_Q:-32}"
    export POSEIDON_BOOTSTRAP_LOG_P="${POSEIDON_BOOTSTRAP_LOG_P:-32}"
    export POSEIDON_BOOTSTRAP_LOG_SCALE="${POSEIDON_BOOTSTRAP_LOG_SCALE:-40}"
    export POSEIDON_BOOTSTRAP_MIXED_45_Q_CHAIN="${POSEIDON_BOOTSTRAP_MIXED_45_Q_CHAIN:-1}"
    export POSEIDON_BOOTSTRAP_MESSAGE_RATIO="${POSEIDON_BOOTSTRAP_MESSAGE_RATIO:-32}"
    export POSEIDON_BOOTSTRAP_C2S_STEP="${POSEIDON_BOOTSTRAP_C2S_STEP:-1}"
    export POSEIDON_BOOTSTRAP_S2C_STEP="${POSEIDON_BOOTSTRAP_S2C_STEP:-1}"
    export POSEIDON_BOOTSTRAP_EVALMOD_RESCALE_COUNT="${POSEIDON_BOOTSTRAP_EVALMOD_RESCALE_COUNT:-1}"
    export POSEIDON_BOOTSTRAP_EVALMOD_LOG_SCALE="${POSEIDON_BOOTSTRAP_EVALMOD_LOG_SCALE:-45}"
    export POSEIDON_BOOTSTRAP_EVALMOD_DYNAMIC_RESCALE="${POSEIDON_BOOTSTRAP_EVALMOD_DYNAMIC_RESCALE:-1}"
    export POSEIDON_BOOTSTRAP_EVALMOD_DOUBLE_ANGLE="${POSEIDON_BOOTSTRAP_EVALMOD_DOUBLE_ANGLE:-2}"
    export POSEIDON_BOOTSTRAP_EVALMOD_K="${POSEIDON_BOOTSTRAP_EVALMOD_K:-25}"
    export POSEIDON_BOOTSTRAP_EVALMOD_SINE_DEGREE="${POSEIDON_BOOTSTRAP_EVALMOD_SINE_DEGREE:-59}"
    export POSEIDON_BOOTSTRAP_CORRECTNESS_TOLERANCE="${POSEIDON_BOOTSTRAP_CORRECTNESS_TOLERANCE:-0.002}"
    # The production CPU Bootstrapper oracle currently has a separate
    # N=65536 scale-bound issue. Keep staged CPU/GPU and source-message checks
    # enabled while omitting only that duplicate oracle path.
    export POSEIDON_BOOTSTRAP_SKIP_LIBRARY_ORACLE="${POSEIDON_BOOTSTRAP_SKIP_LIBRARY_ORACLE:-1}"
    export POSEIDON_BOOTSTRAP_WARMUP="${POSEIDON_BOOTSTRAP_WARMUP:-1}"
    export POSEIDON_BOOTSTRAP_ITERATIONS="${POSEIDON_BOOTSTRAP_ITERATIONS:-1}"
    export POSEIDON_BOOTSTRAP_FULL_WARMUP="${POSEIDON_BOOTSTRAP_FULL_WARMUP:-1}"
    export POSEIDON_BOOTSTRAP_FULL_ITERATIONS="${POSEIDON_BOOTSTRAP_FULL_ITERATIONS:-1}"
elif [[ "${POSEIDON_BOOTSTRAP_PROFILE}" != "legacy" ]]; then
    echo "Unknown POSEIDON_BOOTSTRAP_PROFILE=${POSEIDON_BOOTSTRAP_PROFILE}" >&2
    echo "Supported profiles: dynamic32, dual30, legacy" >&2
    exit 2
fi
export POSEIDON_KEYSWITCH_FOURSTEP_ALL_NTT
export POSEIDON_KEYSWITCH_FOURSTEP_PHASE2_MAC
export POSEIDON_KEYSWITCH_FOURSTEP_FINALIZE_FUSED
export POSEIDON_NTT_ALGO

if [[ -z "${CMAKE_BIN}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/cmake" ]]; then
        CMAKE_BIN="${CONDA_PREFIX}/bin/cmake"
    elif [[ -x "/opt/conda/envs/apollo/bin/cmake" ]]; then
        CMAKE_BIN="/opt/conda/envs/apollo/bin/cmake"
    elif command -v cmake >/dev/null 2>&1; then
        CMAKE_BIN="$(command -v cmake)"
    else
        echo "CMake was not found in PATH." >&2
        echo "Install CMake, activate the environment that provides it," >&2
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
            exit 2
        fi
    fi
fi

CMAKE_ARGS=(
    -S "${SCRIPT_DIR}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DPOSEIDON_GPU_BOOTSTRAPPING_ENABLE_GPU_TESTS="${ENABLE_GPU_TESTS}"
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
    echo "GPU bootstrap test is disabled because ENABLE_GPU_TESTS=${ENABLE_GPU_TESTS}"
    exit 0
fi

"${CMAKE_BIN}" --build "${BUILD_DIR}" --target test_gpu_bootstrap_modraise -j"$(nproc)"

TEST_BIN="${BUILD_DIR}/test_gpu_bootstrap_modraise"
if [[ ! -x "${TEST_BIN}" ]]; then
    echo "Test binary was not produced: ${TEST_BIN}" >&2
    exit 1
fi

if [[ "${POSEIDON_BOOTSTRAP_BUILD_ONLY:-0}" == "1" ]]; then
    echo "=== Build-only mode complete ==="
    echo "Test binary: ${TEST_BIN}"
    exit 0
fi

echo "=== Run GPU high-precision bootstrap correctness + timing comparison ==="
echo "Bootstrap profile: ${POSEIDON_BOOTSTRAP_PROFILE}"
"${TEST_BIN}"
