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
POSEIDON_KEYSWITCH_P9_PREWEIGHT_P="${POSEIDON_KEYSWITCH_P9_PREWEIGHT_P:-1}"
POSEIDON_KEYSWITCH_P9_P_TO_Q_ROW_TILED_8="${POSEIDON_KEYSWITCH_P9_P_TO_Q_ROW_TILED_8:-1}"
POSEIDON_KEYSWITCH_P9_FOURSTEP_P_INTT="${POSEIDON_KEYSWITCH_P9_FOURSTEP_P_INTT:-1}"
POSEIDON_KEYSWITCH_P9_FOURSTEP_QP="${POSEIDON_KEYSWITCH_P9_FOURSTEP_QP:-1}"
POSEIDON_KEYSWITCH_P9_P_TO_Q_FOURSTEP="${POSEIDON_KEYSWITCH_P9_P_TO_Q_FOURSTEP:-1}"
POSEIDON_DOUBLE_HOIST_P9_MODUP_ROW_TILED_8="${POSEIDON_DOUBLE_HOIST_P9_MODUP_ROW_TILED_8:-1}"
POSEIDON_DOUBLE_HOIST_P9_PREWEIGHT_P="${POSEIDON_DOUBLE_HOIST_P9_PREWEIGHT_P:-1}"
POSEIDON_DOUBLE_HOIST_P9_P_TO_Q_ROW_TILED_8="${POSEIDON_DOUBLE_HOIST_P9_P_TO_Q_ROW_TILED_8:-1}"
POSEIDON_DOUBLE_HOIST_P9_QP_FOURSTEP="${POSEIDON_DOUBLE_HOIST_P9_QP_FOURSTEP:-1}"
POSEIDON_DOUBLE_HOIST_P9_P_TO_Q_FOURSTEP="${POSEIDON_DOUBLE_HOIST_P9_P_TO_Q_FOURSTEP:-1}"
POSEIDON_DOUBLE_HOIST_QP_MAC_GROUP_TILED_8="${POSEIDON_DOUBLE_HOIST_QP_MAC_GROUP_TILED_8:-1}"
POSEIDON_DOUBLE_HOIST_QP_MAC_COMPONENT_FUSED="${POSEIDON_DOUBLE_HOIST_QP_MAC_COMPONENT_FUSED:-1}"
POSEIDON_DOUBLE_HOIST_QP_MAC_DIRECT_INIT="${POSEIDON_DOUBLE_HOIST_QP_MAC_DIRECT_INIT:-1}"
POSEIDON_DOUBLE_HOIST_FUSED_BABY_KEYSWITCH_C0="${POSEIDON_DOUBLE_HOIST_FUSED_BABY_KEYSWITCH_C0:-1}"
POSEIDON_DOUBLE_HOIST_DIRECT_GIANT_ACCUMULATE="${POSEIDON_DOUBLE_HOIST_DIRECT_GIANT_ACCUMULATE:-1}"
POSEIDON_DOUBLE_HOIST_BATCHED_GIANT_INTT="${POSEIDON_DOUBLE_HOIST_BATCHED_GIANT_INTT:-1}"
POSEIDON_GPU_DOUBLE_HOIST_DNUM1_BABY_TILE="${POSEIDON_GPU_DOUBLE_HOIST_DNUM1_BABY_TILE:-8}"
POSEIDON_EVALMOD_LAZY_RELIN="${POSEIDON_EVALMOD_LAZY_RELIN:-1}"
POSEIDON_BOOTSTRAP_PROFILE="${POSEIDON_BOOTSTRAP_PROFILE:-dynamic32}"

if [[ "${POSEIDON_BOOTSTRAP_PROFILE}" == "dynamic32" ||
      "${POSEIDON_BOOTSTRAP_PROFILE}" == "dual30" ||
      "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim59_da2" ||
      "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3" ||
      "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3_c2s5433" ]]; then
    # 40-bit normal CKKS scale and 45-bit bootstrap scale over GPU-friendly
    # physical primes. The mixed <=32-bit Q chain and all stage widths are
    # selected by the CPU-compatible min_scale/2 dynamic planner.
    if [[ "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim59_da2" ||
          "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3" ||
          "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3_c2s5433" ]]; then
        DEFAULT_BOOTSTRAP_DEGREE=65536
    else
        DEFAULT_BOOTSTRAP_DEGREE=16384
    fi
    export POSEIDON_BOOTSTRAP_DEGREE="${POSEIDON_BOOTSTRAP_DEGREE:-${DEFAULT_BOOTSTRAP_DEGREE}}"
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
    if [[ "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim59_da2" ]]; then
        export POSEIDON_BOOTSTRAP_SLIM_STC_EVALMOD_PROBE=1
        export POSEIDON_BOOTSTRAP_EVALMOD_SINE_DEGREE=59
        export POSEIDON_BOOTSTRAP_EVALMOD_DOUBLE_ANGLE=2
    elif [[ "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3" ||
            "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3_c2s5433" ]]; then
        export POSEIDON_BOOTSTRAP_SLIM_STC_EVALMOD_PROBE=1
        export POSEIDON_BOOTSTRAP_EVALMOD_SINE_DEGREE=22
        export POSEIDON_BOOTSTRAP_EVALMOD_GENERATION_DEGREE=59
        export POSEIDON_BOOTSTRAP_EVALMOD_TRUNCATE_DEGREE=22
        export POSEIDON_BOOTSTRAP_EVALMOD_DOUBLE_ANGLE=3
        # A degree-22 polynomial needs only degree-3 baby bases. Keep this
        # choice local to the named experiment; production and degree-59
        # profiles retain their existing automatic split.
        export POSEIDON_EVALMOD_LOG_SPLIT="${POSEIDON_EVALMOD_LOG_SPLIT:-2}"
        # T32 is not consumed by the degree-22, baby-4 polynomial DAG. Retain
        # only its virtual q-count transition for dynamic level planning.
        export POSEIDON_EVALMOD_VIRTUAL_DEGREE_BOUND="${POSEIDON_EVALMOD_VIRTUAL_DEGREE_BOUND:-1}"
        if [[ "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3_c2s5433" ]]; then
            export POSEIDON_BOOTSTRAP_SLIM_C2S_5433=1
            # Normalized post-ModRaise chain selected for the independent
            # [5,4,3,3] C2S experiment. It trades about 1.7 ms against the
            # historical fastest chain for one additional usable output Q
            # limb: C2S 34->28, EvalMod 28->13, final scale about 2^45.
            # An explicit environment value still overrides this default.
            export POSEIDON_BOOTSTRAP_Q_BIT_CHAIN="${POSEIDON_BOOTSTRAP_Q_BIT_CHAIN:-32,32,32,32,32,32,32,32,32,32,32,32,28,30,31,31,32,32,30,31,32,31,32,32,31,31,31,32,22,31,32,32,32,30}"
        fi
    fi
elif [[ "${POSEIDON_BOOTSTRAP_PROFILE}" != "legacy" ]]; then
    echo "Unknown POSEIDON_BOOTSTRAP_PROFILE=${POSEIDON_BOOTSTRAP_PROFILE}" >&2
    echo "Supported profiles: dynamic32, dual30, slim59_da2, slim22_da3, slim22_da3_c2s5433, legacy" >&2
    exit 2
fi
export POSEIDON_KEYSWITCH_FOURSTEP_ALL_NTT
export POSEIDON_KEYSWITCH_FOURSTEP_PHASE2_MAC
export POSEIDON_KEYSWITCH_FOURSTEP_FINALIZE_FUSED
export POSEIDON_KEYSWITCH_P9_PREWEIGHT_P
export POSEIDON_KEYSWITCH_P9_P_TO_Q_ROW_TILED_8
export POSEIDON_KEYSWITCH_P9_FOURSTEP_P_INTT
export POSEIDON_KEYSWITCH_P9_FOURSTEP_QP
export POSEIDON_KEYSWITCH_P9_P_TO_Q_FOURSTEP
export POSEIDON_DOUBLE_HOIST_P9_MODUP_ROW_TILED_8
export POSEIDON_DOUBLE_HOIST_P9_PREWEIGHT_P
export POSEIDON_DOUBLE_HOIST_P9_P_TO_Q_ROW_TILED_8
export POSEIDON_DOUBLE_HOIST_P9_QP_FOURSTEP
export POSEIDON_DOUBLE_HOIST_P9_P_TO_Q_FOURSTEP
export POSEIDON_DOUBLE_HOIST_QP_MAC_GROUP_TILED_8
export POSEIDON_DOUBLE_HOIST_QP_MAC_COMPONENT_FUSED
export POSEIDON_DOUBLE_HOIST_QP_MAC_DIRECT_INIT
export POSEIDON_DOUBLE_HOIST_FUSED_BABY_KEYSWITCH_C0
export POSEIDON_DOUBLE_HOIST_DIRECT_GIANT_ACCUMULATE
export POSEIDON_DOUBLE_HOIST_BATCHED_GIANT_INTT
export POSEIDON_GPU_DOUBLE_HOIST_DNUM1_BABY_TILE
export POSEIDON_EVALMOD_LAZY_RELIN
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
if [[ -n "${POSEIDON_EVALMOD_LOG_SPLIT:-}" ]]; then
    echo "[WARN] Experimental EvalMod split: log_split=${POSEIDON_EVALMOD_LOG_SPLIT}, baby_width=$((1 << POSEIDON_EVALMOD_LOG_SPLIT))"
    if [[ "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3" ||
          "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3_c2s5433" ]]; then
        echo "       baby_width=4 is the named slim22_da3 default; an explicit value can override it for A/B experiments."
    else
        echo "       Unset POSEIDON_EVALMOD_LOG_SPLIT to restore the automatic/default split."
    fi
fi
if [[ "${POSEIDON_EVALMOD_FLAT_BSGS_B8:-0}" != "0" ]]; then
    echo "[WARN] Experimental EvalMod layout: flat_b8 (L0 + L1*T8 + ... + L7*T56)"
    echo "       Unset POSEIDON_EVALMOD_FLAT_BSGS_B8 to restore recursive Chebyshev BSGS."
fi
if [[ "${POSEIDON_BOOTSTRAP_SLIM_STC_FIRST_PROBE:-0}" != "0" ]]; then
    echo "[WARN] Experimental slim StC-first phase-1 probe: low-level StC only."
    echo "       The production bootstrap schedule is unchanged; unset POSEIDON_BOOTSTRAP_SLIM_STC_FIRST_PROBE to run it."
fi
if [[ "${POSEIDON_BOOTSTRAP_SLIM_STC_MODRAISE_PROBE:-0}" != "0" ]]; then
    echo "[WARN] Experimental slim StC-first phase-2 probe: low-level StC followed by ModRaise."
    echo "       The production bootstrap schedule is unchanged; unset POSEIDON_BOOTSTRAP_SLIM_STC_MODRAISE_PROBE to run it."
fi
if [[ "${POSEIDON_BOOTSTRAP_SLIM_STC_C2S_PROBE:-0}" != "0" ]]; then
    echo "[WARN] Experimental slim StC-first phase-3 probe: low-level StC, ModRaise, then C2S."
    echo "       The production bootstrap schedule is unchanged; unset POSEIDON_BOOTSTRAP_SLIM_STC_C2S_PROBE to run it."
fi
if [[ "${POSEIDON_BOOTSTRAP_SLIM_STC_EVALMOD_PROBE:-0}" != "0" ]]; then
    echo "[WARN] Experimental slim StC-first phase-4 probe: full StC, ModRaise, C2S, EvalMod schedule."
    echo "       Active message_ratio=${POSEIDON_BOOTSTRAP_MESSAGE_RATIO}; the dynamic32 profile keeps the production default at 32."
    echo "       The production bootstrap schedule is unchanged; unset POSEIDON_BOOTSTRAP_SLIM_STC_EVALMOD_PROBE to run it."
fi
if [[ "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim59_da2" ||
      "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3" ||
      "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3_c2s5433" ]]; then
    echo "[WARN] Named StC-first profile ${POSEIDON_BOOTSTRAP_PROFILE}: degree=${POSEIDON_BOOTSTRAP_EVALMOD_SINE_DEGREE}, double_angle=${POSEIDON_BOOTSTRAP_EVALMOD_DOUBLE_ANGLE}."
    echo "       dynamic32 remains the production profile; the two StC-first variants are independent paths."
fi
if [[ "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3" ||
      "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3_c2s5433" ]]; then
    echo "       slim22_da3 is a Chebyshev truncation experiment: generate degree=${POSEIDON_BOOTSTRAP_EVALMOD_GENERATION_DEGREE}, then retain T0..T${POSEIDON_BOOTSTRAP_EVALMOD_TRUNCATE_DEGREE}."
    echo "       Its default polynomial split is baby_width=4."
    echo "       Degree-22 lazy relinearization is enabled when graph-level/scale eligibility checks pass."
    if [[ "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3_c2s5433" ]]; then
        echo "       C2S uses independent [5,4,3,3] fused-layer groups; the two 3-layer matrices use a direct single-giant-group plan."
        echo "       Normalized Q chain is the profile default: C2S 34->28, EvalMod 28->13, final scale approximately 2^45."
        echo "       This keeps one more usable output Q limb than the former 34->12 chain; set POSEIDON_BOOTSTRAP_Q_BIT_CHAIN explicitly for A/B experiments."
    fi
    if [[ "${POSEIDON_EVALMOD_VIRTUAL_DEGREE_BOUND}" != "0" ]]; then
        echo "       The unused T32 degree-bound basis is planning-only and is not executed on GPU."
    else
        echo "       Diagnostic override: build the legacy T32 setup plan; final liveness pruning still removes it from GPU execution."
    fi
    echo "       It is not a degree-22 Remez refit; poor approximation accuracy is expected and remains a test failure."
fi
if [[ "${POSEIDON_BOOTSTRAP_PLAINTEXT_COMPRESSION_PROBE:-0}" != "0" ]]; then
    echo "[WARN] Experimental WHET plaintext-compression probe: build compact QP device storage and verify it bit-for-bit."
    echo "       Compressed plaintexts are not consumed by a compute kernel in this probe."
fi
if [[ "${POSEIDON_BOOTSTRAP_COMPRESSED_QP_MAC_PROBE:-0}" != "0" ]]; then
    echo "[WARN] Experimental compact-QP MAC A/B probe: full and compressed plaintext paths both run."
    echo "       The default bootstrap path is unchanged; Q/P MAC residues and CtS/StC outputs must be bit-exact."
fi
echo "[WARN] Double-Hoist QP-MAC defaults: direct accumulator initialization=${POSEIDON_DOUBLE_HOIST_QP_MAC_DIRECT_INIT}, dnum=1 baby tile=${POSEIDON_GPU_DOUBLE_HOIST_DNUM1_BABY_TILE}."
echo "       Set POSEIDON_DOUBLE_HOIST_QP_MAC_DIRECT_INIT=0 and POSEIDON_GPU_DOUBLE_HOIST_DNUM1_BABY_TILE=0 to restore zero-fill and the global baby-tile setting."
echo "       dnum=1 baby KeySwitch+c0 fusion=${POSEIDON_DOUBLE_HOIST_FUSED_BABY_KEYSWITCH_C0}; set it to 0 to restore the separate c0 kernel."
echo "       N=65536 giant-source batched INTT=${POSEIDON_DOUBLE_HOIST_BATCHED_GIANT_INTT}; set it to 0 to restore per-group INTT launches."
"${TEST_BIN}"
