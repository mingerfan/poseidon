#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSEIDON_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
TEST_BIN="${BUILD_DIR}/test_gpu_bootstrap_modraise"
REPORT_DIR="${NSYS_REPORT_DIR:-${SCRIPT_DIR}/profiles}"
REPORT_NAME="${NSYS_REPORT_NAME:-bootstrap_launch_$(date +%Y%m%d_%H%M%S)}"
REPORT_PREFIX="${REPORT_DIR}/${REPORT_NAME}"
CHECK_ONLY="${POSEIDON_NSYS_CHECK_ONLY:-0}"
SKIP_BUILD="${POSEIDON_NSYS_SKIP_BUILD:-0}"
CMAKE_BIN="${CMAKE_BIN:-}"

fail()
{
    echo "[FAILED] $*" >&2
    exit 1
}

require_command()
{
    local command_name="$1"
    local install_hint="$2"
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        fail "Required command '${command_name}' was not found. ${install_hint}"
    fi
    echo "[OK] ${command_name}: $(command -v "${command_name}")"
}

version_at_least()
{
    local actual="$1"
    local required="$2"
    [[ "$(printf '%s\n%s\n' "${required}" "${actual}" | sort -V | head -n 1)" == "${required}" ]]
}

write_stats_report()
{
    local report="$1"
    local output_file="$2"
    if nsys stats --report "${report}" "${REPORT_PREFIX}.nsys-rep" \
        >"${output_file}" 2>&1; then
        echo "[OK] ${report}: ${output_file}"
    else
        echo "[WARN] Nsight Systems could not generate '${report}'." >&2
        echo "       See ${output_file} for the diagnostic output." >&2
    fi
}

echo "=== Nsight Systems environment check ==="

[[ "$(uname -s)" == "Linux" ]] ||
    fail "This script currently supports the Linux Nsight Systems CLI only."
echo "[OK] operating system: $(uname -sr)"

require_command nsys \
    "Install NVIDIA Nsight Systems or add its bin directory to PATH."
require_command nvidia-smi \
    "Install/configure the NVIDIA driver and expose the GPU to this shell."
require_command sort "Install GNU coreutils."
require_command nproc "Install GNU coreutils."

if [[ -z "${CMAKE_BIN}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/cmake" ]]; then
        CMAKE_BIN="${CONDA_PREFIX}/bin/cmake"
    elif [[ -x "/opt/conda/envs/apollo/bin/cmake" ]]; then
        CMAKE_BIN="/opt/conda/envs/apollo/bin/cmake"
    elif command -v cmake >/dev/null 2>&1; then
        CMAKE_BIN="$(command -v cmake)"
    else
        fail "CMake was not found. Activate the Poseidon environment or set CMAKE_BIN=/absolute/path/to/cmake."
    fi
fi
[[ -x "${CMAKE_BIN}" ]] || fail "CMAKE_BIN is not executable: ${CMAKE_BIN}"

CMAKE_VERSION="$(${CMAKE_BIN} --version | sed -n '1s/.* //p')"
[[ -n "${CMAKE_VERSION}" ]] || fail "Unable to determine the CMake version."
version_at_least "${CMAKE_VERSION}" "3.26.4" ||
    fail "CMake ${CMAKE_VERSION} is too old; this test requires CMake >= 3.26.4."
echo "[OK] cmake: ${CMAKE_BIN} (${CMAKE_VERSION})"

if [[ -n "${CUDACXX:-}" ]]; then
    [[ -x "${CUDACXX}" ]] || fail "CUDACXX is not executable: ${CUDACXX}"
    CUDA_COMPILER="${CUDACXX}"
elif command -v nvcc >/dev/null 2>&1; then
    CUDA_COMPILER="$(command -v nvcc)"
elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
else
    fail "nvcc was not found. Load the CUDA toolkit module or set CUDACXX/CUDA_HOME."
fi
echo "[OK] CUDA compiler: ${CUDA_COMPILER}"
"${CUDA_COMPILER}" --version | tail -n 1

GPU_LIST="$(nvidia-smi -L 2>&1)" ||
    fail "nvidia-smi cannot access a GPU: ${GPU_LIST}"
[[ -n "${GPU_LIST}" ]] || fail "No NVIDIA GPU is visible to nvidia-smi."
echo "[OK] visible GPU(s):"
printf '%s\n' "${GPU_LIST}" | sed 's/^/     /'

NSYS_VERSION_OUTPUT="$(nsys --version 2>&1 || true)"
NSYS_VERSION="$(
    sed -n '/NVIDIA Nsight Systems version/{p;q;}' \
        <<<"${NSYS_VERSION_OUTPUT}"
)"
[[ -n "${NSYS_VERSION}" ]] || NSYS_VERSION="${NSYS_VERSION_OUTPUT}"
echo "[OK] Nsight Systems: ${NSYS_VERSION}"
# Nsight Systems 2023.x prints this help successfully but returns status 1.
# Inspect the text rather than treating that legacy exit status as a failure.
NSYS_REPORT_HELP="$(nsys stats --help-reports 2>/dev/null || true)"
if [[ "${NSYS_REPORT_HELP}" != *"cuda_kern_exec_sum"* ]]; then
    fail "This nsys installation does not provide the cuda_kern_exec_sum report required for launch-latency analysis."
fi

echo "--- Nsight profiling capability ---"
nsys status --environment ||
    fail "Nsight Systems reports that the profiling environment is unavailable."

REQUIRED_FILES=(
    "${SCRIPT_DIR}/run.sh"
    "${SCRIPT_DIR}/CMakeLists.txt"
    "${SCRIPT_DIR}/test_gpu_bootstrap_modraise.cpp"
    "${POSEIDON_ROOT}/CMakeLists.txt"
    "${POSEIDON_ROOT}/third_party/rmm/CMakeLists.txt"
    "${POSEIDON_ROOT}/third_party/rmm/cmake/thirdparty/get_spdlog.cmake"
    "${POSEIDON_ROOT}/third_party/rmm/cmake/thirdparty/get_cccl.cmake"
    "${POSEIDON_ROOT}/third_party/rmm/cmake/thirdparty/get_nvtx.cmake"
    "${POSEIDON_ROOT}/third_party/rapids-cmake/rapids-cmake/export/template/build_package.cmake.in"
)
for required_file in "${REQUIRED_FILES[@]}"; do
    [[ -f "${required_file}" ]] ||
        fail "Required repository file is missing: ${required_file}. Check that bundled third_party files were committed and pulled."
done
echo "[OK] Poseidon test sources and bundled RMM/RAPIDS build files are present."

if [[ "${CHECK_ONLY}" == "1" ]]; then
    echo "=== Environment check passed (check-only mode) ==="
    exit 0
fi

# Profile a single full GPU bootstrap. The test binary still performs setup and
# one warmup outside the profiler capture range, so the Nsight timeline starts
# at the measured bootstrap rather than at CPU matrix/key preparation.
export POSEIDON_BOOTSTRAP_PROFILE="${POSEIDON_BOOTSTRAP_PROFILE:-dynamic32}"
export POSEIDON_NTT_ALGO="${POSEIDON_NTT_ALGO:-fourstep}"
export POSEIDON_KEYSWITCH_FOURSTEP_ALL_NTT="${POSEIDON_KEYSWITCH_FOURSTEP_ALL_NTT:-1}"
export POSEIDON_KEYSWITCH_FOURSTEP_PHASE2_MAC="${POSEIDON_KEYSWITCH_FOURSTEP_PHASE2_MAC:-1}"
export POSEIDON_KEYSWITCH_FOURSTEP_FINALIZE_FUSED="${POSEIDON_KEYSWITCH_FOURSTEP_FINALIZE_FUSED:-1}"
export POSEIDON_KEYSWITCH_P9_PREWEIGHT_P="${POSEIDON_KEYSWITCH_P9_PREWEIGHT_P:-1}"
export POSEIDON_KEYSWITCH_P9_P_TO_Q_ROW_TILED_8="${POSEIDON_KEYSWITCH_P9_P_TO_Q_ROW_TILED_8:-1}"
export POSEIDON_KEYSWITCH_P9_FOURSTEP_P_INTT="${POSEIDON_KEYSWITCH_P9_FOURSTEP_P_INTT:-1}"
export POSEIDON_KEYSWITCH_P9_FOURSTEP_QP="${POSEIDON_KEYSWITCH_P9_FOURSTEP_QP:-1}"
export POSEIDON_KEYSWITCH_P9_P_TO_Q_FOURSTEP="${POSEIDON_KEYSWITCH_P9_P_TO_Q_FOURSTEP:-1}"
export POSEIDON_DOUBLE_HOIST_P9_MODUP_ROW_TILED_8="${POSEIDON_DOUBLE_HOIST_P9_MODUP_ROW_TILED_8:-1}"
export POSEIDON_DOUBLE_HOIST_P9_PREWEIGHT_P="${POSEIDON_DOUBLE_HOIST_P9_PREWEIGHT_P:-1}"
export POSEIDON_DOUBLE_HOIST_P9_P_TO_Q_ROW_TILED_8="${POSEIDON_DOUBLE_HOIST_P9_P_TO_Q_ROW_TILED_8:-1}"
export POSEIDON_DOUBLE_HOIST_P9_QP_FOURSTEP="${POSEIDON_DOUBLE_HOIST_P9_QP_FOURSTEP:-1}"
export POSEIDON_DOUBLE_HOIST_P9_P_TO_Q_FOURSTEP="${POSEIDON_DOUBLE_HOIST_P9_P_TO_Q_FOURSTEP:-1}"
export POSEIDON_DOUBLE_HOIST_QP_MAC_GROUP_TILED_8="${POSEIDON_DOUBLE_HOIST_QP_MAC_GROUP_TILED_8:-1}"
export POSEIDON_DOUBLE_HOIST_QP_MAC_COMPONENT_FUSED="${POSEIDON_DOUBLE_HOIST_QP_MAC_COMPONENT_FUSED:-1}"
export POSEIDON_DOUBLE_HOIST_QP_MAC_DIRECT_INIT="${POSEIDON_DOUBLE_HOIST_QP_MAC_DIRECT_INIT:-1}"
export POSEIDON_DOUBLE_HOIST_FUSED_BABY_KEYSWITCH_C0="${POSEIDON_DOUBLE_HOIST_FUSED_BABY_KEYSWITCH_C0:-1}"
export POSEIDON_DOUBLE_HOIST_FUSED_BABY_KEYSWITCH_PLAIN_MAC="${POSEIDON_DOUBLE_HOIST_FUSED_BABY_KEYSWITCH_PLAIN_MAC:-1}"
export POSEIDON_DOUBLE_HOIST_FUSED_BABY_BLOCK_SIZE="${POSEIDON_DOUBLE_HOIST_FUSED_BABY_BLOCK_SIZE:-128}"
export POSEIDON_DOUBLE_HOIST_DIRECT_GIANT_ACCUMULATE="${POSEIDON_DOUBLE_HOIST_DIRECT_GIANT_ACCUMULATE:-1}"
export POSEIDON_DOUBLE_HOIST_BATCHED_GIANT_INTT="${POSEIDON_DOUBLE_HOIST_BATCHED_GIANT_INTT:-1}"
export POSEIDON_GPU_DOUBLE_HOIST_DNUM1_BABY_TILE="${POSEIDON_GPU_DOUBLE_HOIST_DNUM1_BABY_TILE:-8}"
export POSEIDON_EVALMOD_LAZY_RELIN="${POSEIDON_EVALMOD_LAZY_RELIN:-1}"
export POSEIDON_EVALMOD_D2D_FREE_DATAFLOW="${POSEIDON_EVALMOD_D2D_FREE_DATAFLOW:-1}"
export POSEIDON_EVALMOD_ZERO_COPY_MODDROP="${POSEIDON_EVALMOD_ZERO_COPY_MODDROP:-1}"
export POSEIDON_EVALMOD_Q_PREFIX_VIEWS="${POSEIDON_EVALMOD_Q_PREFIX_VIEWS:-1}"
export POSEIDON_RELIN_RESCALE_X2="${POSEIDON_RELIN_RESCALE_X2:-1}"
export POSEIDON_GPU_LINEAR_TRANSFORM_MODE="${POSEIDON_GPU_LINEAR_TRANSFORM_MODE:-double_hoist}"
DEFAULT_DOUBLE_HOIST_BABY_TILE=4
if [[ "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3_c2s5433" ||
      "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_direct_da3_c2s5433" ]]; then
    DEFAULT_DOUBLE_HOIST_BABY_TILE=15
fi
export POSEIDON_GPU_DOUBLE_HOIST_BABY_TILE="${POSEIDON_GPU_DOUBLE_HOIST_BABY_TILE:-${DEFAULT_DOUBLE_HOIST_BABY_TILE}}"
export POSEIDON_GPU_DOUBLE_HOIST_MAX_WORKSPACE_MB="${POSEIDON_GPU_DOUBLE_HOIST_MAX_WORKSPACE_MB:-1024}"
export POSEIDON_BOOTSTRAP_DEGREE="${POSEIDON_BOOTSTRAP_DEGREE:-65536}"
export POSEIDON_BOOTSTRAP_Q_COUNT="${POSEIDON_BOOTSTRAP_Q_COUNT:-34}"
export POSEIDON_BOOTSTRAP_P_COUNT="${POSEIDON_BOOTSTRAP_P_COUNT:-9}"
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
export POSEIDON_BOOTSTRAP_SKIP_LIBRARY_ORACLE="${POSEIDON_BOOTSTRAP_SKIP_LIBRARY_ORACLE:-1}"
export POSEIDON_BOOTSTRAP_GPU_ONLY_TIMING="${POSEIDON_BOOTSTRAP_GPU_ONLY_TIMING:-1}"
export POSEIDON_BOOTSTRAP_NSYS_CAPTURE_FULL="${POSEIDON_BOOTSTRAP_NSYS_CAPTURE_FULL:-1}"
export POSEIDON_BOOTSTRAP_NSYS_KEYSWITCH_DETAIL="${POSEIDON_BOOTSTRAP_NSYS_KEYSWITCH_DETAIL:-1}"
export POSEIDON_BOOTSTRAP_IGNORE_CORRECTNESS_FAILURE="${POSEIDON_BOOTSTRAP_IGNORE_CORRECTNESS_FAILURE:-1}"
export POSEIDON_BOOTSTRAP_ITERATIONS="${POSEIDON_BOOTSTRAP_ITERATIONS:-1}"
export POSEIDON_BOOTSTRAP_WARMUP="${POSEIDON_BOOTSTRAP_WARMUP:-0}"
export POSEIDON_BOOTSTRAP_FULL_ITERATIONS="${POSEIDON_BOOTSTRAP_FULL_ITERATIONS:-1}"
export POSEIDON_BOOTSTRAP_FULL_WARMUP="${POSEIDON_BOOTSTRAP_FULL_WARMUP:-1}"
export POSEIDON_BOOTSTRAP_STAGE_PROFILE_ITERATIONS="${POSEIDON_BOOTSTRAP_STAGE_PROFILE_ITERATIONS:-1}"

# run.sh applies named-profile defaults inside its own process. The profiler
# launches TEST_BIN directly after the build, so reproduce the profile-derived
# runtime settings here instead of losing them at that process boundary.
case "${POSEIDON_BOOTSTRAP_PROFILE}" in
    slim59_da2)
        export POSEIDON_BOOTSTRAP_SLIM_STC_EVALMOD_PROBE=1
        export POSEIDON_BOOTSTRAP_EVALMOD_SINE_DEGREE=59
        export POSEIDON_BOOTSTRAP_EVALMOD_DOUBLE_ANGLE=2
        ;;
    slim22_da3|slim22_da3_c2s5433|slim22_direct_da3_c2s5433)
        export POSEIDON_BOOTSTRAP_SLIM_STC_EVALMOD_PROBE=1
        export POSEIDON_BOOTSTRAP_EVALMOD_SINE_DEGREE=22
        if [[ "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_direct_da3_c2s5433" ]]; then
            export POSEIDON_BOOTSTRAP_EVALMOD_GENERATION_DEGREE=22
            export POSEIDON_BOOTSTRAP_EVALMOD_FIXED_DEGREE_REFIT=1
            unset POSEIDON_BOOTSTRAP_EVALMOD_TRUNCATE_DEGREE
        else
            export POSEIDON_BOOTSTRAP_EVALMOD_GENERATION_DEGREE=59
            export POSEIDON_BOOTSTRAP_EVALMOD_TRUNCATE_DEGREE=22
        fi
        export POSEIDON_BOOTSTRAP_EVALMOD_DOUBLE_ANGLE=3
        export POSEIDON_EVALMOD_LOG_SPLIT="${POSEIDON_EVALMOD_LOG_SPLIT:-2}"
        export POSEIDON_EVALMOD_VIRTUAL_DEGREE_BOUND="${POSEIDON_EVALMOD_VIRTUAL_DEGREE_BOUND:-1}"
        if [[ "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_da3_c2s5433" ||
              "${POSEIDON_BOOTSTRAP_PROFILE}" == "slim22_direct_da3_c2s5433" ]]; then
            export POSEIDON_BOOTSTRAP_SLIM_C2S_5433=1
            export POSEIDON_BOOTSTRAP_Q_BIT_CHAIN="${POSEIDON_BOOTSTRAP_Q_BIT_CHAIN:-32,32,32,32,32,32,32,32,32,32,32,32,28,30,31,31,32,32,30,31,32,31,32,32,31,31,31,32,22,31,32,32,32,30}"
        fi
        ;;
esac

if [[ "${SKIP_BUILD}" != "1" ]]; then
    echo "=== Configure and build profiler target ==="
    CMAKE_BIN="${CMAKE_BIN}" \
    BUILD_DIR="${BUILD_DIR}" \
    POSEIDON_BOOTSTRAP_BUILD_ONLY=1 \
        "${SCRIPT_DIR}/run.sh"
fi

[[ -x "${TEST_BIN}" ]] ||
    fail "The test binary is missing or not executable: ${TEST_BIN}. Run without POSEIDON_NSYS_SKIP_BUILD=1 first."

mkdir -p "${REPORT_DIR}"
[[ -w "${REPORT_DIR}" ]] || fail "Report directory is not writable: ${REPORT_DIR}"

echo "=== Nsight Systems bootstrap profile ==="
echo "test binary : ${TEST_BIN}"
echo "report       : ${REPORT_PREFIX}.nsys-rep"
echo "parameters   : N=${POSEIDON_BOOTSTRAP_DEGREE}, Q=${POSEIDON_BOOTSTRAP_Q_COUNT}, P=${POSEIDON_BOOTSTRAP_P_COUNT}"
echo "mode         : ${POSEIDON_GPU_LINEAR_TRANSFORM_MODE}"
echo "baby tile    : ${POSEIDON_GPU_DOUBLE_HOIST_BABY_TILE} (full-baby default for [5,4,3,3] C2S is 15)"
echo "iterations   : setup checks outside capture; full warmup=${POSEIDON_BOOTSTRAP_FULL_WARMUP}; captured full=${POSEIDON_BOOTSTRAP_FULL_ITERATIONS}"
echo "capture      : cudaProfilerStart/Stop around one full GPU bootstrap"
echo "note         : profiler-instrumented timing is not a release benchmark"

nsys profile \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --flush-on-cudaprofilerstop=true \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output="${REPORT_PREFIX}" \
    "${TEST_BIN}"

[[ -f "${REPORT_PREFIX}.nsys-rep" ]] ||
    fail "Nsight Systems did not produce ${REPORT_PREFIX}.nsys-rep"

echo "=== Generate text summaries ==="
write_stats_report \
    cuda_kern_exec_sum \
    "${REPORT_PREFIX}.kernel_launch_summary.txt"
write_stats_report \
    cuda_kern_exec_trace \
    "${REPORT_PREFIX}.kernel_launch_trace.txt"
write_stats_report \
    cuda_gpu_kern_sum \
    "${REPORT_PREFIX}.kernel_summary.txt"
write_stats_report \
    cuda_api_sum \
    "${REPORT_PREFIX}.cuda_api_summary.txt"
write_stats_report \
    cuda_api_trace \
    "${REPORT_PREFIX}.cuda_api_trace.txt"
write_stats_report \
    nvtx_sum \
    "${REPORT_PREFIX}.nvtx_summary.txt"
write_stats_report \
    nvtx_kern_sum \
    "${REPORT_PREFIX}.nvtx_gpu_summary.txt"

echo "=== Nsight Systems profile complete ==="
echo "Open the timeline with:"
echo "  nsys-ui ${REPORT_PREFIX}.nsys-rep"
echo "Launch-latency summary:"
echo "  ${REPORT_PREFIX}.kernel_launch_summary.txt"
echo "Per-launch trace:"
echo "  ${REPORT_PREFIX}.kernel_launch_trace.txt"
