#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSEIDON_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
TEST_BIN="${BUILD_DIR}/test_gpu_bootstrap_modraise"
REPORT_DIR="${NSYS_REPORT_DIR:-${SCRIPT_DIR}/nsys_reports}"
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
    if command -v cmake >/dev/null 2>&1; then
        CMAKE_BIN="$(command -v cmake)"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/cmake" ]]; then
        CMAKE_BIN="${CONDA_PREFIX}/bin/cmake"
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

NSYS_VERSION="$(nsys --version 2>&1 | head -n 1)"
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

# Match run.sh's correctness-first dual-30 profile while keeping the profile
# short enough for a practical Nsight Systems report. Every value remains
# overridable by the caller's environment.
export POSEIDON_BOOTSTRAP_PROFILE="${POSEIDON_BOOTSTRAP_PROFILE:-dual30}"
export POSEIDON_NTT_ALGO="${POSEIDON_NTT_ALGO:-fourstep}"
export POSEIDON_KEYSWITCH_FOURSTEP_ALL_NTT="${POSEIDON_KEYSWITCH_FOURSTEP_ALL_NTT:-1}"
export POSEIDON_KEYSWITCH_FOURSTEP_PHASE2_MAC="${POSEIDON_KEYSWITCH_FOURSTEP_PHASE2_MAC:-1}"
export POSEIDON_KEYSWITCH_FOURSTEP_FINALIZE_FUSED="${POSEIDON_KEYSWITCH_FOURSTEP_FINALIZE_FUSED:-1}"
export POSEIDON_GPU_LINEAR_TRANSFORM_MODE="${POSEIDON_GPU_LINEAR_TRANSFORM_MODE:-double_hoist}"
export POSEIDON_GPU_DOUBLE_HOIST_BABY_TILE="${POSEIDON_GPU_DOUBLE_HOIST_BABY_TILE:-4}"
export POSEIDON_GPU_DOUBLE_HOIST_MAX_WORKSPACE_MB="${POSEIDON_GPU_DOUBLE_HOIST_MAX_WORKSPACE_MB:-1024}"
export POSEIDON_BOOTSTRAP_DEGREE="${POSEIDON_BOOTSTRAP_DEGREE:-16384}"
export POSEIDON_BOOTSTRAP_Q_COUNT="${POSEIDON_BOOTSTRAP_Q_COUNT:-34}"
export POSEIDON_BOOTSTRAP_P_COUNT="${POSEIDON_BOOTSTRAP_P_COUNT:-9}"
export POSEIDON_BOOTSTRAP_LOG_Q="${POSEIDON_BOOTSTRAP_LOG_Q:-30}"
export POSEIDON_BOOTSTRAP_LOG_P="${POSEIDON_BOOTSTRAP_LOG_P:-30}"
export POSEIDON_BOOTSTRAP_LOG_SCALE="${POSEIDON_BOOTSTRAP_LOG_SCALE:-30}"
export POSEIDON_BOOTSTRAP_MESSAGE_RATIO="${POSEIDON_BOOTSTRAP_MESSAGE_RATIO:-32}"
export POSEIDON_BOOTSTRAP_C2S_STEP="${POSEIDON_BOOTSTRAP_C2S_STEP:-2}"
export POSEIDON_BOOTSTRAP_S2C_STEP="${POSEIDON_BOOTSTRAP_S2C_STEP:-2}"
export POSEIDON_BOOTSTRAP_EVALMOD_RESCALE_COUNT="${POSEIDON_BOOTSTRAP_EVALMOD_RESCALE_COUNT:-2}"
export POSEIDON_BOOTSTRAP_EVALMOD_LOG_SCALE="${POSEIDON_BOOTSTRAP_EVALMOD_LOG_SCALE:-60}"
export POSEIDON_BOOTSTRAP_EVALMOD_DOUBLE_ANGLE="${POSEIDON_BOOTSTRAP_EVALMOD_DOUBLE_ANGLE:-2}"
export POSEIDON_BOOTSTRAP_EVALMOD_K="${POSEIDON_BOOTSTRAP_EVALMOD_K:-25}"
export POSEIDON_BOOTSTRAP_EVALMOD_SINE_DEGREE="${POSEIDON_BOOTSTRAP_EVALMOD_SINE_DEGREE:-59}"
export POSEIDON_BOOTSTRAP_ITERATIONS="${POSEIDON_BOOTSTRAP_ITERATIONS:-1}"
export POSEIDON_BOOTSTRAP_WARMUP="${POSEIDON_BOOTSTRAP_WARMUP:-0}"
export POSEIDON_BOOTSTRAP_FULL_ITERATIONS="${POSEIDON_BOOTSTRAP_FULL_ITERATIONS:-1}"
export POSEIDON_BOOTSTRAP_FULL_WARMUP="${POSEIDON_BOOTSTRAP_FULL_WARMUP:-0}"
export POSEIDON_BOOTSTRAP_STAGE_PROFILE_ITERATIONS="${POSEIDON_BOOTSTRAP_STAGE_PROFILE_ITERATIONS:-1}"

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
echo "iterations   : op=${POSEIDON_BOOTSTRAP_ITERATIONS}, full=${POSEIDON_BOOTSTRAP_FULL_ITERATIONS}, stage=${POSEIDON_BOOTSTRAP_STAGE_PROFILE_ITERATIONS}"
echo "note         : profiler-instrumented timing is not a release benchmark"

nsys profile \
    --trace=cuda,nvtx \
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
    nvtx_gpu_proj_sum \
    "${REPORT_PREFIX}.nvtx_gpu_summary.txt"

echo "=== Nsight Systems profile complete ==="
echo "Open the timeline with:"
echo "  nsys-ui ${REPORT_PREFIX}.nsys-rep"
echo "Launch-latency summary:"
echo "  ${REPORT_PREFIX}.kernel_launch_summary.txt"
echo "Per-launch trace:"
echo "  ${REPORT_PREFIX}.kernel_launch_trace.txt"
