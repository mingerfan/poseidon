#!/usr/bin/env bash
# Execute on 188Server; the client only transfers sources and retrieves logs.
set -euo pipefail
task_base=/home/xuming/poseidon-framework-validation-20261015-6dc8cbfb
task_logs="$task_base/logs"
cmake_bin=/home/xuming/poseidon-tools/cmake/cmake/data/bin/cmake
rapids_dir=/home/xuming/poseidon/build-runtime-gpu-rmm2412/_deps/rapids-cmake-src/rapids-cmake
trap 'status=$?; printf "%s\n" "$status" > "$task_logs/exit-status"' EXIT
printf 'REMOTE_OFFLINE_DEPENDENCIES_START\n'
{
  "$cmake_bin" -S "$task_base/src/third_party/fmt" -B "$task_base/deps/fmt-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$task_base/deps/install" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DFMT_TEST=OFF -DFMT_DOC=OFF -DFMT_INSTALL=ON
  "$cmake_bin" --build "$task_base/deps/fmt-build" --target install -j4
  "$cmake_bin" -S "$task_base/src/third_party/spdlog" -B "$task_base/deps/spdlog-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$task_base/deps/install" \
    -DCMAKE_PREFIX_PATH="$task_base/deps/install" -DSPDLOG_FMT_EXTERNAL=ON \
    -DSPDLOG_BUILD_PIC=ON -DSPDLOG_BUILD_EXAMPLE=OFF -DSPDLOG_BUILD_TESTS=OFF \
    -DSPDLOG_INSTALL=ON
  "$cmake_bin" --build "$task_base/deps/spdlog-build" --target install -j4
} > "$task_logs/offline-dependencies.log" 2>&1
printf 'REMOTE_CONFIGURE_START\n'
"$cmake_bin" -S "$task_base/src" -B "$task_base/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_PREFIX_PATH="$task_base/deps/install" \
  -DPOSEIDON_BUILD_EXAMPLES=OFF -DPOSEIDON_BUILD_BENCH=OFF \
  -DPOSEIDON_BUILD_DEPS=OFF -DPOSEIDON_USE_ZLIB=OFF -DPOSEIDON_USE_ZSTD=OFF \
  -DPOSEIDON_USE_MSGSL=OFF -DPOSEIDON_USE_INTEL_HEXL=OFF \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_API=ON -DPOSEIDON_BUILD_CKKS_RUNTIME_TESTS=ON \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_GPU_TESTS=ON \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_GPU_NCCL=OFF -DPOSEIDON_BUILD_CKKS_RUNTIME_MPI=OFF \
  -DCMAKE_MODULE_PATH="$rapids_dir" -Drapids-cmake-dir="$rapids_dir" \
  -DCPM_fmt_SOURCE="$task_base/src/third_party/fmt" \
  -DCPM_spdlog_SOURCE="$task_base/src/third_party/spdlog" \
  -DCPM_CCCL_SOURCE="$task_base/src/third_party/cccl" \
  -DPOSEIDON_BUNDLED_RMM_DIR=/home/xuming/deps/rmm-v24.12.01 \
  > "$task_logs/configure.log" 2>&1
printf 'REMOTE_BUILD_START\n'
/usr/bin/time -v "$cmake_bin" --build "$task_base/build" \
  --target poseidon_runtime_gpu_api_tests poseidon_runtime_gpu_prepare_oom_tests -j12 \
  > "$task_logs/build.log" 2>&1
printf 'REMOTE_BUILD_COMPLETE\n'
sha256sum "$task_base/build/bin/poseidon_runtime_gpu_api_tests" \
  "$task_base/build/bin/poseidon_runtime_gpu_prepare_oom_tests" \
  "$task_base/src/src/poseidon/gpu/kernels/gpu_ntt_kernels.cu" \
  "$task_base/src/src/poseidon/runtime_api/poseidon_gpu_api.cpp" \
  "$task_base/src/src/poseidon/tests/runtime_api/poseidon_gpu_api_test.cpp" \
  "$task_base/src/third_party/ckks-runtime/runtime/runtime.hpp" \
  > "$task_logs/build.status"
run_case() {
  local label=$1
  shift
  printf 'REMOTE_TEST_START %s\n' "$label"
  set +e
  /usr/bin/time -v timeout 600 env OMP_NUM_THREADS=1 "$@" \
    > "$task_logs/$label.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" > "$task_logs/$label.exit-status"
  printf 'REMOTE_TEST_COMPLETE %s status=%s\n' "$label" "$status"
  return "$status"
}
run_case gpu-api-four CUDA_VISIBLE_DEVICES=0,1,2,3 \
  POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB=512 \
  "$task_base/build/bin/poseidon_runtime_gpu_api_tests"
run_case waterline-four CUDA_VISIBLE_DEVICES=0,1,2,3 \
  POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB=2560 \
  POSEIDON_WATERLINE_ONLY=1 POSEIDON_WATERLINE_LOG_N=16 \
  POSEIDON_WATERLINE_Q_COUNT=32 POSEIDON_WATERLINE_SCALE=40 \
  POSEIDON_WATERLINE_HOST_STAGED=0 \
  "$task_base/build/bin/poseidon_runtime_gpu_api_tests"
run_case waterline-four-host-staged CUDA_VISIBLE_DEVICES=0,1,2,3 \
  POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB=2560 \
  POSEIDON_WATERLINE_ONLY=1 POSEIDON_WATERLINE_LOG_N=16 \
  POSEIDON_WATERLINE_Q_COUNT=32 POSEIDON_WATERLINE_SCALE=40 \
  POSEIDON_WATERLINE_HOST_STAGED=1 \
  "$task_base/build/bin/poseidon_runtime_gpu_api_tests"
run_case gpu-api-reordered CUDA_VISIBLE_DEVICES=3,1,0,2 \
  POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB=512 \
  "$task_base/build/bin/poseidon_runtime_gpu_api_tests"
run_case gpu-prepare-oom CUDA_VISIBLE_DEVICES=1 \
  "$task_base/build/bin/poseidon_runtime_gpu_prepare_oom_tests"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,ecc.errors.uncorrected.volatile.total \
  --format=csv > "$task_logs/devices-after.csv"
printf 'REMOTE_VALIDATION_COMPLETE\n'
