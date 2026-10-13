#!/usr/bin/env bash
# Existing 188Server offline dependency/tool paths used for this acceptance.
set -euo pipefail
base=${POSEIDON_188_WORKSPACE:-/home/xuming/poseidon-waterline-20261013}
cmake_bin=/home/xuming/poseidon-tools/cmake/cmake/data/bin/cmake
rapids_dir=/home/xuming/poseidon/build-runtime-gpu-rmm2412/_deps/rapids-cmake-src/rapids-cmake
rmm_dir=/home/xuming/deps/rmm-v24.12.01
mkdir -p "$base/logs"
"$cmake_bin" -S "$base/src" -B "$base/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DPOSEIDON_BUILD_EXAMPLES=OFF -DPOSEIDON_BUILD_BENCH=OFF \
  -DPOSEIDON_BUILD_DEPS=OFF -DPOSEIDON_USE_ZLIB=OFF -DPOSEIDON_USE_ZSTD=OFF \
  -DPOSEIDON_USE_MSGSL=OFF -DPOSEIDON_USE_INTEL_HEXL=OFF \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_API=ON -DPOSEIDON_BUILD_CKKS_RUNTIME_TESTS=ON \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_GPU_TESTS=ON \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_GPU_NCCL=OFF -DPOSEIDON_BUILD_CKKS_RUNTIME_MPI=OFF \
  -DCMAKE_MODULE_PATH="$rapids_dir" -Drapids-cmake-dir="$rapids_dir" \
  -DCPM_fmt_SOURCE="$base/src/third_party/fmt" \
  -DCPM_spdlog_SOURCE="$base/src/third_party/spdlog" \
  -DCPM_CCCL_SOURCE="$base/src/third_party/cccl" \
  -DPOSEIDON_BUNDLED_RMM_DIR="$rmm_dir" > "$base/logs/configure-reproduce.log" 2>&1
/usr/bin/time -v "$cmake_bin" --build "$base/build" \
  --target poseidon_runtime_gpu_api_tests poseidon_runtime_gpu_qwen_e2e -j12 \
  > "$base/logs/build-reproduce.log" 2>&1
sha256sum "$base/build/bin/poseidon_runtime_gpu_api_tests" \
  "$base/build/bin/poseidon_runtime_gpu_qwen_e2e" \
  "$base/src/src/poseidon/runtime_api/poseidon_gpu_api.cpp" \
  "$base/src/third_party/ckks-runtime/runtime/runtime.hpp" \
  > "$base/logs/latest-build.status"
printf 'BUILD_COMPLETE\n'
