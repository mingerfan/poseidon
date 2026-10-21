#!/usr/bin/env bash
set -euo pipefail
base=/home/xuming/poseidon-threshold-rescale-20261021
cmake=/home/xuming/poseidon-tools/cmake/cmake/data/bin/cmake
canonical=/home/xuming/poseidon-framework-validation-20261015-6dc8cbfb
rapids=/home/xuming/poseidon/build-runtime-gpu-rmm2412/_deps/rapids-cmake-src/rapids-cmake
"$cmake" -S "$base/src" -B "$base/build" -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
 -DCMAKE_PREFIX_PATH="$canonical/deps/install" -DPOSEIDON_BUILD_EXAMPLES=OFF \
 -DPOSEIDON_BUILD_BENCH=OFF -DPOSEIDON_BUILD_DEPS=OFF -DPOSEIDON_USE_ZLIB=OFF \
 -DPOSEIDON_USE_ZSTD=OFF -DPOSEIDON_USE_MSGSL=OFF -DPOSEIDON_USE_INTEL_HEXL=OFF \
 -DPOSEIDON_BUILD_CKKS_RUNTIME_API=ON -DPOSEIDON_BUILD_CKKS_RUNTIME_TESTS=ON \
 -DPOSEIDON_BUILD_CKKS_RUNTIME_GPU_TESTS=ON -DPOSEIDON_BUILD_CKKS_RUNTIME_GPU_NCCL=OFF \
 -DPOSEIDON_BUILD_CKKS_RUNTIME_MPI=OFF -DCMAKE_MODULE_PATH="$rapids" -Drapids-cmake-dir="$rapids" \
 -DCPM_fmt_SOURCE="$base/src/third_party/fmt" -DCPM_spdlog_SOURCE="$base/src/third_party/spdlog" \
 -DCPM_CCCL_SOURCE="$base/src/third_party/cccl" -DPOSEIDON_BUNDLED_RMM_DIR=/home/xuming/deps/rmm-v24.12.01
"$cmake" --build "$base/build" --target poseidon_runtime_gpu_qwen_e2e poseidon_runtime_gpu_api_tests -j 10
