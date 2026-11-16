#!/usr/bin/env bash
set -euo pipefail
D=${1:?usage: build_native.sh EXPERIMENT_ROOT}
S="$D/src"
B="$D/build"
g++ -O3 -std=c++20 -fopenmp -DFMT_HEADER_ONLY=1 -DSPDLOG_FMT_EXTERNAL \
  -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE \
  -I"$B" -I"$S/src" -I"$D/tools/gmp/include" \
  -I"$S/third_party/spdlog/include" -I"$S/third_party/fmt/include" \
  -I"$S/third_party/rmm/include" -I"$S/third_party/cccl/libcudacxx/include" \
  -I"$S/third_party/cccl/thrust" -I"$S/third_party/cccl/cub" \
  -I"$S/third_party/nvtx/c/include" -I/usr/local/cuda/include \
  "$S/experiments/gpu_keyload/native_profile.cpp" \
  "$B/libposeidon_gpu.a" -L"$B" -lposeidon_shared \
  -L/usr/local/cuda/lib64 -lcudart -pthread \
  -Wl,-rpath,"$B" -Wl,-rpath,/usr/local/cuda/lib64 \
  -o "$D/native_profile"
