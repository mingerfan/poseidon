#!/usr/bin/env bash
# Link only; never reconfigure/rebuild the production library or run a GPU test.
set -euo pipefail
B=$(realpath "${1:?usage: build.sh EXISTING_BUILD_DIR [OUTPUT]}")
S=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${2:-"$B/qp_prepare"}
CUDA=${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}
GMP=${GMP_ROOT:-/usr}
read -r -a extra_libs <<< "${EXTRA_LIBS:-}"
"${CXX:-g++}" -O3 -std=c++20 -fopenmp -DFMT_HEADER_ONLY=1 -DSPDLOG_FMT_EXTERNAL \
  -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE \
  -I"$B" -I"$S/src" -I"$GMP/include" \
  -I"$S/third_party/spdlog/include" -I"$S/third_party/fmt/include" \
  -I"$S/third_party/rmm/include" -I"$S/third_party/cccl/libcudacxx/include" \
  -I"$S/third_party/cccl/thrust" -I"$S/third_party/cccl/cub" \
  -I"$S/third_party/nvtx/c/include" -I"$CUDA/include" \
  "$S/bench/qp_prepare/main.cpp" "$B/libposeidon_gpu.a" \
  -L"$B" -lposeidon_shared -L"$GMP/lib" -lgmpxx -lgmp \
  -L"$CUDA/lib64" -lcudart -pthread "${extra_libs[@]}" \
  -Wl,-rpath,"$B" -Wl,-rpath,"$GMP/lib" -Wl,-rpath,"$CUDA/lib64" -o "$OUT"
printf 'Built %s (not executed)\n' "$OUT"
