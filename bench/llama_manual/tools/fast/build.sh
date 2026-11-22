#!/bin/bash
# Build Poseidon GPU core + manual Llama on node1. Usage: build.sh [target...]
set -euo pipefail
R1=/root/private_data/llama-fast-20261116
CM=$R1/deps/cmake-wheel/cmake/data/bin/cmake
P=$R1/deps/prefix
export GMP_ROOT=$P CPM_LOCAL_PACKAGES_ONLY=ON CUDA_VISIBLE_DEVICES=
export PATH=/usr/local/cuda/bin:$PATH
if [ ! -f $R1/build/CMakeCache.txt ]; then
  $CM -S $R1/src -B $R1/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DPOSEIDON_BUILD_GPU=ON -DPOSEIDON_BUILD_LLAMA_MANUAL=ON -DPOSEIDON_BUILD_CKKS_RUNTIME_API=OFF \
    -DPOSEIDON_BUILD_EXAMPLES=OFF -DPOSEIDON_BUILD_DEPS=OFF -DPOSEIDON_USE_ZLIB=OFF -DPOSEIDON_USE_ZSTD=OFF \
    -DPOSEIDON_USE_MSGSL=OFF -DPOSEIDON_USE_OPENMP=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DCPM_LOCAL_PACKAGES_ONLY=ON -DCMAKE_PREFIX_PATH=$P -DGMP_ROOT=$P "-DCMAKE_CXX_FLAGS=-fopenmp -fno-fast-math"
fi
if [ $# -eq 0 ]; then set -- llama31_poseidon_manual; fi
$CM --build $R1/build --target "$@" --parallel ${BUILD_JOBS:-$(nproc)}
