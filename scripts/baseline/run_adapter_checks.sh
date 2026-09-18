#!/usr/bin/env bash
# CPU-side adapter/preflight checks. No CUDA, Dacapo, or MLIR is required.
# Mock artifact success is NOT compiler or encrypted GPU execution success.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_paths.sh"
work="$POSEIDON_WORK_ROOT"
build="$work/build-poseidon/agent-dsl-adapter"
gmp="$work/deps/gmp-6.3.0"
test -f "$gmp/include/gmp.h"
test -f "$gmp/lib/libgmp.so"
mkdir -p "$work/results"
log="$work/results/adapter-$(date -u +%Y%m%dT%H%M%S)-$$.log"
exec > >(tee "$log") 2>&1
printf 'scope=cpu_adapter_preflight_only\nsource=%s\nbuild=%s\nlog=%s\n' "$repo" "$build" "$log"
timeout -k 3s 30s git -C "$repo" status --short --branch
timeout -k 3s 20s git -C "$repo" rev-parse HEAD
export CPLUS_INCLUDE_PATH="$gmp/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$gmp/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$gmp/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=2
cmake -S "$repo" -B "$build" \
  -DPOSEIDON_BUILD_DEPS=OFF -DPOSEIDON_USE_MSGSL=OFF \
  -DPOSEIDON_USE_ZSTD=OFF -DPOSEIDON_USE_ZLIB=ON \
  -DPOSEIDON_USE_HARDWARE=OFF -DPOSEIDON_BUILD_EXAMPLES=OFF \
  -DPOSEIDON_BUILD_BENCH=OFF -DPOSEIDON_BUILD_MGPU_BENCH=OFF \
  -DPOSEIDON_BUILD_MGPU=ON -DPOSEIDON_BUILD_MGPU_TESTS=ON \
  -DPOSEIDON_BUILD_MGPU_TOOLS=ON -DPOSEIDON_BUILD_MGPU_GPU_RUNTIME=OFF \
  -DPOSEIDON_BUILD_MGPU_GPU_OBJECTS=OFF -DPOSEIDON_BUILD_MGPU_CUDA_COMM=OFF \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH="$gmp/lib"
cmake --build "$build" -j2
ctest --test-dir "$build" --output-on-failure --timeout 60 -j2
