#!/usr/bin/env bash
# Run from WSL: bash scripts/baseline/run_cpu_ckks.sh
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_paths.sh"
work="$POSEIDON_WORK_ROOT"
build="$work/build-poseidon/agent-dsl-cpu"
gmp="$work/deps/gmp-6.3.0"
test -f "$gmp/include/gmp.h"
test -f "$gmp/lib/libgmp.so"
mkdir -p "$work/results"
log="$work/results/cpu-ckks-$(date -u +%Y%m%dT%H%M%S)-$$.log"
exec > >(tee "$log") 2>&1
printf 'source=%s\nbuild=%s\nlog=%s\n' "$repo" "$build" "$log"
timeout -k 3s 20s git -C "$repo" status --short --branch
timeout -k 3s 20s git -C "$repo" rev-parse HEAD
export CPLUS_INCLUDE_PATH="$gmp/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$gmp/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$gmp/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Optional GSL and Zstd integrations disabled; encryption parameters unchanged.
# No automatic downloads, installation, CUDA, or HPU.
cmake -S "$repo" -B "$build" \
  -DPOSEIDON_BUILD_DEPS=OFF -DPOSEIDON_USE_MSGSL=OFF \
  -DPOSEIDON_USE_ZSTD=OFF -DPOSEIDON_USE_ZLIB=ON \
  -DPOSEIDON_USE_HARDWARE=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_RPATH="$gmp/lib"
cmake --build "$build" --target test_ckks_deterministic -j2
ldd "$build/libposeidon_shared.so"
timeout -k 3s 120s env OMP_NUM_THREADS=2 "$build/bin/test_ckks_deterministic"
