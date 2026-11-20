#!/usr/bin/env bash
# H100 interface: cross-build is possible elsewhere; tests require an H100.
set -euo pipefail
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the extracted complete snapshot root}"
: "${GMP_ROOT:?Set GMP_ROOT to a GMP development installation}"
mode=${1:-build}
build=${BUILD_DIR:-"$SOURCE_ROOT/build-sm90-audit"}
cuda=${CUDA_ROOT:-/usr/local/cuda}
export PATH="$cuda/bin:$PATH"
case "$mode" in
  build)
    cmake -S "$SOURCE_ROOT/migration" -B "$build" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
      -DPOSEIDON_BUILD_EXAMPLES=OFF -DPOSEIDON_BUILD_GPU=ON \
      -DPOSEIDON_BUILD_RESNET20_S2C_FIRST=ON -DPOSEIDON_USE_ZLIB=OFF \
      -DPOSEIDON_USE_ZSTD=OFF -DPOSEIDON_USE_MSGSL=OFF -DSPDLOG_FMT_EXTERNAL_HO=OFF \
      -DFETCHCONTENT_SOURCE_DIR_SPDLOG="$SOURCE_ROOT/migration/third_party/spdlog" \
      -DGMP_INCLUDE_DIR="$GMP_ROOT/include" -DGMPXX_INCLUDE_DIR="$GMP_ROOT/include" \
      -DGMP_LIBRARY="$GMP_ROOT/lib/libgmp.so" -DGMPXX_LIBRARY="$GMP_ROOT/lib/libgmpxx.so"
    cmake --build "$build" --target resnet20_wide_functional functional_wide_ntt_probe functional_wide_bv_probe --parallel "${JOBS:-8}"
    ;;
  test)
    : "${COEFF_DIR:?Set COEFF_DIR to Trident functional_params}"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,driver_version,memory.total --format=csv
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name --format=csv,noheader | grep -qi H100 || {
      echo 'H100 was not detected; refusing to label these results as H100.' >&2; exit 1;
    }
    export POSEIDON_WIDE_NTT_FUSED=1 POSEIDON_WIDE_NTT_TILE=1024
    export POSEIDON_WIDE_FULL_KEYS=1 POSEIDON_WIDE_HOIST_DIGITS=1 POSEIDON_WIDE_HOIST_KEYCACHE=1
    export POSEIDON_CTS_LOG_BSGS_RATIO=1 POSEIDON_WIDE_FAST_LIFT=1
    # Set POSEIDON_WIDE_POOL_GIB only after checking the actual card's memory.
    "$build/bin/functional_wide_ntt_probe" 31
    "$build/bin/functional_wide_bv_probe" 31
    "$build/bin/resnet20_wide_functional" 31 "${IMAGE_ID:-0}" "$COEFF_DIR" --benchmark
    ;;
  *) echo 'Usage: h100-retest.sh build|test' >&2; exit 2 ;;
esac
