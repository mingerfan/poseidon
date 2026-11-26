#!/usr/bin/env bash
set -euo pipefail

bench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$bench_dir/../.." && pwd)"
archive="$bench_dir/vendor/upstream-be4bc9f5.tar.gz"
overlay="$bench_dir/vendor/local.patch"
cmake_bin="${CMAKE:-cmake}"
build_dir="${BUILD_DIR:-$repo_dir/build_agent22}"
mkdir -p -- "$build_dir"
build_dir="$(cd -- "$build_dir" && pwd)"

# A change to the snapshot or patch gets its own source and CMake cache.
version="$(cat "$archive" "$overlay" | sha256sum | cut -c1-16)"
source_dir="$build_dir/source-$version"
binary_dir="$build_dir/build-$version"
if [[ ! -d "$source_dir" ]]; then
    staging="$(mktemp -d "$build_dir/source.XXXXXX")"
    trap 'rm -rf -- "$staging"' EXIT
    tar -xzf "$archive" -C "$staging"
    patch --batch --forward -d "$staging" -p1 < "$overlay"
    ln -s -- "$repo_dir/third_party" "$staging/third_party"
    mv -T -- "$staging" "$source_dir"
    trap - EXIT
fi

"$cmake_bin" -S "$source_dir" -B "$binary_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-70}" \
    -DCMAKE_CUDA_COMPILER="${NVCC:-/usr/local/cuda/bin/nvcc}" \
    -DPOSEIDON_AGENT22_BENCH_DIR="$bench_dir" \
    -DPOSEIDON_BUILD_GPU=ON -DPOSEIDON_BUILD_DEPS=OFF \
    -DPOSEIDON_BUILD_EXAMPLES=OFF -DPOSEIDON_USE_OPENMP=ON \
    -DPOSEIDON_USE_MSGSL=OFF -DPOSEIDON_USE_ZLIB=OFF -DPOSEIDON_USE_ZSTD=OFF
"$cmake_bin" --build "$binary_dir" --target agent22_bootstrap --parallel "${BUILD_JOBS:-6}"

if [[ "${1:-}" != --build-only ]]; then
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    # The benchmark process holds secret keys during preparation and verification.
    ulimit -c 0
    exec "$binary_dir/bin/agent22_bootstrap" "$@"
fi
