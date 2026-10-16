#!/usr/bin/env bash
set -euo pipefail
package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo=${1:-$(cd -- "$package/../.." && pwd)}
build=${2:-"$package/build/gpu"}
if [[ -z ${CUDACXX:-} && -x /usr/local/cuda/bin/nvcc ]]; then export CUDACXX=/usr/local/cuda/bin/nvcc; fi
cmake -S "$repo/DSL/resnet20_no_hoist" -B "$build" \
  -DPOSEIDON_ROOT="$repo" -DPOSEIDON_IMAGENET_TARGETS_FILE="$package/tools/targets.cmake" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDAARCHS:-native}" \
  -DPOSEIDON_BUILD_EXAMPLES=OFF -DPOSEIDON_BUILD_BENCH=OFF
cmake --build "$build" --target resnet18_native resnet18_dsl resnet18_same_cipher -j"${JOBS:-2}"
