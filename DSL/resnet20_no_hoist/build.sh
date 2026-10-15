#!/usr/bin/env bash
set -euo pipefail
package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build=${BUILD_DIR:-"$package/build"}
cmake_bin=${CMAKE_BIN:-cmake}
if [[ -z ${CUDACXX:-} && -x /usr/local/cuda/bin/nvcc ]]; then export CUDACXX=/usr/local/cuda/bin/nvcc; fi
"$cmake_bin" -S "$package" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-70}"
"$cmake_bin" --build "$build" --target resnet20_dsl --parallel "${BUILD_JOBS:-2}"
printf '%s\n' "$build/bin/resnet20_dsl"
