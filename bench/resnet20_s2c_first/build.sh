#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
build_dir=${POSEIDON_S2C_BUILD_DIR:-"$repo_dir/build_resnet20_s2c_first"}
cmake_bin=${CMAKE_BIN:-}
if [[ -z "$cmake_bin" ]]; then
  if [[ -n ${CONDA_PREFIX:-} && -x "$CONDA_PREFIX/bin/cmake" ]]; then
    cmake_bin="$CONDA_PREFIX/bin/cmake"
  elif [[ -x /opt/conda/envs/apollo/bin/cmake ]]; then
    cmake_bin=/opt/conda/envs/apollo/bin/cmake
  else
    cmake_bin=cmake
  fi
fi

"$cmake_bin" -S "$repo_dir/src/poseidon/tests/bootstrapping" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DPOSEIDON_BUILD_RESNET20_S2C_FIRST=ON
"$cmake_bin" --build "$build_dir" --target poseidon_resnet20_s2c_first \
  --parallel "${BUILD_JOBS:-2}"
binary="$build_dir/bin/poseidon_resnet20_s2c_first"
# Preserve the old default output and optional argument without external .o files.
output=${1:-/tmp/poseidon_resnet20_s2c_first}
if [[ $(readlink -m -- "$output") != $(readlink -m -- "$binary") ]]; then
  cp -- "$binary" "$output"
fi
printf '%s\n' "$output"
