#!/usr/bin/env bash
# Host C++ only: no nvcc, no Cerium build, no Python/SEAL library build.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CERIUM_SRC:?set CERIUM_SRC to the directory containing cerium/runtime}"
: "${SEAL_INCLUDE:?set SEAL_INCLUDE to the directory containing seal/seal.h}"
: "${CERIUM_RUNTIME_SO:?set CERIUM_RUNTIME_SO to the existing libCeriumRuntime.so}"
: "${CUDATOOLKIT_ROOT:?set CUDATOOLKIT_ROOT to a CUDA toolkit with include/cuda_runtime_api.h}"
[[ -f "$CERIUM_SRC/cerium/runtime/io/generator.h" ]]
[[ -f "$SEAL_INCLUDE/seal/seal.h" ]]
[[ -f "$CERIUM_RUNTIME_SO" ]]
[[ -f "$CUDATOOLKIT_ROOT/include/cuda_runtime_api.h" ]]
runtime_dir=$(cd "$(dirname "$CERIUM_RUNTIME_SO")" && pwd)
CERIUM_RUNTIME_SO="$runtime_dir/$(basename "$CERIUM_RUNTIME_SO")"
# Prefer the SAME runtime major as Cerium, even when CUDA headers are older.
needed=$(readelf -d "$CERIUM_RUNTIME_SO" | awk '/NEEDED/ && /libcudart/ {gsub(/[][]/, "", $NF); print $NF}')
[[ -n "$needed" ]] || { echo "Cannot determine Cerium libcudart DT_NEEDED" >&2; exit 2; }
if [[ -z ${CUDART_SO:-} ]]; then
  for dir in "$runtime_dir" "$CUDATOOLKIT_ROOT/lib64" "$CUDATOOLKIT_ROOT/targets/x86_64-linux/lib"; do
    if [[ -f "$dir/$needed" ]]; then CUDART_SO="$dir/$needed"; break; fi
  done
fi
: "${CUDART_SO:?set CUDART_SO to the explicit libcudart.so matching Cerium (not necessarily the toolkit major)}"
[[ -f "$CUDART_SO" ]]
cudart_dir=$(cd "$(dirname "$CUDART_SO")" && pwd)
CUDART_SO="$cudart_dir/$(basename "$CUDART_SO")"
soname=$(readelf -d "$CUDART_SO" | awk '/SONAME/ {gsub(/[][]/, "", $NF); print $NF}')
[[ "$soname" == "$needed" ]] || {
  echo "Refusing mixed CUDA runtimes: Cerium needs $needed, selected $soname" >&2; exit 2;
}
mkdir -p "$here/build"
primes_source=${PRIMES_SOURCE:-"$CERIUM_SRC/benchmarks/bootstrap_32K/primes.py"}
bash "$here/extract_primes.sh" "$primes_source" "$here/build/cerium_bench_primes.h"
"${CXX:-g++}" -std=c++17 -O3 -DNDEBUG -Wall -Wextra -pthread \
  -I"$here/build" -isystem "$CERIUM_SRC" -isystem "$SEAL_INCLUDE" \
  -isystem "$CUDATOOLKIT_ROOT/include" \
  "$here/main.cpp" "$CERIUM_RUNTIME_SO" "$CUDART_SO" \
  -Wl,-rpath,"$runtime_dir" -Wl,-rpath,"$cudart_dir" \
  -Wl,-rpath-link,"$runtime_dir" -Wl,-rpath-link,"$cudart_dir" \
  -ldl -o "$here/build/cerium_keyload"
printf 'Built %s (CUDA headers: %s; linked runtime: %s)\n' \
  "$here/build/cerium_keyload" "$CUDATOOLKIT_ROOT" "$CUDART_SO" >&2
