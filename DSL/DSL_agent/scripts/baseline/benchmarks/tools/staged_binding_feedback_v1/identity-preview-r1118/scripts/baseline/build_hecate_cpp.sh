#!/usr/bin/env bash
# Invoke inside the already realized, pinned dacapo-shell.nix environment.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_paths.sh"
root="$POSEIDON_ROOT"
if [[ "$(pwd -P)" != "$root" || -z "${IN_NIX_SHELL:-}" ]]; then
  printf 'Requires the approved checkout and pinned Nix shell.\n' >&2
  exit 1
fi
build="$POSEIDON_PLATFORM_WORK_ROOT/build-dacapo/hecate-18.1.2-nix"
results="$POSEIDON_PLATFORM_WORK_ROOT/results"
head=$(timeout -k 3s 20s git -C third_party/dacapo rev-parse HEAD)
if [[ "$head" != 4616402710f39df3e5f5bd7930a6c036025aaac3 ]]; then
  printf 'Unexpected Dacapo commit: %s\n' "$head" >&2
  exit 1
fi
: "${CC:?Missing pinned CC}" "${CXX:?Missing pinned CXX}"
: "${LLVM_DIR:?Missing LLVM_DIR}" "${MLIR_DIR:?Missing MLIR_DIR}" "${SEAL_DIR:?Missing SEAL_DIR}"
: "${NIX_CC:?Missing Nix compiler wrapper}"
case "${HECATE_C_COMPILER_NAME:-}:${HECATE_CXX_COMPILER_NAME:-}:${HECATE_CXX_COMPILER_VERSION:-}" in
  gcc:g++:13.2.0|clang:clang++:18.1.2) ;;
  *) printf 'Unsupported compiler selection in pinned Nix shell.\n' >&2; exit 1 ;;
esac
test "$CC" = "$NIX_CC/bin/$HECATE_C_COMPILER_NAME"
test "$CXX" = "$NIX_CC/bin/$HECATE_CXX_COMPILER_NAME"
test "$(timeout -k 3s 20s llvm-config --version)" = 18.1.2
timeout -k 3s 20s mlir-tblgen --version | grep -F '18.1.2'
if [[ "$HECATE_C_COMPILER_NAME" == gcc ]]; then
  test "$(timeout -k 3s 20s "$CXX" -dumpfullversion)" = 13.2.0
else
  timeout -k 3s 20s "$CXX" --version | grep -F '18.1.2'
fi
mkdir -p "$results"
report="$results/hecate-cpp-$(date -u +%Y%m%dT%H%M%S)-$$"
timeout -k 3s 60s cmake -S "$root/scripts/baseline/toolchain-probe" \
  -B "$build/toolchain-probe" -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DLLVM_DIR="$LLVM_DIR" \
  -DDACAPO_INCLUDE_DIR="$root/third_party/dacapo/include" \
  2>&1 | tee "$report.toolchain-configure.log"
timeout -k 3s 60s cmake --build "$build/toolchain-probe" --parallel 2 \
  2>&1 | tee "$report.toolchain-build.log"
timeout -k 3s 20s "$build/toolchain-probe/poseidon_llvm_link_probe" \
  | tee "$report.toolchain-run.log"
timeout -k 3s 30s python3 scripts/baseline/hevm_abi.py \
  --probe "$build/toolchain-probe/poseidon_hevm_abi_probe" \
  --header "$root/third_party/dacapo/include/hecate/Support/HEVMHeader.h" \
  --output "$build/hevm-abi.json" | tee "$report.abi.json"
timeout -k 10s 5m cmake -S "$root/third_party/dacapo" -B "$build" -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  '-DCMAKE_JOB_POOLS=compile=2;link=1' \
  -DCMAKE_JOB_POOL_COMPILE=compile -DCMAKE_JOB_POOL_LINK=link \
  -DHECATE_LINK_ALL_MLIR_LIBS=OFF \
  -DLLVM_DIR="$LLVM_DIR" -DMLIR_DIR="$MLIR_DIR" -DSEAL_DIR="$SEAL_DIR" \
  2>&1 | tee "$report.configure.log"
timeout -k 10s 2h cmake --build "$build" --parallel 2 \
  --target hecate-opt HecateFrontend SEAL_HEVM 2>&1 | tee "$report.build.log"
timeout -k 3s 30s "$build/bin/hecate-opt" --show-dialects \
  --ckks-config="$root/third_party/dacapo/config.json" \
  | tee "$report.dialects.log"
# Shared-library loading tests only, not Hecate import or encrypted execution.
timeout -k 3s 30s python3 "$root/scripts/baseline/check_hecate_libraries.py" "$build" \
  | tee "$report.libraries.json"
printf 'C++ build and library-load checks passed; no trace/HEVM/FHE claim. Reports: %s.*\n' "$report"
