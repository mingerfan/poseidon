#!/usr/bin/env bash
set -euo pipefail
compiler=${HECATE_OPT:?Set HECATE_OPT to the real hecate-opt with the native-scales patch}
package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output=${1:-"$package/build/recompiled"}
mkdir -p "$output"
"$compiler" --ckks-config="$package/compiler/profiled_HEAAN_GPU_blb5.json" \
  --pass-pipeline="builtin.module(func.func(emit-hevm{prefix=$output/resnet20}))" \
  "$package/source/strict_native_resnet20_20261015_v1.mlir" -o "$output/emitted.mlir"
python3 "$package/tools/audit_strict_native_hevm.py" \
  "$output/resnet20.strict_native_resnet20.hevm" \
  "$package/source/strict_native_resnet20_20261015_v1.lowering.json" "$output/audit.json"
