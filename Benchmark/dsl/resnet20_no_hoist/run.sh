#!/usr/bin/env bash
set -euo pipefail
package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=${1:-metadata}
build=${BUILD_DIR:-"$package/build"}
binary=${RESNET20_DSL_BINARY:-"$build/bin/resnet20_dsl"}
assets=${ASSET_DIR:-"$package/build/assets"}
python_bin=${PYTHON:-python3}
unset POSEIDON_RESCALE_META_BITS POSEIDON_RESCALE_MODEL_BITS POSEIDON_DSL_SCALE_SHIFT
"$python_bin" "$package/tools/prepare_assets.py" --output "$assets"
args=("$package/runtime/native_snapshot/Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt"
      "$package/artifacts/strict-native-v1.strict_native_resnet20.hevm"
      "$assets/strict-native-v1.cst" "$assets/strict-native-v1.inputs.f64")
if [[ "$mode" == metadata ]]; then exec "$binary" --metadata-only "${args[@]}"; fi
case "$mode" in short) limit=700;; boot-smoke) limit=1800;; full) limit=0;; *) echo 'Modes: metadata, short, boot-smoke, full'; exit 2;; esac
gpu=${2:?Pass an idle physical GPU index for encrypted execution}
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU index must be numeric'; exit 2; }
if [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "GPU$gpu has a compute process; no run started"; exit 2
fi
free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
[[ "$free" -ge 28672 ]] || { echo 'Need 26GiB pool + 2GiB headroom'; exit 2; }
unset POSEIDON_RESCALE_META_BITS POSEIDON_RESCALE_MODEL_BITS POSEIDON_DSL_SCALE_SHIFT
unset POSEIDON_DSL_INPUT_Q POSEIDON_DSL_FRACTIONAL_RESCALE_CORRECTION
unset POSEIDON_DSL_CLEAR_REFERENCE_DIAG POSEIDON_DSL_CLEAR_DIAG_CHUNK_OPS
export CUDA_VISIBLE_DEVICES="$gpu" POSEIDON_NTT_ALGO=fourstep
export POSEIDON_GPU_LINEAR_TRANSFORM_MODE=no_hoist POSEIDON_BOOTSTRAP_PROFILE=dynamic32
export POSEIDON_BASELINE_POOL_GIB=26 POSEIDON_BASELINE_POOL_HEADROOM_GIB=2
if [[ "$limit" == 0 ]]; then unset POSEIDON_DSL_MAX_OPS; else export POSEIDON_DSL_MAX_OPS="$limit"; fi
exec "$binary" --unsafe-accuracy-only "${args[@]}"
