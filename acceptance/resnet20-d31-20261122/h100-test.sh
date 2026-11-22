#!/usr/bin/env bash
# Run on an H100 after building this snapshot with CMAKE_CUDA_ARCHITECTURES=90.
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
source_dir=${POSEIDON_H100_SOURCE:-"$root/h100-source"}
build_dir=${POSEIDON_H100_BUILD:-"$root/h100-build"}
mode=${1:-best}
image=${2:-0}
case "$mode" in best|base) ;; *) echo 'usage: h100_test.sh [best|base] [image-id]' >&2; exit 2;; esac
log_dir=$(mktemp -d "$root/h100-${mode}-${image}-XXXXXXXX")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export POSEIDON_WIDE_NTT_FUSED=1 POSEIDON_WIDE_NTT_TILE=1024
export POSEIDON_WIDE_POOL_GIB=${POSEIDON_WIDE_POOL_GIB:-60} POSEIDON_WIDE_FULL_KEYS=1
export POSEIDON_WIDE_HOIST_DIGITS=1 POSEIDON_WIDE_HOIST_KEYCACHE=1 POSEIDON_CTS_LOG_BSGS_RATIO=1
export POSEIDON_WIDE_FAST_LIFT=1 POSEIDON_WIDE_FUSED_ACCUMULATE=1
export POSEIDON_WIDE_NTT_PAIR=1 POSEIDON_WIDE_INTT_PAIR=1
unset POSEIDON_WIDE_NTT_TRIPLE POSEIDON_WIDE_INTT_TRIPLE POSEIDON_WIDE_NTT_SINGLE_BARRIER POSEIDON_WIDE_NTT_KEYACC POSEIDON_WIDE_NTT_FIVE POSEIDON_WIDE_NTT_FIVE_SCALAR POSEIDON_WIDE_NTT_FOUR_SCALAR POSEIDON_WIDE_BATCH_KS POSEIDON_WIDE_BATCH_HOIST POSEIDON_WIDE_LAZY_ACCUMULATE POSEIDON_WIDE_POLY_OVERWRITE POSEIDON_WIDE_POLY_AFFINE || true
if [ "$mode" = best ]; then
    # This file is generated from the selected, fully validated experiment.
    . "$root/h100-selected-flags.env"
fi
{ nvidia-smi -L; nvidia-smi; uname -a; } > "$log_dir/hardware.txt"
env | grep '^POSEIDON_' | sort > "$log_dir/settings.txt"
sha256sum "$build_dir/bin/resnet20_wide_functional" > "$log_dir/binary.sha256"
nvidia-smi --query-gpu=index,uuid,timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 200 > "$log_dir/memory.csv" 2>&1 & monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
"$build_dir/bin/functional_wide_ntt_probe" 31 > "$log_dir/ntt-probe.log" 2>&1
"$build_dir/bin/functional_wide_bv_probe" 31 > "$log_dir/bv-probe.log" 2>&1
"$build_dir/bin/resnet20_wide_functional" 31 "$image" "$source_dir/trident-functional/resnet20/functional_params" --benchmark > "$log_dir/network.log" 2>&1
grep -E '^(GPU_ACTIVITY_TIMING|ONLINE_TIMING|NETWORK image=)' "$log_dir/network.log"
echo "Logs: $log_dir"
