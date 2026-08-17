#!/usr/bin/env bash

set -u -o pipefail

artifact_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
poseidon_root=$(cd "$artifact_dir/../.." && pwd)
build_dir=${POSEIDON_BUILD_DIR:-"$poseidon_root/build-runtime-gpu-api-release"}
binary="$build_dir/bin/poseidon_runtime_gpu_mlp_e2e"

device_count=${1:-}
workload=${2:-}
case "$device_count" in
    4)
        default_devices=0,1,2,3
        ;;
    8)
        default_devices=0,1,2,3,4,5,6,7
        ;;
    *)
        echo "usage: $0 4|8 mlp|probe [REPORT_JSON]" >&2
        exit 2
        ;;
esac
case "$workload" in
    mlp|probe) ;;
    *)
        echo "usage: $0 4|8 mlp|probe [REPORT_JSON]" >&2
        exit 2
        ;;
esac

if [[ ! -x "$binary" ]]; then
    echo "GPU runtime executable not found: $binary" >&2
    echo "Set POSEIDON_BUILD_DIR to the Poseidon build directory." >&2
    exit 1
fi

report=${3:-"$artifact_dir/reports/${device_count}gpu-${workload}.json"}
mkdir -p "$(dirname "$report")"
report_tmp="${report}.tmp.$$"
trap 'rm -f "$report_tmp"' EXIT

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$default_devices}
export POSEIDON_RUNTIME_DEVICE_WORKERS=$device_count
export POSEIDON_GPU_MLP_ALLOW_NO_BOOT=1

set +e
"$binary" \
    "$artifact_dir/plans/gpu/${workload}-${device_count}gpu.runtime-plan.json" \
    "$artifact_dir/profiles/gpu-operator-spec.json" \
    "$artifact_dir/plaintext-bundle" \
    "$artifact_dir/fixture/input.json" \
    "$artifact_dir/fixture/mock-result.json" \
    "$report_tmp"
status=$?
set -e

if [[ ! -s "$report_tmp" ]]; then
    if ((status == 0)); then
        echo "GPU runtime returned success without writing a report" >&2
        status=1
    fi
    exit "$status"
fi

mv "$report_tmp" "$report"
trap - EXIT
echo "report=$report"
if ((status != 0)); then
    echo "runtime completed; numerical validation result is ignored" >&2
fi
exit 0
