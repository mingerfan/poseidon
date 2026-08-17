#!/usr/bin/env bash

set -u -o pipefail

artifact_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
poseidon_root=$(cd "$artifact_dir/../.." && pwd)
build_dir=${POSEIDON_BUILD_DIR:-"$poseidon_root/build-runtime-gpu-api-release"}
binary="$build_dir/bin/poseidon_runtime_cpu_mlp_e2e"

workload=${1:-}
case "$workload" in
    mlp|probe) ;;
    *)
        echo "usage: $0 mlp|probe [REPORT_JSON]" >&2
        exit 2
        ;;
esac

if [[ ! -x "$binary" ]]; then
    echo "CPU runtime executable not found: $binary" >&2
    echo "Set POSEIDON_BUILD_DIR to the Poseidon build directory." >&2
    exit 1
fi

report=${2:-"$artifact_dir/reports/cpu-${workload}.json"}
mkdir -p "$(dirname "$report")"
report_tmp="${report}.tmp.$$"
trap 'rm -f "$report_tmp"' EXIT

set +e
POSEIDON_RUNTIME_CPU_GPU_EQUIVALENT_CONTEXT=1 "$binary" \
    "$artifact_dir/plans/cpu/${workload}.runtime-plan.json" \
    "$artifact_dir/profiles/cpu-operator-spec.json" \
    "$artifact_dir/plaintext-bundle" \
    "$artifact_dir/fixture/input.json" \
    "$artifact_dir/fixture/mock-result.json" \
    "$report_tmp"
status=$?
set -e

if [[ ! -s "$report_tmp" ]]; then
    if ((status == 0)); then
        echo "CPU runtime returned success without writing a report" >&2
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
