#!/usr/bin/env bash
# Offline API/shape checks only; this script never selects the GPU path.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bin="$here/build/cerium_keyload"
[[ -x "$bin" ]] || { echo "Run build.sh first" >&2; exit 2; }
[[ $# == 0 || ( $# == 1 && $1 == --full ) ]] || { echo "usage: $0 [--full]" >&2; exit 2; }
results=$(mktemp -d "$here/build/cpu-check.XXXXXX")
trap 'rm -rf "$results"' EXIT
run_check() {
  local label=$1; shift
  "$bin" "$@" > "$results/$label.json" 2> "$results/$label.stderr"
  grep -q '"ok":true' "$results/$label.json"
  grep -q '"gpu_calls":false' "$results/$label.json"
  grep -q '"temporary_files_removed":true' "$results/$label.json"
  local temporary
  temporary=$(awk -F '"temporary_directory":"' 'NF == 2 {split($2, a, "\""); print a[1]}' "$results/$label.json")
  [[ -n "$temporary" && ! -e "$temporary" ]]
  printf '%s: PASS (no CUDA calls, no retained temporary keys)\n' "$label"
}
run_check default_plan --plan-only
grep -q '"keypair_count":18' "$results/default_plan.json"
grep -q '"payload_bytes":452984832' "$results/default_plan.json"
grep -q '"expected_file_bytes":453000368' "$results/default_plan.json"
run_check small --cpu-only --q 2 --p 1 --k 1
grep -q '"verified_host_exact":true' "$results/small.json"
grep -q '"actual_keypair_count":4' "$results/small.json"
run_check uneven_relin_only --cpu-only --q 5 --p 2 --k 0
grep -q '"digits":3' "$results/uneven_relin_only.json"
grep -q '"rotation_steps":\[\]' "$results/uneven_relin_only.json"
if "$bin" --plan-only --q 64 --p 1 > "$results/invalid.json" 2> "$results/invalid.stderr"; then
  echo 'Invalid Q+P accepted' >&2; exit 1
fi
grep -q '"ok":false' "$results/invalid.json"
if [[ ${1:-} == --full ]]; then
  run_check default_cpu_roundtrip --cpu-only
  grep -q '"actual_keypair_count":18' "$results/default_cpu_roundtrip.json"
  grep -q '"actual_payload_bytes":452984832' "$results/default_cpu_roundtrip.json"
  grep -q '"actual_file_bytes":453000368' "$results/default_cpu_roundtrip.json"
  grep -q '"verified_host_exact":true' "$results/default_cpu_roundtrip.json"
fi
