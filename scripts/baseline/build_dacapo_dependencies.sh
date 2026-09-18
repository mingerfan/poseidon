#!/usr/bin/env bash
# User-approved C++ dependency stage only; never installs Python wheels or CUDA.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_paths.sh"
if [[ "$(pwd -P)" != "$POSEIDON_ROOT" ]]; then
  printf 'Run from this source checkout: %s\n' "$POSEIDON_ROOT" >&2
  exit 1
fi
if [[ $# != 2 || "$1" != --approved ]]; then
  printf 'After approval and dry-run budget review: %s --approved {sources|seal|toolchain|shell}\n' "$0" >&2
  exit 2
fi
recipe=src/poseidon/tools/dacapo/dacapo-dependencies.nix
case "$2" in
  sources) targets=(toolchain.src seal.src); limit=20m ;;
  seal) targets=(seal); limit=45m ;;
  toolchain) targets=(toolchain clang); limit=12h ;;
  shell) recipe=src/poseidon/tools/dacapo/dacapo-shell.nix; targets=(); limit=12h ;;
  *) printf 'Unknown dependency stage: %s\n' "$2" >&2; exit 2 ;;
esac
results="$POSEIDON_WORK_ROOT/results"
mkdir -p "$results"
report="$results/dacapo-build-$2-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'Dependency stage: %s; log: %s\n' "$2" "$report"
set +e
timeout -k 10s "$limit" bash scripts/baseline/nix_portable.sh \
  nix build --option allow-import-from-derivation false \
  --option max-silent-time 600 --max-jobs 1 --cores 2 \
  --no-link --keep-failed --print-out-paths --print-build-logs \
  --file "$recipe" "${targets[@]}" 2>&1 | tee "$report"
pipeline_status=("${PIPESTATUS[@]}")
status=${pipeline_status[0]}
if (( status == 0 && pipeline_status[1] != 0 )); then
  status=${pipeline_status[1]}
fi
set -e
printf 'Dependency stage %s exit code: %s (not a DSL/GPU validation)\n' "$2" "$status" | tee -a "$report"
exit "$status"
