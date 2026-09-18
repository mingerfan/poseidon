#!/usr/bin/env bash
# Metadata inspection only: no nix-shell entry, package realization or installation.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_paths.sh"
if [[ "$(pwd -P)" != "$POSEIDON_ROOT" ]]; then
  printf 'Run from this source checkout: %s\n' "$POSEIDON_ROOT" >&2
  exit 1
fi
results="$POSEIDON_WORK_ROOT/results"
mkdir -p "$results"
report="$results/dacapo-dependency-plan-$(date -u +%Y%m%dT%H%M%S)-$$"
timeout -k 3s 90s bash scripts/baseline/nix_portable.sh \
  nix eval --offline --option allow-import-from-derivation false --json \
  --file src/poseidon/tools/dacapo/dacapo-dependencies.nix metadata \
  | tee "$report.json"
timeout -k 3s 180s bash scripts/baseline/nix_portable.sh \
  nix build --dry-run --option allow-import-from-derivation false --no-link \
  --file src/poseidon/tools/dacapo/dacapo-shell.nix 2>&1 \
  | tee "$report.log"
printf 'Planning reports (not installation results): %s.{json,log}\n' "$report"
