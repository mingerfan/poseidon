#!/usr/bin/env bash
# Dependency environment only; this is not an untrusted-code sandbox.
# No installer or automatic package downloads. Provision the launcher separately.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_paths.sh"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  printf 'Pinned nix-portable and wheels require Linux x86_64; use an Ubuntu x86_64 backend.\n' >&2
  exit 2
fi
runtime="$POSEIDON_WORK_ROOT/deps/nix-portable-v012"
launcher="$runtime/nix-portable"
if [[ ! -x "$launcher" ]]; then
  printf 'Missing approved nix-portable v012 launcher: %s\n' "$launcher" >&2
  exit 1
fi
if (( $# == 0 )); then
  printf 'Usage: bash scripts/baseline/nix_portable.sh nix-command [arguments...]\n' >&2
  exit 2
fi

# Fingerprint of the v012 x86_64 asset downloaded over HTTPS on 2026-09-05.
# This detects later replacement/corruption; it is not a publisher signature.
expected_sha256=b409c55904c909ac3aeda3fb1253319f86a89ddd1ba31a5dec33d4a06414c72a
if ! printf '%s  %s\n' "$expected_sha256" "$launcher" | sha256sum --check --status; then
  printf 'nix-portable launcher checksum mismatch; refusing to run.\n' >&2
  exit 1
fi

export NP_LOCATION="$runtime"
export NP_GIT=/usr/bin/git
export NP_RUNTIME=bwrap
export NP_BWRAP=/usr/bin/bwrap
# Debug mode in upstream writes the complete environment to /tmp/np_env.
unset NP_DEBUG NP_RUN
export XDG_CACHE_HOME="$POSEIDON_WORK_ROOT/cache/nix-portable-v012"
export XDG_CONFIG_HOME="$runtime/xdg-config"
export XDG_STATE_HOME="$runtime/xdg-state"
export TMPDIR="$POSEIDON_WORK_ROOT/cache/nix-portable-tmp"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_STATE_HOME" "$TMPDIR"

# nix-portable re-creates shared tmpbin on each invocation. Serialize access.
# max-jobs=1 plus cores=2 also caps future Nix derivation builds to two workers.
export NIX_CONFIG="${NIX_CONFIG:+$NIX_CONFIG$'\n'}max-jobs = 1
cores = 2
connect-timeout = 10
stalled-download-timeout = 30
download-attempts = 1"
exec flock --exclusive --timeout 30 "$runtime/launcher.lock" "$launcher" "$@"
