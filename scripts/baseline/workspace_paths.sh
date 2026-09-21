#!/usr/bin/env bash
# Source this from Linux helpers. No downloads or directory creation.
POSEIDON_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
POSEIDON_WORK_ROOT=$(python3 "$POSEIDON_ROOT/scripts/baseline/workspace_paths.py" --base) || return 1
POSEIDON_PLATFORM_WORK_ROOT=$(python3 "$POSEIDON_ROOT/scripts/baseline/workspace_paths.py") || return 1
POSEIDON_PLATFORM=$(python3 "$POSEIDON_ROOT/scripts/baseline/platform_config.py" --require) || return 1
export POSEIDON_ROOT POSEIDON_WORK_ROOT POSEIDON_PLATFORM POSEIDON_PLATFORM_WORK_ROOT
