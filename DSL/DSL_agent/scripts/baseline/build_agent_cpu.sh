#!/usr/bin/env bash
# Run only in the already realized pinned Nix shell. No downloads or key creation.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_paths.sh"
test -n "${IN_NIX_SHELL:-}"
test "$(pwd -P)" = "$POSEIDON_ROOT"

bash scripts/baseline/build_hecate_cpp.sh

key_build="$POSEIDON_PLATFORM_WORK_ROOT/build-dacapo/seal-golden-keys"
timeout -k 3s 120s cmake -S "$POSEIDON_ROOT/scripts/baseline/seal_keys" \
  -B "$key_build" -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" -DSEAL_DIR="$SEAL_DIR" \
  '-DCMAKE_JOB_POOLS=compile=2;link=1' \
  -DCMAKE_JOB_POOL_COMPILE=compile -DCMAKE_JOB_POOL_LINK=link
timeout -k 10s 600s cmake --build "$key_build" --parallel 2 \
  --target seal_golden_keys seal_golden_metadata seal_packed_keys seal_packed_metadata

python3 -B - <<'PY'
import ctypes
from pathlib import Path
import sys
sys.path.insert(0, "scripts/baseline")
from workspace_paths import WORK
from hevm_abi import check_elf
from agent_setup import ensure_frontend_link
ensure_frontend_link(WORK)
for name in ("libseal_golden_metadata.so", "libseal_packed_metadata.so"):
    path = WORK / "build-dacapo/seal-golden-keys" / name
    check_elf(path)
    ctypes.CDLL(str(path))
print("Native compiler/runtime built and libraries loaded; encrypted execution not yet tested.")
PY
