#!/usr/bin/env bash
# Compatibility name: build locally with the target/toolchain selected by CMake.
# Historical private SSH deployment, dependency copying and pip downloads are
# intentionally not part of the portable build entry point.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$here/build.sh" "$@"
