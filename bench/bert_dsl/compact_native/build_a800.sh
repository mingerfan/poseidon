#!/usr/bin/env bash
# Compatibility name: configure/build the standalone Linear engine locally.
# No private SSH host, dependency bundle, compiler version or GPU architecture.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../../.." && pwd)
export BERT_SOURCE_DIR=$here
export BERT_BUILD_DIR=${BERT_BUILD_DIR:-$root/build-bert-linear}
exec bash "$here/../build.sh" "$@"
