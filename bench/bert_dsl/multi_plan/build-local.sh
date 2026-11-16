#!/usr/bin/env bash
# Portable standalone multi-plan build. Never run tests or device workloads.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../../.." && pwd)
export BERT_SOURCE_DIR=$here
export BERT_BUILD_DIR=${BERT_BUILD_DIR:-$root/build-bert-multi-plan}
exec bash "$here/../build.sh" "$@"
