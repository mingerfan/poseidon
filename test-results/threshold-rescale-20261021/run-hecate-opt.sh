#!/usr/bin/env bash
set -euo pipefail
base=/home/xuming/poseidon-threshold-rescale-20261021/toolchain
exec "$base/lib/ld-linux-x86-64.so.2" --library-path "$base/lib:/opt/conda/lib:/lib/x86_64-linux-gnu" "$base/bin/hecate-opt" "$@"
