#!/usr/bin/env bash
set -euo pipefail
exec /home/xuming/poseidon-threshold-rescale-20261021/toolchain/lib/ld-linux-x86-64.so.2 --library-path /home/xuming/poseidon-threshold-rescale-20261021/toolchain/lib:/opt/conda/lib:/lib/x86_64-linux-gnu /home/xuming/poseidon-model-regression-20261021/hecate-opt-baseline "$@"
