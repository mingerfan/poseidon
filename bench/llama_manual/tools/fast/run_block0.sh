#!/bin/bash
# Run one Block0 (V1 layer0) manual Llama case on node1.
# Usage: run_block0.sh <tag> <plan> [VAR=value ...]
set -euo pipefail
R1=/root/private_data/llama-fast-20261116
tag=$1; plan=$2; shift 2
out=$R1/runs/$tag; mkdir -p $out
A=$R1/assets
W=/root/private_data/llama-cerium-align-a800-20261114/assets/layer0
rep=$out/SUMMARY.json; rm -f $rep
cat > $out/manifest.tsv <<EOF
MANUAL_LLAMA_INPUTS_V1
fixture $A/r20_fixture.txt d9a79047125a6506c6b2d4c335737ccdb52a216f398a8150b133d47a44f72f56
weights $W/layer.weights.bin 3a4780271abf7780094e272be10abe3dcc089f3cbde6301d0cc33329a70e0e7d
index $W/index.tsv 36b6f66f819aa0e34683f9adce037b599d2c4572ae97a3ea670b7600a4d49b32
embeddings $A/embeddings.bin 5e1a1e39b2021a66119bdb7c5b2dd8a65d4e6fd701c2ce16f52000f27f00839f
reference $A/reference-block01.bin 7cbc25944cfeddc720415e9ceb6b61001f92cdf20fd9e93a389dcaabdf11cc25
report $rep -
EOF
for kv in "$@"; do export "$kv"; done
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
gpu_idx=${CUDA_VISIBLE_DEVICES%%,*}
used=$(nvidia-smi -i "$gpu_idx" --query-gpu=memory.used --format=csv,noheader,nounits)
apps=$(nvidia-smi -i "$gpu_idx" --query-compute-apps=pid --format=csv,noheader)
if [ -n "$apps" ] || [ "$used" -gt 1000 ]; then echo "GPU busy: used=$used apps=$apps"; exit 3; fi
export LD_LIBRARY_PATH=$R1/deps/prefix/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7} POSEIDON_GPU_QP_PREPARE=optimized POSEIDON_GPU_QP_PREPARE_THREADS=4 POSEIDON_GPU_QP_VERIFY_REFERENCE=0
env | grep -E '^(LLAMA|POSEIDON)_' | sort > $out/env.txt
exec 9>$R1/runs/gpu${gpu_idx}.lock; flock -n 9 || { echo "own GPU lock held"; exit 4; }
( while true; do nvidia-smi -i "$gpu_idx" --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader >> $out/gpu.csv; sleep 5; done ) &
mon=$!
set +e
PYBIN=$(command -v python3 || echo /opt/conda/bin/python3)
start=$($PYBIN -c 'import time; print(time.monotonic())')
$R1/build/bin/llama31_poseidon_manual --run $plan $A/r20_fixture.txt $W/layer.weights.bin $W/index.tsv \
  $A/embeddings.bin $A/reference-block01.bin 1 38 $out/manifest.tsv $rep > $out/stdout.log 2> $out/stderr.log
rc=$?
$PYBIN -c "import time; print(f'wall={time.monotonic()-$start:.3f}')" >> $out/stderr.log
kill $mon
echo "rc=$rc" > $out/rc.txt; echo "rc=$rc"
grep -E "MANUAL_LLAMA_(BLOCK|DONE|FAILED)" $out/stdout.log $out/stderr.log | tail -5
