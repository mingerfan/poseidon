#!/usr/bin/env bash
# Reuse the two small plans produced by lazy-rescale-composite-regression.py.
# Keep a separate SSH connection alive. Single-card tests use physical GPU 1.
set -euo pipefail
base=${POSEIDON_188_WORKSPACE:-/home/xuming/poseidon-waterline-20261013}
input=${POSEIDON_COMPOSITE_INPUT_DIR:-$base/lazy-rescale-composite}
logs="$base/logs/${POSEIDON_COMPOSITE_LABEL:-lazy-rescale-composite-final}"
mkdir "$logs"
trap 'code=$?; printf "%s\n" "$code" > "$logs/exit-status"' EXIT
sha256sum -c "$base/logs/latest-build.status" > "$logs/build-verification.log"
cp "$base/logs/latest-build.status" "$logs/build.status"
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,ecc.errors.uncorrected.volatile.total \
  --format=csv > "$logs/devices-before.csv"
for scale in 40 120; do
  plan="$input/scale$scale/optimized._hecate_composite_scale_regression"
  sha256sum "$plan.runtime-plan.json" "$input/scale$scale/operator-spec.json" \
    "$input/scale$scale/fixture.json" "$plan.bundle/manifest.json" "$plan.bundle/data.bin" \
    > "$logs/scale$scale-inputs.sha256"
  for backend in cpu gpu auto; do
    /usr/bin/time -v env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
      "$base/build/bin/poseidon_runtime_gpu_qwen_e2e" \
      "$plan.runtime-plan.json" "$input/scale$scale/operator-spec.json" \
      "$input/scale$scale/fixture.json" "$logs/scale$scale-$backend-report.json" \
      "$logs/scale$scale-$backend-decoded.json" --bundle "$plan.bundle" --backend "$backend" \
      > "$logs/scale$scale-$backend.log" 2>&1
  done
done
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,ecc.errors.uncorrected.volatile.total \
  --format=csv > "$logs/devices-after.csv"
printf 'LAZY_RESCALE_40_AND_120_SIX_MODE_COMPOSITE_PASS\n'
