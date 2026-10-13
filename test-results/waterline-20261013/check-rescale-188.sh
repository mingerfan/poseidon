#!/usr/bin/env bash
# Run in tmux while a separate SSH keepalive remains connected.
set -euo pipefail
base=${POSEIDON_188_WORKSPACE:-/home/xuming/poseidon-waterline-20261013}
logs="$base/logs/rescale-normalization-checks"
mkdir "$logs"
trap 'code=$?; printf "%s\n" "$code" > "$logs/exit-status"' EXIT
if test -n "${POSEIDON_BUILD_EXIT_STATUS:-}"; then
  while ! test -f "$POSEIDON_BUILD_EXIT_STATUS"; do sleep 10; done
  test "$(cat "$POSEIDON_BUILD_EXIT_STATUS")" = 0
fi
sha256sum -c "$base/logs/latest-build.status" > "$logs/build-verification.log"
cp "$base/logs/latest-build.status" "$logs/build.status"
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,ecc.errors.uncorrected.volatile.total \
  --format=csv > "$logs/devices-before.csv"
/usr/bin/time -v env OMP_NUM_THREADS=1 "$base/build/bin/poseidon_runtime_cpu_api_tests" \
  > "$logs/cpu-api.log" 2>&1
/usr/bin/time -v env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
  "$base/build/bin/poseidon_runtime_gpu_api_tests" > "$logs/gpu-api-single.log" 2>&1
/usr/bin/time -v env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
  "$base/build/bin/poseidon_runtime_gpu_scalar_tests" > "$logs/gpu-scalar-single.log" 2>&1
for selection in single all host-staged; do
  visible=0,1,2,3
  staged=0
  if test "$selection" = single; then visible=1; fi
  if test "$selection" = host-staged; then staged=1; fi
  /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$visible" OMP_NUM_THREADS=1 \
    POSEIDON_WATERLINE_ONLY=1 POSEIDON_WATERLINE_LOG_N=16 \
    POSEIDON_WATERLINE_Q_COUNT=32 POSEIDON_WATERLINE_SCALE=40 \
    POSEIDON_WATERLINE_HOST_STAGED="$staged" \
    "$base/build/bin/poseidon_runtime_gpu_api_tests" > "$logs/waterline-$selection.log" 2>&1
done
candidate="$base/q50-small-regression"
mkdir "$candidate"
tar --touch -xf "$base/q50-small-regression.tar" -C "$candidate"
plan="$candidate/q_proj/optimized._hecate_cyclic_linear_smoke"
for backend in cpu gpu auto; do
  /usr/bin/time -v env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
    "$base/build/bin/poseidon_runtime_gpu_qwen_e2e" \
    "$plan.runtime-plan.json" "$candidate/profiles/operator-spec.json" \
    "$candidate/q_proj/fixture.json" "$logs/q-proj-$backend-report.json" \
    "$logs/q-proj-$backend-decoded.json" --bundle "$plan.bundle" --backend "$backend" \
    > "$logs/q-proj-$backend.log" 2>&1
done
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,ecc.errors.uncorrected.volatile.total \
  --format=csv > "$logs/devices-after.csv"
printf 'CPU_GPU_AUTO_RESCALE_AND_ALL_GPU_PRESSURE_COMPLETE\n'
