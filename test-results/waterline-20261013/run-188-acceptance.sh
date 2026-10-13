#!/usr/bin/env bash
# Keep an independent long-lived SSH connection open while this runs in tmux.
set -euo pipefail
base=${POSEIDON_188_WORKSPACE:-/home/xuming/poseidon-waterline-20261013}
wait_pid=${POSEIDON_WAIT_FOR_PID:-0}
run_label=${POSEIDON_ACCEPTANCE_LABEL:-qwen3-bsgs-auto-final}
run_logs="$base/logs/$run_label"
cd "$base"
if (( wait_pid > 0 )); then
  printf 'waiting_for_existing_model_pid=%s\n' "$wait_pid"
  identity=$(awk '{print $22}' "/proc/$wait_pid/stat")
  while test -r "/proc/$wait_pid/stat"; do
    current=$(awk '{print $22}' "/proc/$wait_pid/stat")
    test "$current" = "$identity" || break
    sleep 10
  done
fi
test -f "$base/logs/latest-build.status"
cat "$base/logs/latest-build.status"
test "$(/opt/conda/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["returncode"])' \
  "$base/qwen3-plans-bsgs-final/gpu4/status.json")" = 0
mkdir "$run_logs"
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,ecc.errors.uncorrected.volatile.total \
  --format=csv > "$run_logs/devices-before.csv"
for selection in single all host-staged; do
  visible=0,1,2,3
  staged=0
  if test "$selection" = single; then visible=1; fi
  if test "$selection" = host-staged; then staged=1; fi
  /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$visible" OMP_NUM_THREADS=1 \
    POSEIDON_WATERLINE_ONLY=1 POSEIDON_WATERLINE_LOG_N=16 \
    POSEIDON_WATERLINE_Q_COUNT=32 POSEIDON_WATERLINE_SCALE=40 \
    POSEIDON_WATERLINE_HOST_STAGED="$staged" \
    "$base/build/bin/poseidon_runtime_gpu_api_tests" \
    > "$run_logs/waterline-$selection.log" 2>&1
  printf 'pressure_complete selection=%s\n' "$selection"
done
plan="$base/qwen3-plans-bsgs-final/gpu4/qwen3._hecate_qwen3_0_6b"
prefix="$run_logs/model"
set +e
/usr/bin/time -v env CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
  "$base/build/bin/poseidon_runtime_gpu_qwen_e2e" \
  "$plan.runtime-plan.bin" "$base/profiles/operator-spec.json" \
  "$base/qwen3-runner-bsgs.json" "$prefix-report.json" "$prefix-decoded.json" \
  --bundle "$plan.bundle" --backend auto --trace "$prefix.trace" \
  > "$prefix.log" 2>&1 &
timed_pid=$!
set -e
while kill -0 "$timed_pid" 2>/dev/null; do
  children=$(cat "/proc/$timed_pid/task/$timed_pid/children")
  if test -n "$children"; then
    read -r model_pid _ <<< "$children"
    printf 'full_bsgs_auto_model_pid=%s\n' "$model_pid"
    /opt/conda/bin/python "$base/monitor_process.py" --pid "$model_pid" \
      --output "$prefix-memory.jsonl" > "$prefix-monitor.log" 2>&1 &
    monitor_pid=$!
    break
  fi
  sleep 1
done
set +e
wait "$timed_pid"
model_status=$?
set -e
printf '%s\n' "$model_status" > "$prefix.exit-status"
if test -n "${monitor_pid:-}"; then wait "$monitor_pid"; fi
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,ecc.errors.uncorrected.volatile.total \
  --format=csv > "$run_logs/devices-after.csv"
if (( model_status != 0 )); then
  printf 'full_bsgs_auto_failed status=%s\n' "$model_status"
  exit "$model_status"
fi
cd "$base/qwen3_dsl-bsgs"
/opt/conda/envs/apollo/bin/python validate_decoded.py \
  --signature "$base/qwen3-traced-bsgs/signature.json" --fixture "$base/qwen3-fixture" \
  --runner-fixture "$base/qwen3-runner-bsgs.json" \
  --decoded "$prefix-decoded.json" --runner-report "$prefix-report.json" \
  --output "$prefix-independent-validation.json" > "$prefix-independent-validation.log" 2>&1
printf 'full_bsgs_auto_numerical_acceptance_complete\n'
