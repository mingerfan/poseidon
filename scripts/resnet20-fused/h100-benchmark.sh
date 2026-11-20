#!/usr/bin/env bash
# On an actual H100 only. All results go into a newly created directory.
set -uo pipefail
: "${BUILD_DIR:?Set BUILD_DIR to the sm_90 build directory}"
: "${COEFF_DIR:?Set COEFF_DIR to Trident functional_params}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader | grep -qi H100 || {
  echo 'Selected device is not an H100; no H100 results will be recorded.' >&2; exit 1;
}
dir=$(mktemp -d "${OUTPUT_PARENT:-$PWD}/h100-d31.XXXXXX") || exit 1
echo "RESULT_DIR=$dir"
export POSEIDON_WIDE_NTT_FUSED=1 POSEIDON_WIDE_NTT_TILE=1024
export POSEIDON_WIDE_FULL_KEYS=1 POSEIDON_WIDE_HOIST_DIGITS=1 POSEIDON_WIDE_HOIST_KEYCACHE=1
export POSEIDON_CTS_LOG_BSGS_RATIO=1 POSEIDON_WIDE_FAST_LIFT=1
# Set POSEIDON_WIDE_POOL_GIB only after checking actual free memory.
{
  date -u
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,driver_version,memory.total,memory.free --format=csv
  nvcc --version
  if [ -n "${SOURCE_ARCHIVE:-}" ]; then sha256sum "$SOURCE_ARCHIVE"; fi
  sha256sum "$BUILD_DIR/bin/resnet20_wide_functional" "$BUILD_DIR/bin/functional_wide_ntt_probe" "$BUILD_DIR/bin/functional_wide_bv_probe"
  env | LC_ALL=C sort | grep '^POSEIDON_\|^CUDA_VISIBLE_DEVICES=' || true
} > "$dir/environment.txt" 2>&1
for probe in functional_wide_ntt_probe functional_wide_bv_probe; do
  "$BUILD_DIR/bin/$probe" 31 > "$dir/$probe.log" 2>&1
  status=$?
  echo "$probe status=$status"
  [ "$status" -eq 0 ] || exit "$status"
done
if command -v ncu >/dev/null 2>&1; then
  ncu --target-processes all --launch-count 1 --metrics gpu__time_duration.sum \
    "$BUILD_DIR/bin/functional_wide_ntt_probe" 31 > "$dir/ncu-probe.log" 2>&1
  echo "ncu status=$? (see log; unavailable counters are not performance results)"
fi
for ((run=1; run<=${REPEATS:-5}; run++)); do
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -lms 200 > "$dir/run-$run-memory.csv" 2>&1 &
  monitor=$!
  "$BUILD_DIR/bin/resnet20_wide_functional" 31 "${IMAGE_ID:-0}" "$COEFF_DIR" --benchmark > "$dir/run-$run.log" 2>&1
  status=$?
  kill "$monitor" 2>/dev/null || true
  wait "$monitor" 2>/dev/null || true
  echo "run=$run status=$status"
  grep -E '^(NETWORK |ONLINE_TIMING |GPU_ACTIVITY_TIMING |BENCHMARK memory_)' "$dir/run-$run.log" || true
  [ "$status" -eq 0 ] || exit "$status"
done
echo H100_BENCHMARK_COMPLETE
