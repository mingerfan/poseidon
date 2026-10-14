#!/usr/bin/env bash
# Execute after the full-model supervisor exits; keep a separate SSH connected.
set -euo pipefail
base=${POSEIDON_188_WORKSPACE:-/home/xuming/poseidon-waterline-20261013}
isolated=${POSEIDON_PACKED_API_DIRECTORY:-$base/host-read-budget-isolated-20261014}
logs="$base/logs/${POSEIDON_PACKED_PRESSURE_LABEL:-packed-host-pressure-fresh}"
mkdir "$logs"
trap 'code=$?; printf "%s\n" "$code" > "$logs/exit-status"' EXIT
# Bind the API that already passed the packed-boundary check, including its
# linked libraries and production headers. Do not mutate the shared build.
python3 - "$isolated" "$logs/input.status" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); data=json.loads((root/'result.json').read_text())
assert data['complete'] and data['passed'] and data['test_exit_code']==0
assert data['shared_inputs_unchanged'] and not data['old_api_cpp_archive_member_extracted']
items={**data['shared_inputs_after'],**data['private_inputs_after']}
items[str(root/'poseidon_gpu_api_tests')]=data['binary']
with Path(sys.argv[2]).open('x') as out:
 for name,record in items.items():
  expected=record['sha256'].removeprefix('sha256:')
  h=hashlib.sha256()
  with open(name,'rb') as source:
   for chunk in iter(lambda:source.read(8*1024*1024),b''):h.update(chunk)
  assert h.hexdigest()==expected,name
  out.write(f'{expected}  {name}\n')
PY
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,ecc.errors.uncorrected.volatile.total \
  --format=csv > "$logs/devices-before.csv"
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory \
  --format=csv > "$logs/processes-before.csv"
for selection in single all host-staged; do
  visible=0,1,2,3
  staged=0
  if test "$selection" = single; then visible=1; fi
  if test "$selection" = host-staged; then staged=1; fi
  command=(/usr/bin/time -v env CUDA_VISIBLE_DEVICES="$visible" OMP_NUM_THREADS=1
    POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB="${POSEIDON_PRESSURE_POOL_MB:-2560}"
    POSEIDON_WATERLINE_ONLY=1 POSEIDON_WATERLINE_LOG_N=16
    POSEIDON_WATERLINE_Q_COUNT=32 POSEIDON_WATERLINE_SCALE=40
    POSEIDON_WATERLINE_HOST_STAGED="$staged" "$isolated/poseidon_gpu_api_tests")
  printf '%q ' "${command[@]}" > "$logs/waterline-$selection.command.sh"
  printf '\n' >> "$logs/waterline-$selection.command.sh"
  date -Is > "$logs/waterline-$selection.started-at"
  "${command[@]}" > "$logs/waterline-$selection.log" 2>&1
  date -Is > "$logs/waterline-$selection.finished-at"
done
sha256sum -c "$logs/input.status" > "$logs/input-verification-after.log"
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,ecc.errors.uncorrected.volatile.total \
  --format=csv > "$logs/devices-after.csv"
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory \
  --format=csv > "$logs/processes-after.csv"
printf 'PACKED_HOST_SINGLE_ALL_HOSTSTAGED_PRESSURE_COMPLETE\n'
