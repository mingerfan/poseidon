#!/usr/bin/env python3
"""Compile native two-GPU placement; eager is an intermediate, never executed."""
import json
import os
from pathlib import Path
import subprocess
import time

out = Path(__file__).resolve().parent
source = Path('/home/xuming/poseidon-waterline-20261013/qwen3-plans-bsgs-q50/gpu4/command.json')
command = json.loads(source.read_text())
folder = out / 'gpu2-source'
folder.mkdir(exist_ok=False)
changes = {
    '--runtime-plan-device-count=4': '--runtime-plan-device-count=2',
    '--runtime-plan-device-counts=4': '--runtime-plan-device-counts=2',
    '--runtime-plan-plaintext-schedule=waterline': '--runtime-plan-plaintext-schedule=eager',
    '--runtime-plan-memory-report=true': '--runtime-plan-memory-report=false',
}
for original in changes:
    assert command.count(original) == 1, original
command = [changes.get(item, item) for item in command]
assert command[-2] == '-o'
command[-1] = str(folder / 'qwen3.mlir')
(folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
(folder / 'scope.json').write_text(json.dumps(dict(
    source_command=str(source), changes=changes,
    purpose='Native two-device HEFT placement with unchanged checkpoint, graph, CKKS arithmetic and rescaling. Eager artifact is only an intermediate for a bounded lazy prefix; no eager GPU execution.'), indent=2) + '\n')
start = time.monotonic()
environment = dict(os.environ, OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
with (folder / 'compile.log').open('w') as log:
    result = subprocess.run(['/usr/bin/time', '-v', '-o', str(folder / 'compile.time.log')] + command,
                            env=environment, stdout=log, stderr=subprocess.STDOUT)
(folder / 'status.json').write_text(json.dumps(dict(exit_code=result.returncode,
    wall_seconds=time.monotonic() - start), indent=2) + '\n')
raise SystemExit(result.returncode)
