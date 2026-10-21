#!/usr/bin/env python3
"""Replay the already-selected optimized schedule, retaining real constants."""
import json
from pathlib import Path
import re
import subprocess

base = Path('/home/xuming/poseidon-model-regression-20261021')
out = base/'resnet20-runtime-optimized'
out.mkdir(exist_ok=False)
logical = (base/'resnet20-after/optimized/logical.mlir').read_text()
schedule = re.search(r'btp_schedule = (\[.*?\])', logical)[1]
(out/'selected-schedule.mlirattr').write_text(schedule+'\n')
tool = Path('/home/xuming/poseidon-threshold-rescale-20261021/toolchain')
command = [str(tool/'lib/ld-linux-x86-64.so.2'), '--library-path',
           f'{base}/frontend/lib:{tool}/lib:/opt/conda/lib:/lib/x86_64-linux-gnu',
           str(base/'replay-boot-schedule'), str(base/'traced/ResNet20.mlirbc'),
           str(base/'compiler-current.json'), str(out/'selected-schedule.mlirattr'),
           str(out/'scheduled.mlirbc')]
(out/'replay-command.json').write_text(json.dumps(command, indent=2)+'\n')
with (out/'replay.log').open('w') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
template = json.loads((base/'resnet20-runtime-depth-dp/command.json').read_text())
command = ['--pars' if arg == '--dacapo' else
           str(out/'scheduled.mlirbc') if arg == str(base/'traced/ResNet20.mlirbc') else
           str(out/'model.mlir') if arg == str(base/'resnet20-runtime-depth-dp/model.mlir') else arg
           for arg in template]
(out/'command.json').write_text(json.dumps(command, indent=2)+'\n')
with (out/'compile.log').open('w') as log:
    result = subprocess.run(['/usr/bin/time', '-v', *command], stdout=log, stderr=subprocess.STDOUT)
(out/'status.json').write_text(json.dumps(dict(returncode=result.returncode,
    source='Exact automatically selected cut/target schedule; search is not rerun'))+'\n')
if result.returncode: raise SystemExit(result.returncode)
command = [str(base/'audit-plan'), str(out/'model._hecate_ResNet20.runtime-plan.bin'),
           str(base/'resnet-runtime-profiles/operator-spec.json'),
           str(out/'model._hecate_ResNet20.bundle'), str(base/'resnet-runtime-fixture.json'),
           str(out/'clear-execute.json'), 'execute']
with (out/'clear-execute.log').open('w') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
observed = json.loads((out/'clear-execute.json').read_text())
reference = json.loads((base/'resnet20-after/optimized/audit.json').read_text())
assert (observed['boots'], observed['rescale_q'], observed['modswitch_q']) == (
    reference['boot_count'], reference['rescale_q'], reference['modswitch_q'])
assert observed['clear_runtime']['passed']
print('Saved optimized schedule has identical Boot/Q counts and passes compiled FP64 logits')
