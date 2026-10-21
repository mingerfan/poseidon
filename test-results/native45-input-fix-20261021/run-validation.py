#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys

base = Path(__file__).resolve().parent
phase = sys.argv[1]
runner = base / ('runner-' + phase)
uuid = 'GPU-0768eaf5-15f4-6cca-f263-141df9bae9c6'
env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1',
           POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='12288')
summary_path = base / ('summary-' + phase + '.json')
rows = json.loads(summary_path.read_text()) if summary_path.exists() else []
for case in sys.argv[2:]:
    out = base / case
    cmd = [str(runner), str(out / 'result.chains.runtime-plan.json'),
           str(out / 'operator-spec.json'), str(out / 'fixture.json'),
           str(out / ('report-' + phase + '.json')),
           str(out / ('decoded-' + phase + '.json')), '--backend', 'gpu']
    (out / ('command-' + phase + '.json')).write_text(json.dumps(dict(
        command=cmd, gpu_uuid=uuid, OMP_NUM_THREADS=1, pool_mb=12288), indent=2) + '\n')
    with (out / ('run-' + phase + '.log')).open('w') as log:
        run = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    report = json.loads(Path(cmd[4]).read_text())
    row = dict(case=case, returncode=run.returncode, gpu_uuid=uuid,
               comparison=report.get('comparison'), error=report.get('error'),
               boot_count=report.get('native_boot_instructions'))
    rows = [r for r in rows if r['case'] != case] + [row]
    summary_path.write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(row), flush=True)
    if run.returncode:
        sys.exit(run.returncode)
