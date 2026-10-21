#!/usr/bin/env python3
"""Run this suite on the isolated 188Server checkout, on physical GPU 3."""
import json
import os
from pathlib import Path
import subprocess

base = Path(__file__).resolve().parent
suite = base/'complex-gpu'
uuid = 'GPU-0650e59e-7268-0a1a-8c3e-58f09c3e7fa2'
env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1',
           POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='6144')
runner = base/'build/bin/poseidon_runtime_gpu_qwen_e2e'
rows = []
for case in json.loads((suite/'manifest.json').read_text()):
    directory = suite/case['name']
    command = [str(runner), str(directory/'result.graph.runtime-plan.json'),
               str(directory/'operator-spec.json'), str(directory/'fixture.json'),
               str(directory/'report.json'), str(directory/'decoded.json'),
               '--bundle', str(directory/'result.graph.bundle'), '--backend', 'gpu']
    with (directory/'run.log').open('w') as log:
        run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    row = dict(case, returncode=run.returncode, gpu_uuid=uuid)
    if (directory/'report.json').exists():
        report = json.loads((directory/'report.json').read_text())
        row.update(comparison=report.get('comparison'),
                   runtime_ms=report.get('runtime_ms'), error=report.get('error'))
    rows.append(row)
    (suite/'gpu-summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(row), flush=True)
