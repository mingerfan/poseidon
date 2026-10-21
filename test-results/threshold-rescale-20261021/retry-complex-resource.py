#!/usr/bin/env python3
"""Repeat the one shared-memory rejection with an explicit 6 GiB pool cap."""
import json
import os
from pathlib import Path
import subprocess

base = Path(__file__).resolve().parent
suite = base/'complex-gpu'; name = 'depth-dp-unequal-depth-residual'
rows = json.loads((suite/'gpu-summary.json').read_text())
row = next(r for r in rows if r['name'] == name)
assert row['comparison'] is None and 'managed pool cap' in row['error']
row['initial_attempt'] = dict(row)
directory = suite/name
for filename in ('report.json', 'run.log', 'decoded.json'):
    path = directory/filename
    if path.exists():
        path.rename(directory/(path.stem+'.initial'+path.suffix))
env = dict(os.environ, CUDA_VISIBLE_DEVICES=row['gpu_uuid'], OMP_NUM_THREADS='1',
           POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='6144')
command = [str(base/'build/bin/poseidon_runtime_gpu_qwen_e2e'),
           str(directory/'result.graph.runtime-plan.json'), str(directory/'operator-spec.json'),
           str(directory/'fixture.json'), str(directory/'report.json'), str(directory/'decoded.json'),
           '--bundle', str(directory/'result.graph.bundle'), '--backend', 'gpu']
with (directory/'run.log').open('w') as log:
    run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
report = json.loads((directory/'report.json').read_text())
row.update(returncode=run.returncode, comparison=report.get('comparison'),
           runtime_ms=report.get('runtime_ms'), error=report.get('error'), maximum_pool_mb=6144)
(suite/'gpu-summary.json').write_text(json.dumps(rows, indent=2)+'\n')
print(json.dumps(row), flush=True)
assert run.returncode == 0 and row['comparison']['passed']
