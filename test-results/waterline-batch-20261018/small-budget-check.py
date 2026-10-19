#!/usr/bin/env python3
"""Isolate completion-publication timing without changing production sources."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

out = Path(__file__).resolve().parent
destination = out / 'small-budget-check-runs'
destination.mkdir(exist_ok=False)
samples = []
for budget in (8, 64):
    case = f'wl-{budget}MiB-gpu-w512'
    source = out / 'server-runs' / f'server-pressure-r0-updated-{case}'
    base_command = json.loads((source / f'r0-{case}' / 'command.json').read_text())
    environment = dict(os.environ, **json.loads((source / 'environment.json').read_text())['environment'])
    baseline = json.loads((source / 'summary.json').read_text())['samples'][0]['report']
    for variant in ('foreground-progress', 'all-progress', 'updated'):
        command = list(base_command)
        if variant != 'updated':
            command[0] = str(out / 'build-small-budget-diagnostic' / variant / 'bin/poseidon_runtime_gpu_qwen_e2e')
        folder = destination / f'{budget}MiB-{variant}'
        folder.mkdir()
        command[4:6] = [str(folder / 'report.json'), str(folder / 'decoded.json')]
        binary_sha = hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()
        command = ['taskset', '-c', '0-11'] + command
        (folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        with (folder / 'run.log').open('w') as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((folder / 'report.json').read_text())
        assert report['passed']
        assert report['runtime_timing']['bundle_read_calls'] == 4096
        for field in ('plan_source_sha256', 'operator_spec_sha256', 'fixture_sha256'):
            assert report[field] == baseline[field]
        samples.append(dict(budget_mib=budget, variant=variant, binary_sha256=binary_sha,
                            timing=report['runtime_timing']))
        (out / 'small-budget-check.json').write_text(json.dumps(dict(
            scope='Isolated alternate headers, not production and excluded from main 206 samples. One sample per case, CPU0-11.',
            samples=samples), indent=2) + '\n')
        print(budget, variant, samples[-1]['timing']['online_execution_ms'], flush=True)
