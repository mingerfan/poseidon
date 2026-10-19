#!/usr/bin/env python3
"""Auxiliary CPU-affinity diagnosis; excluded from the main speed medians."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

out = Path(__file__).resolve().parent
destination = out / 'server-affinity-runs'
destination.mkdir(exist_ok=False)
samples = []
case = 'wl-8MiB-gpu-w512'
for cpus in ('0-11', '12-23'):
    for repeat in range(3):
        variants = ('previous', 'updated') if repeat % 2 == 0 else ('updated', 'previous')
        for variant in variants:
            source = out / 'server-runs' / f'server-pressure-r0-{variant}-{case}'
            environment = dict(os.environ, **json.loads((source / 'environment.json').read_text())['environment'])
            command = json.loads((source / f'r0-{case}' / 'command.json').read_text())
            folder = destination / f'cpus-{cpus}-r{repeat}-{variant}'
            folder.mkdir()
            command[4:6] = [str(folder / 'report.json'), str(folder / 'decoded.json')]
            baseline = json.loads((source / 'summary.json').read_text())
            binary_hash = 'sha256:' + hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()
            assert binary_hash == baseline['runner_hashes']['qwen']
            command = ['taskset', '-c', cpus] + command
            (folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
            with (folder / 'run.log').open('w') as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((folder / 'report.json').read_text())
            assert report['passed']
            assert report['runtime_timing']['bundle_read_calls'] == 4096
            assert report['preparation_measurements']['cpu_prepare_calls'] == 0
            assert report['pinned_transfer_live_after_plan_bytes'] == 0
            assert report['pinned_transfer_live_after_validation_bytes'] == 0
            for field in ('plan_source_sha256', 'operator_spec_sha256', 'fixture_sha256'):
                assert report[field] == baseline['samples'][0]['report'][field]
            samples.append(dict(cpus=cpus, repeat=repeat, variant=variant, binary_sha256=binary_hash,
                                online_ms=report['runtime_timing']['online_execution_ms']))
            print(cpus, repeat, variant, samples[-1]['online_ms'], flush=True)
rows = []
for cpus in ('0-11', '12-23'):
    for variant in ('previous', 'updated'):
        values = [s['online_ms'] for s in samples if s['cpus'] == cpus and s['variant'] == variant]
        rows.append(dict(cpus=cpus, variant=variant, median_ms=statistics.median(values),
                         min_ms=min(values), max_ms=max(values)))
(out / 'server-affinity.json').write_text(json.dumps(dict(
    scope='Auxiliary 8 MiB V100 comparison with the complete process restricted to one CPU socket. Does not pin individual workers, GPU clocks or OS cache. Excluded from main 206 samples.',
    verified_samples=len(samples), comparisons=rows, samples=samples), indent=2) + '\n')
