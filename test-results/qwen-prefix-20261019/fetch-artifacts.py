#!/usr/bin/env python3
"""Fetch compact evidence, excluding model weights, binary plans and decoded slots."""
import io
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
import sys,tarfile
root=Path('/home/xuming/poseidon-qwen-prefix-20261019')
paths=[]
for name in ('instrumentation.json','source-bindings.json','prefix-command.json',
             'prefix-generation.log','run-supervisor.log','run-status.json','run-launch.json',
             'live-proc.jsonl','observed-build-v2.log','observer-pause-diagnostic.json',
             'observer-aba-diagnostic.json','server-provenance.json','server-health-final.txt'):
    paths.append(root/name)
for name in ('observer-query-aba-diagnostic.json','superseded-smi-run.json',
             'nvml-observer-check.json','run-launch-nvml.json','run-supervisor-nvml.log','run-status-nvml.json'):
    paths.append(root/name)
for name in ('prefix-operations.json','inspect-prefix-command.json','inspect-prefix-build.log',
             'server-topology.txt'):
    paths.append(root/name)
for name in ('commands.json','libraries.json'):
    paths.append(root/'build-observed-v2'/name)
for name in ('prefix-summary.json','fixture.json'):
    paths.append(root/'plans/prefix4-2000000'/name)
for variant in ('updated','previous'):
    paths.extend(p for p in (root/'runs'/variant).glob('*')
                 if p.is_file() and p.name != 'decoded.json')
    paths.extend(p for p in (root/'runs-nvml'/variant).glob('*')
                 if p.is_file() and p.name != 'decoded.json')
for name in ('status.json','compile.log'):
    paths.append(root/'gpu2-source'/name)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
    for p in paths:
        if p.is_file():
            arc=p.relative_to(root)
            if arc.parts[0]=='plans':
                arc=Path('prefix')/arc.name
            archive.add(p,arcname=str(arc),recursive=False)
'''

result = subprocess.run(['ssh','188Server','python3','-'],input=REMOTE.encode(),
                        stdout=subprocess.PIPE,check=True)
with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
    archive.extractall(ROOT,filter='data')
print('Fetched',len(result.stdout),'bytes of compact evidence.')
