#!/usr/bin/env python3
"""Replay 20,478 scalar-token tasks to isolate CPU scheduler costs."""
import json
from pathlib import Path
import statistics
import subprocess

out = Path(__file__).resolve().parent
repo = out.parents[1]
prior = out.parent/'waterline-cause-20261018'
build = repo/'build-waterline-batch-20261018-ready/cpu-control-complete'
build.mkdir(exist_ok=False)
record = next(r for r in json.loads((prior/'local-build-commands.json').read_text()) if r['variant']=='current')
command = record['command']
command = [('-I'+str(repo/'third_party/ckks-runtime')) if x.endswith('build-waterline-cause-20261018-v3/current/include') else x for x in command]
command[command.index('-o')+1] = str(build/'probe')
(out/'cpu-build-command.json').write_text(json.dumps(command,indent=2)+'\n')
with (out/'cpu-build.log').open('w') as log:
    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
programs = dict(legacy=repo/'build-waterline-cause-20261018-v3/legacy/probe',
                previous=repo/'build-waterline-cause-20261018-v3/current/probe',updated=build/'probe')
suite = repo/'build-waterline-benchmark-20261018/pressure-local-binary'
samples=[];variants=list(programs)
for repeat in range(7):
    shift=repeat%3
    for variant in variants[shift:]+variants[:shift]:
        command=[str(programs[variant]),str(suite/'eager.json'),str(suite/'operator-spec.json'),'4096']
        result=json.loads(subprocess.check_output(command,text=True))
        assert result['passed'] and result['execution_tasks']==20478 and result['compute_calls']==8191
        samples.append(dict(variant=variant,repeat=repeat,command=command,**result))
rows=[]
for variant in variants:
    selected=[s for s in samples if s['variant']==variant]
    rows.append(dict(variant=variant,**{key:dict(median=statistics.median(s[key] for s in selected),
        min=min(s[key] for s in selected),max=max(s[key] for s in selected)) for key in (
            'online_ms','voluntary_context_switches','involuntary_context_switches','main_collections','worker_collections')}))
result=dict(scope='Local CPU-only scalar-token control; no CUDA, CKKS encoding, weight I/O or GPU event queries. Context switches cover the full Runtime call, while online_ms covers online execution. Not FHE/GPU latency.',verified_samples=len(samples),comparisons=rows,samples=samples)
(out/'cpu-control.json').write_text(json.dumps(result,indent=2)+'\n')
for row in rows: print(row['variant'],row['online_ms'],row['voluntary_context_switches'])
