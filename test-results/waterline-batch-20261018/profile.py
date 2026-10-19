#!/usr/bin/env python3
"""Separate Nsight checks; never included in unprofiled latency medians."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess

out = Path(__file__).resolve().parent
repo = out.parents[1]
profiles = repo/'build-waterline-batch-20261018-ready/nsys-complete'
profiles.mkdir(exist_ok=False)
result = dict(scope='Separate profiles, excluded from latency medians. Kernel span covers the whole process, not Runtime online. Gaps may include copies. CUDA API durations are host wall time, not exclusive CPU work.', profiles={})
for name, group, variant, case in (
        ('previous-qproj','local-qproj','previous','wl-64MiB-gpu-w128'),
        ('updated-qproj','local-qproj','updated','wl-64MiB-gpu-w128'),
        ('updated-mlp','local-mlp','updated','eager')):
    label = variant + ('-eager' if case == 'eager' else '-waterline')
    source = out/'local-runs'/f'{group}-r1-{label}'
    environment = dict(os.environ, **json.loads((source/'environment.json').read_text())['environment'])
    command = json.loads((source/f'r0-{case}'/'command.json').read_text())
    folder = profiles/name; folder.mkdir()
    report_index = len(command)-1 if group == 'local-mlp' else 4
    command[report_index] = str(folder/'report.json')
    if group != 'local-mlp': command[5] = str(folder/'decoded.json')
    prefix = folder/'profile'
    profile = ['nsys','profile','--trace=cuda','--sample=none','--cpuctxsw=none',
               '--cuda-memory-usage=false','--stats=false','-o',str(prefix)] + command
    (folder/'command.json').write_text(json.dumps(profile,indent=2)+'\n')
    with (folder/'profile.log').open('w') as log:
        subprocess.run(profile,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['nsys','export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    assert json.loads((folder/'report.json').read_text())['passed']
    db = sqlite3.connect(str(prefix)+'.sqlite')
    names = dict(db.execute('select id,value from StringIds'))
    kernels = list(db.execute('select start,end,demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ from CUPTI_ACTIVITY_KIND_KERNEL order by start'))
    shapes = Counter((names[r[2]],)+r[3:] for r in kernels)
    signature = [dict(name=k[0],grid=k[1:4],block=k[4:7],count=v) for k,v in sorted(shapes.items())]
    start, end = kernels[0][:2]; busy = 0
    for row in kernels[1:]:
        if row[0] > end:
            busy += end-start; start,end = row[:2]
        else: end = max(end,row[1])
    busy += end-start
    span = max(r[1] for r in kernels)-kernels[0][0]
    calls = list(db.execute('select s.value,count(*),sum(r.end-r.start)/1e6 from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like "%EventQuery%" or s.value like "%EventSynchronize%" or s.value like "%LaunchKernel%" group by s.value'))
    result['profiles'][name] = dict(passed=True,kernels=len(kernels),kernel_shapes=signature,
        kernel_shape_sha256=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest(),
        kernel_union_ms=busy/1e6,kernel_span_ms=span/1e6,no_kernel_between_first_last_ms=(span-busy)/1e6,
        cuda_api_summary=[dict(api=r[0],calls=r[1],host_wall_ms=r[2]) for r in calls])
    db.close()
    print(name,len(kernels),busy/1e6,span/1e6,flush=True)
old = json.loads((out.parent/'waterline-legacy-20261018/nsys-mlp.json').read_text())['profiles']['legacy']
assert result['profiles']['updated-mlp']['kernel_shape_sha256'] == old['kernel_shape_sha256']
assert result['profiles']['previous-qproj']['kernel_shape_sha256'] == result['profiles']['updated-qproj']['kernel_shape_sha256']
result['same_kernel_shapes_and_counts'] = True
(out/'nsys-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
