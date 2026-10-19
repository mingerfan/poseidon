import collections
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

root = Path(__file__).resolve().parents[2]
import argparse
parser = argparse.ArgumentParser(description='Profile the original and fixed production runners; outputs are excluded from timing medians.')
parser.add_argument('--poll-results', type=Path, default=root/'test-results/waterline-fix-20261018')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
base = args.poll_results.resolve()
output = args.output.resolve()
output.mkdir(exist_ok=False)
environment = os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    POSEIDON_RUNTIME_ROTATION_KEY_POLICY='binary', POSEIDON_RUNTIME_DEVICE_WORKERS='1', POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB='6144', POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='6144')
summary = {'scope':'Separate profiled binary-manifest q_proj runs; excluded from unprofiled latency medians. Kernel span is first-to-last GPU kernel, not runtime.run wall time.', 'profiles':{}}
for variant in ('before','after'):
    folder = output/variant
    folder.mkdir()
    command = json.loads((base/f'local-qproj-{variant}/r0-wl-64MiB-gpu-w128/command.json').read_text())
    command[4], command[5] = str(folder/'report.json'), str(folder/'decoded.json')
    prefix = folder/'profile'
    profile = ['nsys','profile','--trace=cuda','--sample=none','--cpuctxsw=none','--cuda-memory-usage=false','--stats=false','-o',str(prefix)]+command
    (folder/'command.json').write_text(json.dumps(profile,indent=2)+'\n')
    with (folder/'profile.log').open('w') as log:
        subprocess.run(profile,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['nsys','export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],stdout=log,stderr=subprocess.STDOUT,check=True)
    report = json.loads((folder/'report.json').read_text())
    assert report['passed'] and report['comparison']['slots_checked']==8192
    db = sqlite3.connect(str(prefix)+'.sqlite')
    names = dict(db.execute('select id,value from StringIds'))
    rows = list(db.execute('select start,end,demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ from CUPTI_ACTIVITY_KIND_KERNEL order by start'))
    shapes = collections.Counter((names[r[2]],)+r[3:] for r in rows)
    shape_rows = [dict(name=k[0],grid=k[1:4],block=k[4:7],count=v) for k,v in sorted(shapes.items())]
    busy = 0
    start, end = rows[0][:2]
    for row in rows[1:]:
        if row[0] > end:
            busy += end-start
            start,end=row[:2]
        else:
            end=max(end,row[1])
    busy += end-start
    span=max(r[1] for r in rows)-rows[0][0]
    queries = list(db.execute('select s.value,count(*),sum(r.end-r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like "%EventQuery%" group by s.value'))
    summary['profiles'][variant] = dict(passed=True, kernels=len(rows), kernel_shapes=shape_rows,
        kernel_shape_sha256=hashlib.sha256(json.dumps(shape_rows,sort_keys=True).encode()).hexdigest(),
        gpu_kernel_union_ms=busy/1e6, gpu_kernel_span_ms=span/1e6, no_kernel_between_first_last_ms=(span-busy)/1e6,
        gpu_busy_fraction=busy/span, cuda_event_query_api=[dict(api=q[0],calls=q[1],cpu_api_ms=q[2]/1e6) for q in queries],
        profiled_runtime_ms=report['runtime_ms'])
    db.close()
    print(variant, len(rows), busy/1e6, span/1e6,flush=True)
summary['same_kernel_shapes_and_counts'] = summary['profiles']['before']['kernel_shape_sha256']==summary['profiles']['after']['kernel_shape_sha256']
(root/'test-results/waterline-fix-20261018/nsys-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
