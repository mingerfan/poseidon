#!/usr/bin/env python3
import copy
import json
import os
from pathlib import Path
import subprocess
import time

base = Path(__file__).resolve().parent
workspace = Path('/home/xuming/poseidon-threshold-rescale-20261021')
compiler = workspace / 'run-hecate-opt.sh'
runner = workspace / 'build/bin/poseidon_runtime_gpu_qwen_e2e'
names = ['optimized-depth20', 'optimized-depth27', 'optimized-depth50']
rows = []
for name in names:
    out = base / name
    command = json.loads((out / 'command.json').read_text())
    local = str(Path(next(a for a in command if a.endswith('/input.mlir'))).parent)
    command = [a.replace(local, str(out)) for a in command]
    command = [f'--runtime-plan-communication-profile-path={workspace}/profiles/communication-profile.json'
               if a.startswith('--runtime-plan-communication-profile-path=') else a for a in command]
    command[0] = str(compiler)
    (out / 'command-server.json').write_text(json.dumps(command, indent=2) + '\n')
    begin = time.monotonic()
    with (out / 'compile-server.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    rows.append(dict(case=name, compile_seconds=time.monotonic()-begin))
    print(json.dumps(rows[-1]), flush=True)
(base / 'compile-server-summary.json').write_text(json.dumps(rows, indent=2) + '\n')

plan = json.loads((base / names[0] / 'result.chains.runtime-plan.json').read_text())
fixture = json.loads((base / names[0] / 'fixture.json').read_text())
for key in ['values', 'external_inputs', 'initialization', 'execution', 'finalization', 'final_outputs']:
    plan[key] = []
fixture['model']['name'] = 'DaCapo optimized scale45; depth20, depth27, depth50'
fixture['inputs'] = []; fixture['expected_outputs'] = []
for index, name in enumerate(names):
    original = json.loads((base / name / 'result.chains.runtime-plan.json').read_text())
    expected = json.loads((base / name / 'fixture.json').read_text())
    ids = {v['id']: str(1000 * index + int(v['id'])) for v in original['values']}
    for value in original['values']:
        value['id'] = ids[value['id']]; plan['values'].append(value)
    for key in ['external_inputs', 'final_outputs']:
        plan[key].extend(ids[v] for v in original[key])
    for stage in ['initialization', 'execution', 'finalization']:
        for op in original[stage]:
            for key in ['inputs', 'outputs']:
                if key in op: op[key] = [ids[v] for v in op[key]]
            for key in ['output', 'value']:
                if key in op: op[key] = ids[op[key]]
            if 'transfer_id' in op: op['transfer_id'] = str(index * 1000 + int(op['transfer_id']))
            plan[stage].append(op)
    for key in ['inputs', 'expected_outputs']:
        for value in expected[key]:
            value['name'] = name + '-' + value['name']; fixture[key].append(value)
ordinal = 0
for stage in ['initialization', 'execution', 'finalization']:
    for op in plan[stage]:
        op['ordinal'] = ordinal; ordinal += 1
out = base / 'gpu-batch'; out.mkdir(exist_ok=True)
for filename, data in [('result.chains.runtime-plan.json', plan), ('fixture.json', fixture)]:
    (out / filename).write_text(json.dumps(data, indent=2) + '\n')
(out / 'operator-spec.json').write_bytes((base / names[0] / 'operator-spec.json').read_bytes())
command = [str(runner), str(out / 'result.chains.runtime-plan.json'), str(out / 'operator-spec.json'),
           str(out / 'fixture.json'), str(out / 'report.json'), str(out / 'decoded.json'), '--backend', 'gpu']
uuid = 'GPU-0768eaf5-15f4-6cca-f263-141df9bae9c6'
env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='12288')
(out / 'command.json').write_text(json.dumps(dict(command=command, gpu_uuid=uuid, pool_mb=12288, OMP_NUM_THREADS=1), indent=2) + '\n')
with (out / 'run.log').open('w') as log:
    subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
report = json.loads((out / 'report.json').read_text())
print(json.dumps(dict(passed=report['passed'], comparison=report['comparison'], boot_count=report['native_boot_instructions'])), flush=True)
