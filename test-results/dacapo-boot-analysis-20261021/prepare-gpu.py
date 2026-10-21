#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import time

root = Path.cwd()
base = Path(__file__).resolve().parent
previous = base.parent
rows = []
for name, source in [
    ('optimized-depth20', previous / 'native45-profile-20261021/greedy-depth20'),
    ('optimized-depth27', previous / 'native45-profile-20261021/depth-dp-depth27'),
    ('optimized-depth50', previous / 'native45-input-fix-20261021/greedy-depth50')]:
    out = base / name
    out.mkdir(exist_ok=True)
    for filename in ['input.mlir', 'operator-spec.json', 'compiler-profile.json', 'fixture.json']:
        (out / filename).write_bytes((source / filename).read_bytes())
    command = [a.replace(str(source), str(out)) for a in json.loads((source / 'command.json').read_text())]
    command = ['--boot-placement=optimized' if a.startswith('--boot-placement=') else a for a in command]
    (out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    start = time.monotonic()
    with (out / 'compile.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    plan = json.loads((out / 'result.chains.runtime-plan.json').read_text())
    desc = {v['id']: v for v in plan['values']}
    boots = [s for s in plan['execution'] if s.get('op') == 'boot']
    row = dict(case=name, compile_seconds=time.monotonic()-start, boot_count=len(boots),
               boot_inputs=[{k: desc[s['inputs'][0]][k] for k in ['level','scale_log2']} for s in boots],
               outputs=[{k: desc[v][k] for k in ['level','scale_log2']} for v in plan['final_outputs']])
    rows.append(row)
    (base / 'compile-summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(row), flush=True)
