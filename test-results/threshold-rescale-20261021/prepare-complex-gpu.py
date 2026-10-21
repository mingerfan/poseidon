#!/usr/bin/env python3
"""Compile the item-1 graph suite for the existing 188Server numeric runner."""
import importlib.util
import json
from pathlib import Path
import subprocess

base = Path(__file__).resolve().parent
root = base.parents[1]
regression = root/'third_party/ckks-runtime/third_party/dacapo/test/runtime-plan/check-threshold-graphs.py'
module_spec = importlib.util.spec_from_file_location('threshold_graphs', regression)
m = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(m)
selected = {'cp-double-rescale', 'cc-double-rescale', 'shared-square',
            'unequal-depth-residual', 'equal-nominal-residual', 'dead-branch'}
cases = [(name, graph, False, 0) for name, graph, _ in m.cases() if name in selected]
g = m.Graph(); previous = 'x'
for i in range(40):
    previous = g.op('mul', previous, 'y' if i % 2 else 'p')
g.outputs = [previous]; cases.append(('mixed-cc-cp-host-boot', g, True, 0))
g = m.Graph(); g.outputs = [g.op('mul', 'x', 'p')]
cases.append(('exhausted-input-host-boot', g, True, 45))
suite = base/'complex-gpu'; suite.mkdir(exist_ok=True)
manifest = []
for name, graph, host, exhausted in cases:
    template = base/('depth40-host-gpu1' if host else 'depth9-gpu1')
    for strategy in ('greedy', 'depth-dp'):
        out = suite/f'{strategy}-{name}'; out.mkdir(exist_ok=True)
        for file in ('operator-spec.json', 'compiler-profile.json'):
            (out/file).write_bytes((template/file).read_bytes())
        (out/'input.mlir').write_text(graph.source(exhausted))
        cmd = [arg.replace(str(template), str(out)) for arg in json.loads((template/'command.json').read_text())]
        cmd = [f'--boot-placement={strategy}' if arg.startswith('--boot-placement=') else
               '--runtime-plan-format=json' if arg.startswith('--runtime-plan-format=') else
               '--runtime-plan-gpu-budget-bytes=15032385536' if arg.startswith('--runtime-plan-gpu-budget-bytes=') else arg for arg in cmd]
        with (out/'compile.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        fixture = json.loads((template/'fixture.json').read_text())
        fixture['model']['name'] = name
        fixture['inputs'] = [dict(name='x', values=[.25]), dict(name='y', values=[1.01])]
        fixture['expected_outputs'] = [dict(name=f'output{i}', values=[v])
                                       for i, v in enumerate(graph.expected(y=1.01))]
        (out/'fixture.json').write_text(json.dumps(fixture, indent=2)+'\n')
        (out/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
        plan = json.loads((out/'result.graph.runtime-plan.json').read_text())
        counts = {op: sum(step.get('op') == op for step in plan['execution'])
                  for op in ('boot', 'rescale', 'mod_switch', 'mul_cc')}
        assert bool(counts['boot']) == host, (name, strategy, counts)
        if name == 'dead-branch':
            assert counts['rescale'] == 0
        manifest.append(dict(name=out.name, host_boot=host,
                             deferred_alignment_precision=name=='equal-nominal-residual', **counts))
        print(out.name, counts, flush=True)
(suite/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
