#!/usr/bin/env python3
"""Batch independent graphs so every phase shares one native Boot setup."""
import copy
import json
from pathlib import Path

base = Path(__file__).resolve().parent
previous = base.parent / 'native45-profile-20261021'
source = json.loads((previous / 'greedy-depth20/result.chains.runtime-plan.json').read_text())
plan = copy.deepcopy(source)
for key in ['values', 'external_inputs', 'initialization', 'execution', 'finalization', 'final_outputs']:
    plan[key] = []
fixture = json.loads((previous / 'greedy-depth20/fixture.json').read_text())
fixture['model']['name'] = 'native45 input-state regression; released inputs, repeated graphs'
fixture['inputs'] = []
fixture['expected_outputs'] = []
cases = ['greedy-depth20'] * 8 + ['depth-dp-depth27'] * 3 + ['depth-dp-depth20']
for index, case in enumerate(cases):
    original = json.loads((previous / case / 'result.chains.runtime-plan.json').read_text())
    expected = json.loads((previous / case / 'fixture.json').read_text())
    ids = {v['id']: str(index * 1000 + int(v['id'])) for v in original['values']}
    for value in original['values']:
        value['id'] = ids[value['id']]
        plan['values'].append(value)
    for key in ['external_inputs', 'final_outputs']:
        plan[key].extend(ids[v] for v in original[key])
    for stage in ['initialization', 'execution', 'finalization']:
        for op in original[stage]:
            for key in ['inputs', 'outputs']:
                if key in op:
                    op[key] = [ids[v] for v in op[key]]
            for key in ['output', 'value']:
                if key in op:
                    op[key] = ids[op[key]]
            if 'transfer_id' in op:
                op['transfer_id'] = str(index * 1000 + int(op['transfer_id']))
            plan[stage].append(op)
    for key in ['inputs', 'expected_outputs']:
        for value in expected[key]:
            value['name'] = f'{index}-{case}-{value["name"]}'
            fixture[key].append(value)
ordinal = 0
for stage in ['initialization', 'execution', 'finalization']:
    for op in plan[stage]:
        op['ordinal'] = ordinal
        ordinal += 1
out = base / 'released-inputs-batch'
out.mkdir(exist_ok=True)
for name, data in [('result.chains.runtime-plan.json', plan), ('fixture.json', fixture), ('cases.json', cases)]:
    (out / name).write_text(json.dumps(data, indent=2) + '\n')
(out / 'operator-spec.json').write_bytes((previous / 'greedy-depth20/operator-spec.json').read_bytes())
