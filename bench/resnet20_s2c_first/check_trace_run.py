#!/usr/bin/env python3
"""Check actual traced RuntimePlan structure and the same-key GPU result."""
import json
import math
from pathlib import Path
import re
import sys

plan_path, log_path = map(Path, sys.argv[1:3])
plan = json.loads(plan_path.read_text())
lines = log_path.read_text().splitlines()


def one(prefix):
    rows = [x for x in lines if x.startswith(prefix)]
    if len(rows) != 1:
        raise ValueError(f"expected one {prefix}, got {len(rows)}")
    return dict(re.findall(r"([^\s=]+)=([^\s]+)", rows[0]))


def need(condition, message):
    if not condition:
        raise ValueError(message)


need(plan['format_version'] == 8, 'expected typed-buffer V8')
values = {x['id']: x for x in plan['values']}
need(len(values) == len(plan['values']), 'duplicate ValueId')
need(len(plan['external_inputs']) == 27, 'expected 27 prepared patches')
for item in plan['external_inputs']:
    value = values[item]
    need(value['kind'] == 'ciphertext' and value['place'] == {'kind': 'device', 'rank': 0, 'index': 0},
         'external patch is not already on GPU0')
need(not plan['initialization'] and not plan['finalization'], 'unexpected offline plan instructions')
produced = {}
used = {}
released = {}
counts = {}
for index, instruction in enumerate(plan['execution']):
    kind = instruction['kind']
    if kind == 'compute':
        need(instruction['place'] == {'kind': 'device', 'rank': 0, 'index': 0}, 'nonlocal instruction')
        need(instruction['op'] == 'ApiCall' and 'reuse_input' not in instruction, 'unexpected operation/reuse')
        callee = instruction['attrs']['callee']
        opcode = callee.split('.')[1]
        counts[opcode] = counts.get(opcode, 0) + 1
        for source in instruction['inputs']:
            need(source not in released, 'input used after Release')
            used[source] = index
        output = instruction['output']
        need(output not in produced, 'duplicate output')
        produced[output] = (index, opcode)
        if opcode == 'rotation_hoist':
            need(values[output]['kind'] == 'buffer', 'hoist result is not a Buffer')
            need(int(values[output]['buffer_bytes']) > 0, 'empty hoist allocation')
        if opcode == 'rotate_from_hoist':
            need(len(instruction['inputs']) == 2, 'rotation suffix needs source and Hoist')
            source, hoist = instruction['inputs']
            need(values[source]['kind'] == 'ciphertext' and values[hoist]['kind'] == 'buffer', 'suffix input kinds')
            need(produced.get(hoist, (None, None))[1] == 'rotation_hoist', 'suffix Hoist producer')
    elif kind == 'release':
        item = instruction['value']
        need(item not in released, 'duplicate Release')
        released[item] = index
    else:
        raise ValueError(f'unexpected communication or operation: {kind}')
for key, value in values.items():
    if value['kind'] == 'buffer':
        need(key in used and key in released and released[key] > used[key], 'Hoist last-use Release missing')
need(counts.get('bootstrap') == 18, 'expected 18 full native Boots')
need(counts.get('rotation_hoist', 0) > 0 and counts.get('rotate_from_hoist', 0) > counts['rotation_hoist'],
     'no shared hoisted rotations')
result = one('API_TRACE_RESULT ')
for key, expected in {'accuracy': 'PASS', 'exact_rns': 'true', 'bootstraps': '18',
                      'hoist_live_bytes': '0', 'encrypted_logits_q': '3', 'devices': '1',
                      'security_approved': 'false', 'cupti': 'true'}.items():
    need(result.get(key) == expected, f'{key} differs from {expected}: {result.get(key)}')
for key in ('exact_residues', 'trace_exact_residues'):
    need(result[key] == str(2 * 3 * 65536), f'{key} differs')
for key, name in [('calls', None), ('hoists', 'rotation_hoist'), ('rotation_suffixes', 'rotate_from_hoist')]:
    expected = sum(counts.values()) if name is None else counts[name]
    need(int(result[key]) == expected, f'{key} disagrees with plan')
need(int(result['hoist_releases']) == counts['rotation_hoist'], 'not all Hoists released')
need(math.isfinite(float(result['max_logit_error'])) and float(result['max_logit_error']) <= 0.1,
     'logit error failed')
need(result['prediction'] == result['reference_prediction'] == '3', 'prediction differs')
activity = one('GPU_ACTIVITY_TIMING ')
need(activity['host_transfers'] == '0' and activity['host_transfer_bytes'] == '0', 'online transfer')
print(json.dumps({'plan': str(plan_path), 'accuracy': 'PASS', 'exact_rns': True,
                  'counts': counts, 'result': result, 'gpu_activity': activity,
                  'note': 'Boot workspace/resident resources are not included in planned value memory.'}, indent=2))
