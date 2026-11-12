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
compute_nodes = {}
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
        compute_nodes[output] = (opcode, callee, instruction['inputs'])
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
    elif key in produced and key in used and key not in plan['final_outputs']:
        need(key in released and released[key] > used[key], 'ciphertext last-use Release missing')
need(counts.get('bootstrap', 0) == 0, 'opaque native Boot remains in the API trace')
need(counts.get('bootstrap_raise') == 18, 'expected one raise stage per Boot')
need(counts.get('bootstrap_linear_layer') == 18 * 6, 'expected six linear layers per Boot')
need(counts.get('eval_mod_high_precision') == 18 * 2, 'expected two EvalMod stages per Boot')

# Follow SSA dependencies rather than inferring stage order from schedule order:
# independent EvalMod branches or unrelated network work may be reordered.
def closest_ancestors(value, opcodes):
    pending, seen, found = [value], set(), set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        node = compute_nodes.get(current)
        if node is None:
            continue
        opcode, _, inputs = node
        if opcode in opcodes:
            found.add(current)
        else:
            pending.extend(inputs)
    return found


def unique_predecessor(value, opcode, label):
    found = closest_ancestors(value, {opcode})
    need(len(found) == 1, f'{label}: expected exactly one preceding {opcode}, got {found}')
    return next(iter(found))


def layer_tag(value, direction, index):
    need(f'.cts{direction}.layer{index}' in compute_nodes[value][1],
         f'wrong Boot linear layer: {compute_nodes[value][1]}')


raises = [v for v, n in compute_nodes.items() if n[0] == 'bootstrap_raise']
for raise_value in raises:
    current = unique_predecessor(raise_value, 'bootstrap_linear_layer', 'Boot S2C')
    for layer in (2, 1, 0):
        layer_tag(current, 0, layer)
        if layer:
            current = unique_predecessor(compute_nodes[current][2][0], 'bootstrap_linear_layer', 'Boot S2C')

eval_by_raise = {}
for value, node in compute_nodes.items():
    if node[0] != 'eval_mod_high_precision':
        continue
    current = unique_predecessor(node[2][0], 'bootstrap_linear_layer', 'Boot C2S')
    split_source = current
    for layer in (2, 1, 0):
        layer_tag(current, 1, layer)
        if layer:
            current = unique_predecessor(compute_nodes[current][2][0], 'bootstrap_linear_layer', 'Boot C2S')
    raise_value = unique_predecessor(compute_nodes[current][2][0], 'bootstrap_raise', 'Boot C2S')
    eval_by_raise.setdefault(raise_value, []).append((value, node[2][0], split_source))
need(set(eval_by_raise) == set(raises) and all(len(branches) == 2 for branches in eval_by_raise.values()),
     'each Boot raise must feed exactly two EvalMod branches through its C2S layers')
needed, pending = set(), list(plan['final_outputs'])
while pending:
    current = pending.pop()
    if current in needed:
        continue
    needed.add(current)
    if current in compute_nodes:
        pending.extend(compute_nodes[current][2])
for branches in eval_by_raise.values():
    left, right = branches
    need(left[1] != right[1] and left[2] == right[2], 'EvalMod must use distinct branches from one C2S')
    need(left[0] in needed and right[0] in needed, 'final result ignores an EvalMod branch')
need(counts.get('rotation_hoist', 0) > 0 and counts.get('rotate_from_hoist', 0) > counts['rotation_hoist'],
     'no shared hoisted rotations')
result = one('API_TRACE_RESULT ')
for key, expected in {'accuracy': 'PASS', 'exact_rns': 'true', 'bootstraps': '18',
                      'boot_layers': '108', 'boot_eval_mods': '36',
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
