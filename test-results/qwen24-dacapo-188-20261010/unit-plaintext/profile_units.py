"""Measure repeated unit encodings in the existing complete native plan."""
from array import array
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import struct
import time

ROOT = Path('/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1')
BASE = ROOT / 'artifacts/qwen24-native'
started = time.monotonic()
resource.setrlimit(resource.RLIMIT_AS, (128 * 2**30,) * 2)
manifest = json.loads((BASE / 'qwen24.depth-dp._hecate_qwen25_24layer.bundle/manifest.json').read_text())
lengths = {entry['byte_length'] // 8 for entry in manifest['blobs']}
units = {'sha256:' + hashlib.sha256(struct.pack('<d', 1.0) * size).hexdigest(): size
         for size in lengths}
plan = json.loads((BASE / 'qwen24.depth-dp._hecate_qwen25_24layer.runtime-plan.json').read_text())
values = plan['values']
uses = array('I', [0]) * len(values)
mul_cp = bytearray(len(values))
unit = bytearray(len(values))
histogram = Counter()
keys = set()
rns_bytes = 0
counts = Counter()
for step in plan['initialization']:
    payload = step['payload']
    if payload['kind'] == 'bundle':
        size = units.get(payload['content'], 0)
    else:
        data = payload['values']
        size = len(data) if all(value == 1.0 for value in data) else 0
    if not size:
        continue
    index = int(step['output'])
    value = values[index]
    kind = 'full_slot' if size == 32768 else 'prefix_mask'
    counts[kind] += 1
    unit[index] = 1 if kind == 'full_slot' else 2
    key = (kind, size, value['scale_log2'], value['level'])
    histogram[key] += 1
    keys.add(key)
    if kind == 'full_slot':
        rns_bytes += 65536 * 4 * (value['level'] + 1)
for step in plan['execution']:
    index = int(step['output'])
    mul_cp[index] = step['op'] == 'mul_cp'
    for source in step['inputs']:
        uses[int(source)] += 1
for source in plan['final_outputs']:
    uses[int(source)] += 1
for step in plan['execution']:
    if step['op'] != 'mul_cp' or unit[int(step['inputs'][1])] != 1:
        continue
    counts['full_unit_mul_cp'] += 1
    source = int(step['inputs'][0])
    if mul_cp[source] and uses[source] == 1:
        counts['full_unit_after_single_use_mul_cp'] += 1
report = {'counts': dict(counts), 'full_unit_encoded_rns_gpu_bytes': rns_bytes,
          'distinct_unit_types': len(keys),
          'histogram': [dict(kind=k[0], payload_slots=k[1], scale_log2=k[2], level=k[3], count=v)
                        for k, v in sorted(histogram.items())],
          'seconds': time.monotonic() - started}
out = ROOT / 'artifacts/qwen24-unit-plaintext'
out.mkdir(exist_ok=True)
(out / 'before-profile.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'histogram'}), flush=True)
