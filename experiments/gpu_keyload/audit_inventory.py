#!/usr/bin/env python3
"""Static Cerium compiler-inventory and Poseidon parameter-root byte audit."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('cerium_program_inputs')
p.add_argument('--log-n', type=int, default=16)
p.add_argument('--poseidon-q', type=int, default=50)
p.add_argument('--poseidon-p', type=int, default=25)
a = p.parse_args()
source = Path(a.cerium_program_inputs)
raw = source.read_bytes()
active = False
terms = 0
keys = {}
groups = set()
rotations = set()
for line in raw.decode().splitlines():
    if line.strip() == 'Evalkey Stream:':
        active = True
        continue
    if not active:
        continue
    if line.strip() == ';':
        break
    if not line.strip():
        continue
    terms += 1
    shape = line.split('|')[1].strip().split(':')
    if len(shape) != 7 or shape[0] != 'K':
        raise ValueError(f'Unknown evalkey syntax: {line}')
    typ, q, ext, rotation = map(int, shape[1:5])
    identity = (typ, q, ext, rotation, shape[6])
    keys[identity] = 2 * (1 << a.log_n) * 4 * (q + ext)
    groups.add((typ, q, ext, rotation))
    if typ in (1, 6):
        rotations.add(rotation)
counts = collections.Counter(k[0] for k in keys)
L, P, N = a.poseidon_q, a.poseidon_p, 1 << a.log_n
roots = 7 * 4 * N * ((L + P) + L * (L + 1) // 2 + L * P)
print(json.dumps({
    'scope': 'Static allocation/payload formulas, not measured resident files or timings',
    'source': str(source), 'source_sha256': hashlib.sha256(raw).hexdigest(),
    'degree': N, 'cerium_term_references': terms,
    'cerium_unique_digit_key_pairs': len(keys), 'cerium_logical_key_groups': len(groups),
    'cerium_distinct_rotations': len(rotations), 'cerium_digit_pairs_by_type': dict(counts),
    'cerium_key_payload_bytes': sum(keys.values()),
    'poseidon_q': L, 'poseidon_p': P,
    'poseidon_full_chain_seven_root_arrays_bytes': roots,
    'poseidon_one_full_gpu_key_bytes': 2 * ((L + P - 1) // P) * N * (L + P) * 4,
}, indent=2))
