#!/usr/bin/env python3
"""Fresh-process first-start A/B cases; no artifact/key cache is used.

The expensive full-CRT oracle case is explicitly separate from latency cases.
Each native probe hashes public constants AFTER its profile preparation timer.
"""
import argparse
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--suite', default='qpfix-20261115')
a = p.parse_args()
d = a.root.resolve()
r = d / a.suite / 'results'
cases = [
    ('reference-r1', 'reference', 1, 0),
    ('optimized-t1-r1', 'optimized', 1, 0),
    ('optimized-t4-r1', 'optimized', 4, 0),
    ('optimized-t6-r1', 'optimized', 6, 0),
    ('oracle-t4', 'optimized', 4, 1),
    ('optimized-t4-r2', 'optimized', 4, 0),
    ('optimized-t6-r2', 'optimized', 6, 0),
    ('optimized-t4-r3', 'optimized', 4, 0),
    ('optimized-t6-r3', 'optimized', 6, 0),
    ('reference-r2', 'reference', 1, 0),
]
for name, mode, workers, verify in cases:
    env = os.environ.copy()
    env.update(POSEIDON_GPU_KEYLOAD_TRACE='1', POSEIDON_GPU_QP_PREPARE=mode,
               POSEIDON_GPU_QP_PREPARE_THREADS=str(workers),
               POSEIDON_GPU_QP_VERIFY_REFERENCE=str(verify))
    command = ['python3', str(d/'src/experiments/gpu_keyload/run_matrix.py'),
               '--binary', str(d/'native_profile'), '--results', str(r), '--case', name,
               '--threads', '1', '--wait', 'PASSIVE', '--affinity', '0-6', '--timeout', '600', '--',
               str(d/'src/experiments/gpu_keyload/public-resnet20-primes.txt'),
               str(r/name/'probe.json'), str(r/name/'public-fingerprints.json')]
    print(f'BEGIN {name}: mode={mode}, workers={workers}, oracle={verify}', flush=True)
    subprocess.run(command, env=env, check=True)
