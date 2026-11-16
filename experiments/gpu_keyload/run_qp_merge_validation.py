#!/usr/bin/env python3
"""Post-review A800 revalidation: reference, optimized, then full-CRT oracle.

Every case creates fresh keys and constants; no saved keys or constant cache.
Each case writes into a new directory and refuses to overwrite existing data.
"""
import argparse
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--suite', default='qp-merge-validation-20261116')
p.add_argument('--cases', nargs='+', choices=['reference-r1', 'optimized-t4-r1', 'oracle-t4'],
               help='Run only these cases; default runs all three')
a = p.parse_args()
d = a.root.resolve()
r = d / a.suite / 'results'
for name, mode, workers, verify in [('reference-r1', 'reference', 1, 0),
                                    ('optimized-t4-r1', 'optimized', 4, 0),
                                    ('oracle-t4', 'optimized', 4, 1)]:
    if a.cases and name not in a.cases:
        continue
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
