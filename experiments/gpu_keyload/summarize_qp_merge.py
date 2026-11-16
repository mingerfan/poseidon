#!/usr/bin/env python3
"""Validate A800 pre-merge review and final-source first-start results.

usage: summarize_qp_merge.py REVIEW_RESULTS FINAL_RESULTS
The reference sample is from REVIEW_RESULTS; FINAL_RESULTS contains the final
optimized and exact-oracle runs. No cached keys/constants are involved.
"""
import json
from pathlib import Path
import sys

review, final = map(Path, sys.argv[1:3])
cases = [(review, 'reference-r1'), (review, 'optimized-t4-r1'), (review, 'oracle-t4'),
         (final, 'optimized-t4-r1'), (final, 'oracle-t4')]
rows = []
expected_fingerprint = None
expected_inventory = None
for directory, name in cases:
    d = directory/name
    launch = json.loads((d/'launch.json').read_text())
    assert launch['exit_code'] == 0 and not launch['timed_out'], (directory, name)
    result = json.loads((d/'probe.json').read_text())
    public = json.loads((d/'public-fingerprints.json').read_text())
    assert public['qp_diagonals'] == 316 and public['c2s_diagonals'] == public['s2c_diagonals'] == 158
    assert public['all_public_plaintext_words_hashed'] and not public['key_payloads_hashed']
    if expected_fingerprint is None:
        expected_fingerprint = public['public_fingerprint']
        expected_inventory = public['public_inventory']
    assert public['public_fingerprint'] == expected_fingerprint, (directory, name)
    assert public['public_inventory'] == expected_inventory, (directory, name)
    trace = (d/'stderr.log').read_text().splitlines()
    prefix = 'QP_UPLOAD_PROFILE '
    qp = [json.loads(line[len(prefix):]) for line in trace if line.startswith(prefix)]
    assert len(qp) == 2 and sum(x['diagonals'] for x in qp) == 316
    verified = sum(x.get('reference_verified_plaintexts', 0) for x in qp)
    assert verified == (316 if name == 'oracle-t4' else 0), (directory, name, verified)
    rows.append({'suite':directory.parent.name, 'case':name, 'native_profile_s':result['native_profile_s'],
                 'process_wall_s_including_fingerprint':launch['process_wall_s'],
                 'fast_plaintexts':sum(x['fast_plaintexts'] for x in qp),
                 'reference_verified_plaintexts':verified,
                 'public_fingerprint':public['public_fingerprint'],
                 'public_plaintext_words':public['public_plaintext_words'],
                 'gpu_model':launch['initial_gpu']['gpu'],
                 'selected_environment':launch['selected_environment']})

baseline, optimized = rows[0]['native_profile_s'], rows[3]['native_profile_s']
print(json.dumps({'scope':'new-process native manual Q50/P25 profile; no disk key/constant cache',
                  'public_fingerprint':expected_fingerprint,'noncollisionproof':True,
                  'full_crt_wordwise_oracle_316_diagonals':True,
                  'single_sample_reference_s':baseline,'single_sample_final_optimized_s':optimized,
                  'single_sample_speedup':baseline/optimized,'single_sample_saved_s':baseline-optimized,
                  'runs':rows}, indent=2))
