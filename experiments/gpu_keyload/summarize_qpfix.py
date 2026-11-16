#!/usr/bin/env python3
"""Check first-start A/B correctness and summarize only completed fresh runs."""
import argparse
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('results', type=Path)
p.add_argument('--require-complete', action='store_true')
a = p.parse_args()
expected = ['reference-r1', 'optimized-t1-r1', 'optimized-t4-r1', 'optimized-t6-r1',
            'oracle-t4', 'optimized-t4-r2', 'optimized-t6-r2',
            'optimized-t4-r3', 'optimized-t6-r3', 'reference-r2']

def distribution(values):
    return {'samples': len(values), 'median_s': statistics.median(values),
            'min_s': min(values), 'max_s': max(values)}

def trace(path, prefix):
    return [json.loads(line[len(prefix):]) for line in path.read_text().splitlines()
            if line.startswith(prefix)]

rows = []
canonical = None
canonical_inventory = None
for name in expected:
    d = a.results/name
    if not (d/'launch.json').exists():
        if a.require_complete:
            raise AssertionError(f'missing completed run: {name}')
        continue
    launch = json.loads((d/'launch.json').read_text())
    assert launch['exit_code'] == 0 and not launch['timed_out'], name
    report = json.loads((d/'probe.json').read_text())
    public = json.loads((d/'public-fingerprints.json').read_text())
    assert (public['c2s_diagonals'], public['s2c_diagonals'], public['qp_diagonals']) == (158,158,316)
    assert public['all_public_plaintext_words_hashed'] and not public['key_payloads_hashed']
    if canonical is None:
        canonical = public['public_fingerprint']
        canonical_inventory = public['public_inventory']
    assert public['public_fingerprint'] == canonical, f'public word/metadata digest mismatch: {name}'
    assert public['public_inventory'] == canonical_inventory, f'public metadata mismatch: {name}'
    phases = {x['phase']: x['wall_s'] for x in trace(d/'stderr.log', 'BOOT_SETUP_PHASE ')}
    qp = trace(d/'stderr.log', 'QP_UPLOAD_PROFILE ')
    assert len(qp) == 2 and sum(x['diagonals'] for x in qp) == 316, name
    verified = sum(x.get('reference_verified_plaintexts', 0) for x in qp)
    if name == 'oracle-t4':
        assert verified == 316, 'not all full-profile Q/P arrays were bytewise checked against full CRT'
    else:
        assert verified == 0, 'expensive oracle leaked into latency case'
    env = launch['selected_environment']
    assert env['OMP_NUM_THREADS'] == '1' and env['POSEIDON_GPU_QP_VERIFY_REFERENCE'] == ('1' if name=='oracle-t4' else '0')
    assert not report['bootstrap_executed']
    rows.append({'case': name, 'native_profile_s': report['native_profile_s'],
        'process_wall_s_including_fingerprint_and_teardown': launch['process_wall_s'],
        'post_timer_fingerprint_s': public['validation_s'],
        'cpu_context_s': report['context_s'], 'secret_key_s': report['secret_key_s'],
        'qp_group_wall_s': phases['c2s_qp_diagonals_pack_and_upload'] + phases['s2c_qp_diagonals_pack_and_upload'],
        'c2s_wall_s': phases['c2s_qp_diagonals_pack_and_upload'],
        's2c_wall_s': phases['s2c_qp_diagonals_pack_and_upload'],
        'galois_keygen_s': phases['galois_keygen_cpu'],
        'relin_keygen_s': phases['relin_keygen_cpu'],
        'dft_encode_s': phases['dft_matrices_cpu'],
        'evalmod_encode_s': phases['evalmod_constants_encode_and_upload'],
        'fast_plaintexts': sum(x.get('fast_plaintexts',0) for x in qp),
        'full_crt_plaintexts': sum(x.get('full_crt_plaintexts',0) for x in qp),
        'reference_verified_plaintexts': verified,
        'worker_limit': [x['worker_limit'] for x in qp],
        'qp_uploaded_bytes': sum(x['uploaded_bytes'] for x in qp),
        'public_plaintext_payload_bytes': public['public_plaintext_payload_bytes'],
        'public_fingerprint': public['public_fingerprint']})

summary = {'scope': 'No disk/key cache. First native manual-profile preparation only; not online Boot or entire model latency.',
           'public_fingerprint_all_completed_runs': canonical,
           'public_fingerprint_is_noncollisionproof': True,
           'full_316_diagonal_bytewise_oracle_passed': any(x['case']=='oracle-t4' for x in rows),
           'runs': rows, 'groups': {}}
for label, prefix in [('reference', 'reference-r'), ('optimized_t1', 'optimized-t1-r'),
                      ('optimized_t4', 'optimized-t4-r'), ('optimized_t6', 'optimized-t6-r')]:
    xs = [x for x in rows if x['case'].startswith(prefix)]
    if xs:
        summary['groups'][label] = {key: distribution([x[key] for x in xs])
            for key in ('native_profile_s', 'qp_group_wall_s', 'c2s_wall_s', 's2c_wall_s',
                        'galois_keygen_s', 'dft_encode_s', 'evalmod_encode_s',
                        'process_wall_s_including_fingerprint_and_teardown')}
ref = summary['groups'].get('reference')
if ref:
    baseline = ref['native_profile_s']['median_s']
    for label, values in summary['groups'].items():
        if label != 'reference':
            median = values['native_profile_s']['median_s']
            values['saved_s'] = baseline-median
            values['speedup'] = baseline/median
            values['reduction_percent'] = 100*(1-median/baseline)
print(json.dumps(summary, indent=2))
