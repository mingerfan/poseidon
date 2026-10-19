#!/usr/bin/env python3
"""Verify final benchmark samples and compare the archived scheduler versions."""
import hashlib
import json
import argparse
from pathlib import Path
import re
import statistics

out = Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--local-only',action='store_true')
args=parser.parse_args()
repo = out.parents[1]
prior = out.parent / 'waterline-legacy-20261018'
records, rows, inputs = [], [], {}
binary_hashes={}
server_hashes={} if args.local_only else json.loads((out/'server-binary-hashes.json').read_text())

def stats(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values), samples=values)

for host in (('local',) if args.local_only else ('local', 'server')):
    jobs = json.loads((out / (host+'-jobs-final.json')).read_text())
    for job in jobs:
        samples = []
        folders = [out / (host+'-runs') / f"{job['group']}-r{repeat}-{job['name']}" for repeat in range(3)]
        if job['group'] == 'local-mlp' and job['name'] in (
                'legacy-eager','previous-eager','updated-eager','previous-waterline','updated-waterline'):
            folders += [out/'local-mlp-extra-runs'/f"local-mlp-r{repeat}-{job['name']}" for repeat in range(7)]
        for folder in folders:
            data = json.loads((folder/'summary.json').read_text())
            suite=json.loads((Path(job['suite'])/'suite.json').read_text()) if host=='local' else data['suite']
            runner=next(c['runner'] for c in suite['cases'] if c['name']==job['case'])
            if host=='local':
                executable=Path(job['build'])/'bin'/f'poseidon_runtime_gpu_{runner}_e2e'
                if str(executable) not in binary_hashes:
                    binary_hashes[str(executable)]='sha256:'+hashlib.sha256(executable.read_bytes()).hexdigest()
                expected_hash=binary_hashes[str(executable)]
            else:expected_hash=server_hashes[job['build']][runner]
            assert data['runner_hashes'][runner]==expected_hash,(folder,'superseded executable')
            assert len(data['samples']) == 1
            sample = data['samples'][0]
            assert sample['passed'] and sample['exit_code'] == 0 and sample['case'] == job['case']
            report = sample['report']
            timing = report['runtime_timing']
            assert timing['fence_calls'] == 0
            identity = dict(plan=report.get('plan_source_sha256', report.get('plan_sha256')),
                            spec=report['operator_spec_sha256'],
                            fixture=report.get('fixture_sha256', report.get('model_sha256')))
            assert all(identity.values())
            key = job['group'], job['case']
            if key in inputs: assert identity == inputs[key], (key, identity, inputs[key])
            inputs[key] = identity
            sample['source'] = str((folder/'summary.json').relative_to(repo))
            sample['variant'] = job['name'].split('-')[0]
            sample['group'] = job['group']
            log = (folder/f"r0-{job['case']}"/'run.log').read_text()
            if job['case'] != 'eager':
                command = json.loads((folder/f"r0-{job['case']}"/'command.json').read_text())
                plan_path = Path(command[1]) if host == 'local' else out/'server-plans'/job['group']/(job['case']+'.json')
                raw_plan = plan_path.read_bytes()
                assert 'sha256:'+hashlib.sha256(raw_plan).hexdigest() == identity['plan']
                plan = json.loads(raw_plan)
                config = plan['waterline']
                degree = report.get('poly_degree') or report['context']['poly_degree']
                retained = [0]*data['suite']['devices']
                kept = set(plan['external_inputs']) | set(plan['final_outputs'])
                for value in plan['values']:
                    if value['id'] in kept and value['place']['kind'] == 'device':
                        assert value['kind'] == 'ciphertext'
                        retained[value['place']['index']] += degree*(value['level']+1)*value['components']*4
                devices = re.findall(r'waterline GPU(\d+) front_peak=(\d+) plain_peak=(\d+) remaining_front=(\d+) remaining_plain=(\d+)', log)
                assert len(devices) == data['suite']['devices']
                for d, front, plain, front_left, plain_left in devices:
                    d, front, plain = int(d), int(front), int(plain)
                    assert front <= config['device_capacity_bytes'][d]-config['device_reserve_bytes'][d]
                    assert plain <= int(config['device_capacity_bytes'][d]*config['device_fractions'][d])
                    assert int(front_left) == retained[d] and int(plain_left) == 0
                shared = re.search(r'waterline shared host_peak=(\d+) pinned_peak=(\d+) raw_peak=(\d+)', log)
                assert shared
                assert all(int(value) <= config[field+'_budget_bytes'] for value,field in zip(shared.groups(),('host','pinned','raw')))
            if 'device_measurements' in report:
                for device in report['device_measurements']:
                    assert device['allocated_peak_bytes'] <= device['pool_capacity_bytes']
                    assert device['pool_capacity_bytes'] == data['pool_mib']*2**20
                assert report['pinned_transfer_live_after_plan_bytes'] == 0
                assert report['pinned_transfer_live_after_validation_bytes'] == 0
                if job['case'] != 'eager':
                    prep = report['preparation_measurements']
                    count = sum(d['gpu_encode_calls'] for d in report['device_measurements']) + prep['cpu_prepare_calls']
                    expected = data['suite']['count'] + (data['suite']['kind'] == 'qproj')
                    assert count == timing['bundle_read_calls'] == expected
                    if job['name'].startswith('updated') and '-auto-' in job['case']:
                        assert prep['cpu_prepare_calls'] == 0
            samples.append(sample)
            records.append(sample)
        row = dict(group=job['group'], name=job['name'], case=job['case'], repeats=len(samples),
                   **{k:stats([s[k] for s in samples]) for k in ('online_ms','forward_ms','process_cpu_seconds')})
        if 'preparation_measurements' in samples[0]['report']:
            row['cpu_prepare_calls'] = stats([s['report']['preparation_measurements']['cpu_prepare_calls'] for s in samples])
            row['gpu_prepare_calls'] = stats([sum(d['gpu_encode_calls'] for d in s['report']['device_measurements']) for s in samples])
        rows.append(row)

assert len(records) == (128 if args.local_only else 206)
for file, expected in json.loads((out/'final-source-sha256.json').read_text()).items():
    assert hashlib.sha256((repo/file).read_bytes()).hexdigest() == expected, file
old_libs = json.loads((repo/'build-waterline-legacy-20261018/libraries.json').read_text())
new_libs = json.loads((repo/'build-waterline-batch-20261018-ready/libraries.json').read_text())
for path,sha in new_libs.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha,path
    if not path.endswith('libposeidon_runtime_gpu_api.a'): assert old_libs[path] == sha
if not args.local_only:
    old_server = json.loads((out/'server-libraries-before.json').read_text())
    new_server = json.loads((out/'server-libraries-after.json').read_text())
    assert old_server == new_server

lookup = {(r['group'],r['name']):r for r in rows}
table = []
for group in dict.fromkeys(r['group'] for r in rows):
    baseline = lookup[group,'unfixed-waterline']
    values = {name:lookup[group,name]['online_ms']['median']/1000 for name in (
        'legacy-eager','previous-eager','updated-eager','previous-waterline','updated-waterline')}
    table.append(dict(group=group, earliest_waterline_s=baseline['online_ms']['median']/1000,
                      online_s=values,
                      resident_speedup=values['previous-eager']/values['updated-eager'],
                      waterline_speedup=values['previous-waterline']/values['updated-waterline']))

summary = dict(scope='All four Runtime header versions link the same corrected GPU backend on each host: exact historical legacy header, unfixed a7245b8 header, previous db09ac6 header, and final d10aff2 header. Each final configuration has 3 fresh-process samples; five main RTX4060 MLP configurations have 7 additional repetitions, for 10 samples each. Jobs rotate; unchanged baseline samples are retained when only the updated implementation changes. Superseded updated binaries are detected by executable hash and their samples archived before replacement. All main columns, including earliest unfixed Waterline, use the same Ready-state-race-fixed GPU backend. Pilot, pre-Ready and superseded updated samples are excluded.',
    verified_samples=len(records), configurations=len(rows), comparisons=rows, table=table,
    checks=['All outputs passed independent numerical validation.',
            'Plan/spec/fixture hashes match across variants of each exact case.',
            'Device ledger retains exactly returned/external ciphertexts and no plaintext credit; all byte peaks satisfy their individual budgets.',
            'Measured managed-pool peaks stay within the identical configured physical pool caps.',
            'Pressure/q_proj Prepare and bundle-read counts match each plaintext exactly once.',
            'All updated Auto cases used GPU preparation only; explicit backend choices were unchanged.',
            'No implicit Fence was introduced; pinned transfer live bytes returned to zero.',
            'Final source hashes match; live GPU library hashes match the snapshots shared by all Runtime versions. Superseded updated executable hashes are excluded.'],
    caveats=['Original means the exact historical Runtime header with current runners, verifier and GPU libraries, not a historical whole-repository rebuild.',
             'Eager online excludes prior encoding/upload; Waterline online includes preparation. Both include final synchronization/drain.',
             'First forward includes initialization; keys and input encryption are separate.',
             'CPU process time includes context creation, key generation and validation; it is not online-only CPU cost.',
             'Byte-peak front and plain maxima need not be simultaneous; their sum is not used as a total-memory assertion.',
             'q_proj is a synthetic proxy, not the complete Qwen model. Full four-GPU Qwen was not rerun.',
             'OS file cache was not flushed and GPU clocks/CPU affinity were not fixed.'],
    input_identities=[dict(group=g,case=c,**v) for (g,c),v in inputs.items()])
prefix='local-' if args.local_only else ''
(out/(prefix+'comparison.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
lines=['All timings below are online medians in seconds (3 fresh processes per final configuration; 10 for five main RTX4060 MLP configurations).',
       'All columns were freshly measured with the same corrected GPU backend on each host.','',
       'workload | original resident | previous resident | updated resident | earliest unfixed Waterline | previous Waterline | updated Waterline']
for r in table:
    v=r['online_s']
    lines.append(r['group']+' | '+' | '.join(f'{x:.6f}' for x in (
        v['legacy-eager'],v['previous-eager'],v['updated-eager'],r['earliest_waterline_s'],v['previous-waterline'],v['updated-waterline'])))
lines+=['','All final configurations (online median [min,max], seconds):']
for r in rows:
    m=r['online_ms'];lines.append(f"{r['group']} {r['name']}: {m['median']/1000:.6f} [{m['min']/1000:.6f}, {m['max']/1000:.6f}]")
(out/(prefix+'measurements.txt')).write_text('\n'.join(lines)+'\n')
print(f'Verified {len(records)} samples across {len(rows)} configurations.')
print('\n'.join(lines[:10]))
