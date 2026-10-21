#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import re

base = Path(__file__).resolve().parent
root = base.parent.parent
phases = []
for phase in ['before', 'stream', 'normalized']:
    r = json.loads((base / 'released-inputs-batch' / f'report-{phase}.json').read_text())
    phases.append(dict(phase=phase, passed_outputs=sum(o['passed'] for o in r['outputs']),
                       outputs=len(r['outputs']), comparison=r['comparison']))
cases = []
for case in ['released-inputs-batch', 'direct-input-controls', 'greedy-depth50', 'depth-dp-depth50']:
    r = json.loads((base / case / 'report-normalized.json').read_text())
    assert r['passed'] and r['comparison']['passed']
    assert r['host_boot_instructions'] == 0 and not r['secret_key_assisted_refresh']
    assert r['cuda_visible_devices'] == 'GPU-0768eaf5-15f4-6cca-f263-141df9bae9c6'
    assert r['tolerances'] == dict(atol=.001, rtol=.0001)
    for field, filename in [('fixture_sha256', 'fixture.json'),
                            ('operator_spec_sha256', 'operator-spec.json'),
                            ('plan_source_sha256', 'result.chains.runtime-plan.json')]:
        assert r[field] == 'sha256:' + hashlib.sha256((base / case / filename).read_bytes()).hexdigest()
    cases.append(dict(case=case, boots=r['native_boot_instructions'],
                      outputs=r['output_count'], comparison=r['comparison'],
                      output_states=[dict(level=o['level'], scale_log2=o['scale_log2'])
                                     for o in r['outputs']]))
api_log = (base / 'gpu-api-tests.log').read_text()
api_count = int(re.search(r'(\d+) Poseidon GPU Runtime Api tests passed', api_log)[1])
bootstrap_log = (base / 'gpu-bootstrap-e2e.log').read_text()
assert '[PASS] RuntimePlan native GPU bootstrap' in bootstrap_log
assert 'output_level=22 output_scale_log2=45' in bootstrap_log
sources = ['src/poseidon/gpu/gpu_evaluator.cpp', 'src/poseidon/gpu/gpu_bootstrap_profile.cpp',
           'src/poseidon/runtime_api/poseidon_gpu_api.cpp']
summary = dict(
    base_commit='a2759448', stream_fix_commit='8d57bb9e',
    server='188Server', physical_gpu=2, gpu_uuid='GPU-0768eaf5-15f4-6cca-f263-141df9bae9c6',
    omp_num_threads=1, pool_mb=12288, security_level='none; numerical validation only',
    profile_id='poseidon-gpu-degree58-da2-scale45', output_level=22, output_scale_log2=45,
    builder_input_min=2, declared_input_min=4, available_output_q=18,
    phases=phases, cases=cases, boot_calls=sum(c['boots'] for c in cases),
    output_count=sum(c['outputs'] for c in cases),
    slots_checked=sum(c['comparison']['slots_checked'] for c in cases),
    gpu_api_tests=api_count, normalization_states=api_log.count('BOOT_INPUT_NORMALIZATION input_level='),
    independent_complex_bootstrap=bootstrap_log.strip(),
    source_sha256={s: hashlib.sha256((root / s).read_bytes()).hexdigest() for s in sources},
    limits=['Residual addition precision compensation is deferred.',
            'The original optimized/PARS path is not adapted.',
            'These Boot and multiplication regressions are not complete Qwen validation.'])
(base / 'validation-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: summary[k] for k in ['boot_calls', 'output_count', 'slots_checked', 'gpu_api_tests', 'normalization_states']}, indent=2))
