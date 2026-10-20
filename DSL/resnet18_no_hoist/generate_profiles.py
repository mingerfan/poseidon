#!/usr/bin/env python3
"""Bind compiler capabilities to a passed framework Native no-hoist probe.

Latency estimates are inherited from the framework, not new measurements.
This configures the backend; the Python model contains no CKKS schedule.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root', type=Path, required=True)
    p.add_argument('--native-probe-log', type=Path, required=True)
    p.add_argument('--compiler-profile-template', type=Path,
                   help='matching Dacapo profiled_HEAAN_GPU.json when using separate build directories')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('use a new output directory')
    text = args.native_probe_log.read_text()
    if '[PASS] RuntimePlan native GPU bootstrap' not in text or 'NATIVE_NOHOIST_PROBE_EXIT=0' not in text:
        raise ValueError('Native no-hoist component must pass before generating capabilities')

    def fields(prefix):
        rows = [line for line in text.splitlines() if line.startswith(prefix+' ')]
        if len(rows) != 1:
            raise ValueError('expected exactly one '+prefix)
        return dict(item.split('=', 1) for item in rows[0].split()[1:])

    boot, context = fields('NATIVE_BOOT_PROFILE'), fields('NATIVE_BOOT_CONTEXT')
    if (int(boot['linear_mode']) != 4 or int(boot['evalmod_degree']) != 58 or
            int(context['degree']) != 65536):
        raise ValueError('expected degree65536, NoHoistBsgs and the complete cosine fit')
    q_bits = [int(v) for v in context['q_bits'].split(',') if v]
    p_bits = [int(v) for v in context['p_bits'].split(',') if v]
    upper, minimum = int(boot['output_level']), int(boot['input_level_min'])
    factor = math.ceil((int(boot['output_scale_log2'])+30)/min(q_bits[:upper+1]))
    logical_upper = (upper-minimum)//factor
    if logical_upper < 3:
        raise ValueError('Native output chain cannot cover the compiler bootstrap budget')
    runtime = args.runtime_root.resolve()
    sys.path.insert(0, str(runtime/'integrations/dacapo'))
    import generate_mlp_gpu_e2e_profiles as base
    spec = base.make_operator_spec(base.DEFAULT_OPERATOR_SPEC, 65536, 'poseidon-65536')
    spec['spec_id'] = 'poseidon-resnet20-frontend-native-nohoist-v1'
    spec['context'] = dict(context_id=spec['spec_id'], poly_degree=65536,
        rns_moduli_log2=q_bits, max_modulus_log2=max(q_bits),
        default_scale_log2=int(context['default_scale_log2']))
    spec['levels'] = dict(lower_bound=upper-factor*logical_upper, upper_bound=upper)
    costs = base.operator_latencies('poseidon-65536', len(q_bits))
    for name, cost in costs.items():
        spec['operators'][name]['latency_us_by_level'] = cost[:upper+1]
    spec['operators']['rescale']['max_levels_per_op'] = max(4, factor)
    spec['boot_profiles'] = [dict(profile_id=boot['id'], implementation='native',
        input_level_min=spec['levels']['lower_bound'], input_level_max=upper,
        input_components=2, output_level=upper,
        output_scale_log2=int(boot['output_scale_log2']), output_components=2,
        latency_us_by_input_level=[base.BOOT_LATENCY_US]*(upper+1),
        noise_by_input_level=None,
        host_requirements=dict(needs_secret_key=False, needs_host_compute=False))]
    spec['provenance'] = dict(kind='test-fixture', repository='poseidon',
        revision='6dc8cbfb65e3311271563000f6f5dce92d0324a8',
        path='scripts/resnet20_dsl/generate_profiles.py',
        source_sha256='sha256:'+hashlib.sha256(args.native_probe_log.read_bytes()).hexdigest())
    profile = json.loads((args.compiler_profile_template or base.DEFAULT_COMPILER_PROFILE).read_text())
    profile.update(polynomialDegree=65536, rescalingFactor=int(boot['output_scale_log2']),
        levelLowerBound=0, levelUpperBound=logical_upper,
        bootstrapLevelLowerBound=1, bootstrapLevelUpperBound=logical_upper)
    profile['latencyTable'] = {name:[max(1,int(v)) for v in table[:logical_upper+1]]
        for name,table in profile['latencyTable'].items()}
    config = dict(profile_id=boot['id'], q0_level=minimum, p_bits=p_bits,
        cost_profile_note='Native capabilities tested; operation latencies remain framework estimates',
        hamming_weight=int(context['hamming_weight']),
        linear_mode=4, evalmod_degree=58, generation_degree=59,
        evalmod_log_scale=int(boot['evalmod_log_scale']),
        project_real=bool(int(boot['project_real'])),
        logical_rescale_level_factor=factor, waterline=40,
        native_probe_sha256=spec['provenance']['source_sha256'])
    assert boot['id']=='poseidon-gpu-degree58-da3-full-cosine-scale60-ratio128-real'
    assert int(boot['evalmod_log_scale'])==60 and int(boot['output_level'])==15
    config['bootstrap_ratio'] = 128
    args.output.mkdir(parents=True)
    for name, value in [('operator-spec',spec),('compiler-profile',profile),('native-config',config)]:
        (args.output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(config,indent=2))


if __name__ == '__main__':
    main()
