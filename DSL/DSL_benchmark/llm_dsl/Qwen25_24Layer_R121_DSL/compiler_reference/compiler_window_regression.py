"""Compile small graphs with pinned/candidate compilers and verify both plans.

Scalar interpretation checks algebra only. Native PlanVerifier checks the real
metadata contract. This is not a CKKS accuracy test and does not delete scratch.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import subprocess
import struct

C = 'tensor<1x!earth.ci<0 * 0>>'
P = 'tensor<1x!earth.pl<0 * 0>>'


def graph(depth, extra_output=False, fanout=False):
    results = f'({C}, {C})' if extra_output else C
    lines = [f'module {{ func.func @probe(%x: {C}) -> {results} {{',
        f'%p = "earth.constant"() {{value = dense<1.01> : tensor<1xf64>, rms_var = 1.01 : f64}} : () -> {P}']
    value, decoded, short = '%x', .25, None
    for i in range(depth):
        new = f'%m{i}'
        lines.append(f'{new} = "earth.mul"({value}, %p) : ({C}, {P}) -> {C}')
        value, decoded = new, decoded * 1.01
        if i == 1:
            short = (value, decoded)
    if fanout:
        previous, total = '%x', .25
        for i in range(8):
            coefficient = 1.02+i/100
            lines.extend([
                f'%q{i} = "earth.constant"() {{value = dense<{coefficient}> : tensor<1xf64>, rms_var = {coefficient} : f64}} : () -> {P}',
                f'%b{i} = "earth.mul"({value}, %q{i}) : ({C}, {P}) -> {C}',
                f'%c{i} = "earth.mul"(%b{i}, %p) : ({C}, {P}) -> {C}',
                f'%s{i} = "earth.add"({previous}, %c{i}) : ({C}, {C}) -> {C}'])
            previous, total = f'%s{i}', total+decoded*coefficient*1.01
        value, decoded = previous, total
    lines.append(f'%y = "earth.add"({value}, %x) : ({C}, {C}) -> {C}')
    decoded += .25
    if extra_output:
        lines.append(f'return {short[0]}, %y : {C}, {C}')
        expected = [short[1], decoded]
    else:
        lines.append(f'return %y : {C}')
        expected = [decoded]
    return '\n'.join(lines+['}}']), expected


def bundle_scalars(directory):
    result={}
    for path in directory.glob('*.bundle/manifest.json'):
        manifest=json.loads(path.read_text())
        assert manifest['bundle_format_version']==2
        raw=(path.parent/'data.bin').read_bytes()
        assert len(raw)==manifest['pack_byte_length']
        for blob in manifest['blobs']:
            start,size=blob['offset'],blob['byte_length']
            assert 0<=start and size>0 and size%8==0 and start+size<=len(raw)
            payload=raw[start:start+size]
            content='sha256:'+hashlib.sha256(payload).hexdigest()
            assert content==blob['content']
            values=struct.unpack('<'+str(size//8)+'d',payload)
            assert all(math.isfinite(x) and x==values[0] for x in values), 'Scalar regression requires uniform payload'
            result[content]=values[0]
    return result


def interpret(plan,bundles=None):
    decoded = {plan['external_inputs'][0]: .25}
    counts = collections.Counter()
    for section in ('initialization', 'execution', 'finalization'):
        for op in plan[section]:
            if op['kind'] == 'release':
                continue  # Lifetime legality is independently checked by PlanVerifier.
            kind = op.get('op', op['kind']);counts[kind] += 1
            if kind == 'encode':
                payload=op['payload']
                if payload['kind']=='bundle':
                    assert bundles is not None
                    value=bundles[payload['content']]
                else:
                    values=payload['values']
                    assert len(values)==1
                    value=values[0]
            else:
                values = [decoded[x] for x in op['inputs']]
                if kind in ('mul_cp', 'mul_cc'):
                    value = values[0]*values[1]
                elif kind in ('add_cp', 'add_cc'):
                    value = values[0]+values[1]
                elif kind == 'negate':
                    value = -values[0]
                else:
                    assert kind in ('boot', 'rescale', 'mod_switch', 'relinearize'), kind
                    value = values[0]
            decoded[op['output']] = value
    return [decoded[x] for x in plan['final_outputs']], dict(counts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--verifier', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.directory.exists() and not a.output.exists()
    a.directory.mkdir()
    spec = json.loads(a.spec.read_text());profile = spec['boot_profiles'][0]
    cfg = json.loads(a.config.read_text())
    assert cfg['rescalingFactor'] == spec['context']['default_scale_log2'] == 43
    cases = [(f'chain{n}', *graph(n)) for n in (6, 9, 12, 17)]
    cases += [('return_only', *graph(17, extra_output=True)), ('fanout', *graph(3, fanout=True))]
    rows = []
    for waterline in (43, 40):
        for name, source, expected in cases:
            row = dict(case=name, waterline=waterline)
            for mode, compiler in (('baseline', a.baseline), ('candidate', a.candidate)):
                work = a.directory/f'{waterline}-{name}-{mode}';work.mkdir()
                input_path, output_path = work/'input.mlir', work/'result.mlir'
                input_path.write_text(source)
                command = [str(compiler), str(input_path), '--dacapo', '--boot-placement=depth-dp',
                    f'--waterline={waterline}', f'--ckks-config={a.config}',
                    '--runtime-plan-format=json', '--runtime-plan-memory=true',
                    '--runtime-plan-memory-reuse=false', '--runtime-plan-device-count=0',
                    '--runtime-plan-id=2026102101', '--runtime-plan-target-id='+spec['target_id'],
                    '--runtime-plan-context-id='+spec['context']['context_id'],
                    '--runtime-plan-operator-spec-id='+spec['spec_id'],
                    '--runtime-plan-operator-spec-sha256=sha256:'+hashlib.sha256(a.spec.read_bytes()).hexdigest(),
                    '--runtime-plan-boot-profile='+profile['profile_id'],
                    '--runtime-plan-boot-implementation=native', '--mlir-disable-threading',
                    '-o', str(output_path)]
                with (work/'compile.log').open('xb') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
                plan_path, = work.glob('*.runtime-plan.json')
                with (work/'verify.log').open('xb') as log:
                    subprocess.run([a.verifier, plan_path, a.spec], stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=60)
                constants=bundle_scalars(work)
                actual, counts = interpret(json.loads(plan_path.read_text()),constants)
                assert len(actual) == len(expected)
                assert all(math.isclose(x,y,rel_tol=1e-11,abs_tol=1e-11) for x,y in zip(actual,expected))
                row[mode] = dict(counts=counts, actual=actual, expected=expected,
                    plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(), native_verified=True,
                    verified_bundle_contents=sorted(constants))
            if waterline == 40:
                assert row['baseline']['counts'] == row['candidate']['counts'], row
            rows.append(row)
    reduced = [r['case'] for r in rows if r['waterline']==43 and
               r['candidate']['counts'].get('boot',0)<r['baseline']['counts'].get('boot',0)]
    assert reduced, 'No intended equal-scale budget effect observed'
    result = dict(passed=True, paired_cases=len(rows), cases=rows,
        equal_scale_reduced_cases=reduced, unequal_scale_counts_unchanged=True,
        baseline_sha256=hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
        scope='Small compiled graph semantics and native PlanVerifier; not CKKS precision')
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


if __name__ == '__main__':
    main()
