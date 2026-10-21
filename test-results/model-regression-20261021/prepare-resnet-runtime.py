#!/usr/bin/env python3
"""Prepare equal-width N65536 profiles for compilation/FP64 replay only."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess

base = Path('/home/xuming/poseidon-model-regression-20261021')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--before', action='store_true')
args = parser.parse_args()
suffix = '-before' if args.before else ''


def prime(n):
    d, r = n-1, 0
    while d % 2 == 0: d //= 2; r += 1
    for a in [2, 3, 5, 7, 11, 13, 17]:
        x = pow(a, d, n)
        if x in (1, n-1): continue
        for _ in range(r-1):
            x = x*x % n
            if x == n-1: break
        else: return False
    return True


def moduli(degree):
    # CoeffModulus::Create(degree, [30]*58): descending search, pop from back.
    factor = 2*degree
    n = ((1 << 30)-1)//factor*factor+1
    values = []
    while len(values) < 58:
        if prime(n): values.append(n)
        n -= factor
    return list(reversed(values))[:50]


original = json.loads((base/('qwen-prefix-before-greedy/gpu1/operator-spec.json' if args.before else 'profiles/operator-spec.json')).read_text())
if not args.before: assert moduli(16384) == original['context']['rns_moduli']
profiles = base/('resnet-runtime-profiles'+suffix)
profiles.mkdir(exist_ok=False)
spec = original
spec['context']['poly_degree'] = 65536
if not args.before: spec['context']['rns_moduli'] = moduli(65536)
spec['provenance']['kind'] = 'test-fixture'
spec['provenance']['path'] = 'model-regression-20261021; N65536 planning and FP64 replay only'
(profiles/'operator-spec.json').write_text(json.dumps(spec, indent=2)+'\n')
(profiles/'compiler-profile.json').write_bytes((base/('compiler-baseline.json' if args.before else 'compiler-current.json')).read_bytes())
(profiles/'communication-profile.json').write_bytes((base/'profiles/communication-profile.json').read_bytes())
fixture = json.loads((base/'fixture.json').read_text())
(base/'resnet-runtime-fixture.json').write_text(json.dumps(dict(
    format_version=1,
    model=dict(name='ResNet20 real d13; compiler FP64 replay, not CKKS validation'),
    tolerances=dict(atol=0.001, rtol=0.0001),
    output_encoding=fixture['output_encoding'],
    inputs=[dict(name='image0', values=fixture['packed_input'])],
    expected_outputs=[dict(name='logits', values=fixture['python_output'])]), indent=2)+'\n')
for strategy in ['greedy', 'depth-dp']:
    template = json.loads((base/f'qwen-prefix{suffix}-{strategy}/gpu1/command.json').read_text())
    out = base/f'resnet20-runtime{suffix}-{strategy}'
    out.mkdir(exist_ok=False)
    command = []
    for i, arg in enumerate(template):
        if arg.startswith('--ckks-config='): arg = f'--ckks-config={profiles}/compiler-profile.json'
        if 'operator-spec-path=' in arg: arg = arg.split('=', 1)[0]+f'={profiles}/operator-spec.json'
        if 'communication-profile-path=' in arg: arg = arg.split('=', 1)[0]+f'={profiles}/communication-profile.json'
        if arg.startswith('--runtime-plan-operator-spec-sha256='):
            arg = '--runtime-plan-operator-spec-sha256=sha256:'+hashlib.sha256((profiles/'operator-spec.json').read_bytes()).hexdigest()
        if arg.endswith('/qwen-prefix/trace-qwen-prefix.mlirbc'): arg = str(base/'traced/ResNet20.mlirbc')
        if i and template[i-1] == '-o': arg = str(out/'model.mlir')
        command.append(arg)
    (out/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (out/'compile.log').open('w') as log:
        result = subprocess.run(['/usr/bin/time', '-v', *command], stdout=log, stderr=subprocess.STDOUT)
    (out/'status.json').write_text(json.dumps(dict(returncode=result.returncode))+'\n')
    print(strategy, result.returncode, flush=True)
