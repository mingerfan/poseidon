#!/usr/bin/env python3
"""Compile the historical 120-bit/5-Q policy with the scale45 contract shim."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

base = Path('/home/xuming/poseidon-model-regression-20261021')
for strategy in ['greedy', 'depth-dp']:
    out = base/f'qwen-prefix-before-{strategy}'/'gpu1'
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((base/'compiler-baseline.json').read_text())
    config['polynomialDegree'] = 16384
    (out/'compiler-profile.json').write_text(json.dumps(config, indent=2)+'\n')
    spec = json.loads((base/'profiles/operator-spec.json').read_text())
    spec['rescale_mode'] = 'lazy'
    spec['context'].pop('rns_moduli', None)
    spec['levels']['upper_bound'] = 22
    for operator in spec['operators'].values():
        for key in ['latency_us_by_level', 'noise_by_level']:
            if isinstance(operator.get(key), list): operator[key] = operator[key][:23]
    spec['operators']['rescale']['max_levels_per_op'] = 5
    for boot in spec['boot_profiles']:
        boot['input_level_max'] = 22
        for key in ['latency_us_by_input_level', 'noise_by_input_level']:
            if isinstance(boot.get(key), list): boot[key] = boot[key][:23]
    spec_path = out/'operator-spec.json'
    spec_path.write_text(json.dumps(spec, indent=2)+'\n')
    command = json.loads((base/f'qwen-prefix-{strategy}/gpu1/command.json').read_text())
    command = [str(base/'run-baseline.sh'), *command[4:]]
    changes = {
        '--ckks-config=': str(out/'compiler-profile.json'),
        '--runtime-plan-physical-level-operator-spec-path=': str(spec_path),
        '--runtime-plan-operator-spec-path=': str(spec_path),
        '--runtime-plan-levels-per-logical-level=': '5',
        '--runtime-plan-operator-spec-sha256=': 'sha256:'+hashlib.sha256(spec_path.read_bytes()).hexdigest()}
    for i, arg in enumerate(command):
        for key, value in changes.items():
            if arg.startswith(key): command[i] = key+value
        if arg == '-o': command[i+1] = str(out/'qwen3.mlir')
    (out/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    start = time.monotonic()
    with (out/'compile.log').open('w') as log:
        result = subprocess.run(['/usr/bin/time', '-v', *command], stdout=log,
                                stderr=subprocess.STDOUT, timeout=240)
    status = dict(returncode=result.returncode, elapsed_seconds=time.monotonic()-start)
    (out/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(strategy, status, flush=True)
