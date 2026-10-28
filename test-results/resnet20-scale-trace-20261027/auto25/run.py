"""Run full ResNet20 with automatic target scales, without manual constraints."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path('/home/xuming/poseidon-manual-boot-20261026')
EXPERIMENT = ROOT / 'resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-q', type=int, choices=[31, 34], required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root / f'q{args.initial_q}'
    out.mkdir(parents=True, exist_ok=False)
    command = json.loads((EXPERIMENT / 'budget25/automatic/manual34/result.json').read_text())['command']
    command = [arg for arg in command if not arg.startswith((
        '--native-target-bootstrap-schedule=', '--native-target-scale-goals=', '--native-input-level='))]
    command += [f'--native-input-level={args.initial_q - 1}']
    command[command.index('-o') + 1] = str(out / 'resnet20.optimized.mlirbc')
    assert '--native-target-scale-planning' in command
    assert '--boot-placement=greedy' in command
    assert not any('bootstrap-schedule' in arg or 'scale-goals' in arg for arg in command)
    report = dict(initial_q=args.initial_q, scale_strategy='target', boot_placement='greedy',
                  manual_scale_goals=False, fixed_boot_schedule=False,
                  bootstrap_q_budget=25, command=command,
                  compiler_sha256=(EXPERIMENT / 'binary.sha256').read_text().strip())
    start = time.monotonic()
    with (out / 'compile.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        report['pid'] = child.pid
        (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        try:
            code = child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            raise
    report.update(returncode=code, compile_seconds=time.monotonic() - start,
                  peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    if code:
        (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        raise RuntimeError((out / 'compile.log').read_text()[-2500:])
    path = next(out.glob('*.runtime-plan.json'))
    plan = json.loads(path.read_text())
    spec_path = ROOT / 'alignment-tests/operator-spec.json'
    spec = json.loads(spec_path.read_text())
    moduli = spec['context']['rns_moduli']
    desc = {v['id']: v for v in plan['values']}
    assert desc[plan['external_inputs'][0]]['level'] == args.initial_q - 1
    boots = []
    largest_rescale_error = 0.
    for step in plan['execution']:
        if step.get('op') == 'boot':
            before, after = desc[step['inputs'][0]], desc[step['output']]
            assert before['level'] == 5 and abs(before['scale_log2'] - 40) < 1e-8
            assert after['level'] == 30 and after['scale_log2'] == 40
            boots.append(dict(input=before, output=after))
        elif step.get('op') == 'rescale':
            before, after = desc[step['inputs'][0]], desc[step['output']]
            assert before['level'] - after['level'] == 1
            expected = before['scale_log2'] - math.log2(moduli[before['level']])
            error = abs(expected - after['scale_log2'])
            largest_rescale_error = max(largest_rescale_error, error)
            assert error < 1e-8 and after['scale_log2'] >= 40 - 1e-8
    validation = subprocess.run([str(ROOT / 'validate-plan'), str(path), str(spec_path)],
                                capture_output=True, text=True, timeout=120)
    report.update(boots=len(boots), counts=dict(Counter(s.get('op', s['kind']) for s in plan['execution'])),
                  input=desc[plan['external_inputs'][0]], output=desc[plan['final_outputs'][0]],
                  plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  operator_spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                  max_q_bits=max(q.bit_length() for q in moduli),
                  largest_rescale_formula_error=largest_rescale_error,
                  validation=validation.stdout + validation.stderr, validation_code=validation.returncode)
    (out / 'boots.json').write_text(json.dumps(boots, indent=2) + '\n')
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    assert validation.returncode == 0, report['validation']
    print(json.dumps({k: v for k, v in report.items() if k != 'command'}), flush=True)


if __name__ == '__main__':
    main()
