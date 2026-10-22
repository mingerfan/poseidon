"""Native-Boot compilation (matching compiler required); binary IR, explicit Release, no input reuse."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hecate-opt', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--compiler-config', type=Path)
    p.add_argument('--operator-spec', type=Path)
    p.add_argument('--device-count', type=int)
    p.add_argument('--boot-profile')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    cmd = [sys.executable, '-B', str(ROOT/'frozen/model/package/compile_qwen24.py'),
        '--hecate-opt', str(a.hecate_opt.resolve()), '--input', str(a.input.resolve()),
        '--output-dir', str(a.output_dir.resolve()), '--retention', 'final', '--dry-run']
    for field in ('compiler_config', 'operator_spec', 'device_count', 'boot_profile'):
        value = getattr(a, field)
        if value is not None:
            cmd += ['--'+field.replace('_','-'), str(value.resolve() if isinstance(value,Path) else value)]
    command = json.loads(subprocess.check_output(cmd, text=True))
    assert command[-2] == '-o'
    command[-1] = str(a.output_dir.resolve()/'compiled.mlirbc')
    command.insert(-2, '--emit-bytecode')
    print(json.dumps(command, indent=2), flush=True)
    if a.dry_run:
        return
    a.output_dir.mkdir(parents=True)
    (a.output_dir/'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    with (a.output_dir/'compile.log').open('xb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    plans = list(a.output_dir.glob('*.runtime-plan.json'))
    if len(plans) != 1:
        raise RuntimeError('Expected one exported RuntimePlan')
    plan = json.loads(plans[0].read_text('utf-8'))
    ops = sum((plan[s] for s in ('initialization','execution','finalization')), [])
    boots = [o for o in ops if o.get('op') == 'boot']
    assert boots and all(o['attrs']['implementation'] == 'native' for o in boots)
    assert any(o['kind'] == 'release' for o in ops)
    assert all('reuse_input' not in o for o in ops)
    print(json.dumps(dict(plan=str(plans[0]), boots=len(boots), ckks_executed=False)))

if __name__ == '__main__':
    main()
