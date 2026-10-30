"""Compile with Dacapo automatic native bootstrap placement; retain all values.

Defaults reproduce the CPU compiler contract, not a GPU execution environment.
A different backend must supply both matching contracts and its device count.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hecate-opt', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--compiler-config', type=Path)
    parser.add_argument('--operator-spec', type=Path)
    parser.add_argument('--boot-profile')
    parser.add_argument('--device-count', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    custom = args.compiler_config is not None or args.operator_spec is not None
    if custom and (args.compiler_config is None or args.operator_spec is None or args.device_count is None):
        parser.error('Custom backend requires --compiler-config, --operator-spec and --device-count together')
    config_path = (args.compiler_config or ROOT / 'config/compiler-config.json').resolve()
    spec_path = (args.operator_spec or ROOT / 'config/operator-spec.json').resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    profiles = [p for p in spec.get('boot_profiles', [])
                if args.boot_profile is None or p['profile_id'] == args.boot_profile]
    if len(profiles) != 1:
        parser.error('Select exactly one native profile with --boot-profile')
    boot = profiles[0]
    if not spec['operators']['boot']['supported'] or boot['implementation'] != 'native':
        parser.error('This package requires supported native bootstrap')
    host = boot.get('host_requirements', {})
    if host.get('needs_secret_key', False):
        parser.error('Bootstrap must not require a secret key')
    lower, upper = config['bootstrapLevelLowerBound'], config['bootstrapLevelUpperBound']
    if not boot['input_level_min'] <= lower <= boot['input_level_max'] or upper != boot['output_level']:
        parser.error('Compiler bootstrap levels do not match the runtime profile')
    scale = config['rescalingFactor']
    if scale != boot['output_scale_log2'] or scale != spec['context']['default_scale_log2']:
        parser.error('Compiler and backend scales do not match')
    if config['polynomialDegree'] != 65536 or spec['context']['poly_degree'] != 65536:
        parser.error('The frozen DSL requires 32768 slots (poly_degree=65536)')
    count = args.device_count if args.device_count is not None else 0
    if count < 0:
        parser.error('Device count must be nonnegative')
    source, dest = args.input.resolve(), args.output_dir.resolve()
    if not source.is_file() or not args.hecate_opt.is_file():
        parser.error('Input MLIR/bytecode and hecate-opt must exist')
    if dest.exists():
        parser.error('Use a new output directory; existing results are preserved')
    command = [str(args.hecate_opt.resolve()), str(source), '--dacapo', '--boot-placement=depth-dp',
        '--waterline=' + str(scale), '--ckks-config=' + str(config_path),
        '--runtime-plan-format=json', '--runtime-plan-bundle-format=files', '--runtime-plan-memory=false',
        '--runtime-plan-id=2026101202', '--runtime-plan-target-id=' + spec['target_id'],
        '--runtime-plan-operator-spec-id=' + spec['spec_id'],
        '--runtime-plan-operator-spec-sha256=sha256:' + hashlib.sha256(spec_bytes).hexdigest(),
        '--runtime-plan-context-id=' + spec['context']['context_id'],
        '--runtime-plan-device-count=' + str(count), '--runtime-plan-boot-profile=' + boot['profile_id'],
        '--runtime-plan-boot-implementation=native', '--mlir-timing', '--mlir-disable-threading',
        '-o', str(dest / 'compiled.mlir')]
    print(json.dumps(command, indent=2), flush=True)
    if args.dry_run:
        return
    dest.mkdir(parents=True)
    (dest / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    with (dest / 'compile.log').open('xb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    plans = list(dest.glob('*.runtime-plan.json'))
    if len(plans) != 1:
        raise RuntimeError('Expected exactly one RuntimePlan')
    plan = json.loads(plans[0].read_text(encoding='utf-8'))
    operations = plan['initialization'] + plan['execution'] + plan['finalization']
    boots = [op for op in operations if op.get('op') == 'boot']
    if not boots or any(op['attrs']['implementation'] != 'native' for op in boots):
        raise RuntimeError('The exported plan must contain native Boot operations')
    if any(op['kind'] == 'release' or 'reuse_input' in op for op in operations):
        raise RuntimeError('Expected retention of all intermediate values')
    print(json.dumps(dict(plan=str(plans[0]), planned_boots=len(boots), ckks_executed=False)))


if __name__ == '__main__':
    main()
