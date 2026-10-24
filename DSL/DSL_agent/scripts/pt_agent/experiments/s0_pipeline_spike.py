"""S0: inspect managed Earth IR with existing compiler passes, without library edits."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import subprocess
import sys
import threading
import time

from .. import toolchain
from ..paths import pt_result, work_result


PASSES = ('check-smu', 'convert-earth-to-ckks', 'convert-upscale-to-mulcp',
          'emit-runtime-plan', 'cse', 'canonicalize')
SMALL = {
    'p1-eva': ('pt-agent-r5d-p1-20261023T093414-bfcb', 'eva', None),
    # These two substitutions and the optimized rebuild were explicitly approved.
    'r2b-depth-dp': ('pt-agent-r2b-c3-boot-auto-dacapo-depth-dp-20261020T125516-3739',
                     'dacapo-depth-dp', 'qwen48'),
    'r2b-pars': ('pt-agent-r2b-c3-boot-agent-pars-20261020T125517-1ff0', 'pars', 'qwen48'),
}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def run(directory, label, argv, *, trusted=False, env=None):
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / (label + '.log')
    if log.exists():
        raise FileExistsError(log)
    compressed_input = Path(argv[1]) if len(argv) > 1 and str(argv[1]).endswith('.mlir.gz') else None
    if compressed_input:
        argv = list(argv)
        argv[1] = '-'
    write(directory / (label + '.argv.json'), argv)
    (directory / (label + '.command.txt')).write_text(shlex.join(argv) + '\n')
    if trusted:
        result = toolchain.trusted_run(directory, argv, log, env_extra=env)
    else:
        if shutil.disk_usage(directory).free < 10 * 1024**3:
            raise RuntimeError('S0 requires 10 GiB free disk space')

        def limits():
            resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
            resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**3, 8 * 1024**3))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        started = time.monotonic()
        with log.open('w') as stream:
            try:
                process = subprocess.Popen(argv, cwd=directory, stdout=stream, stderr=subprocess.STDOUT,
                                           stdin=subprocess.PIPE if compressed_input else None,
                                           env={**os.environ, **(env or {})}, preexec_fn=limits)
                writer_error = []

                def feed():
                    try:
                        with gzip.open(compressed_input, 'rb') as source:
                            shutil.copyfileobj(source, process.stdin, length=1024**2)
                    except BrokenPipeError:
                        pass  # An early parser rejection may close stdin.
                    except Exception as exc:
                        writer_error.append(str(exc))
                    finally:
                        try:
                            process.stdin.close()
                        except BrokenPipeError:
                            pass

                writer = threading.Thread(target=feed, daemon=True) if compressed_input else None
                if writer:
                    writer.start()
                code = process.wait(timeout=3600)
                if writer:
                    writer.join(timeout=5)
                result = dict(exit=code, timeout=False)
                if writer_error:
                    raise RuntimeError('Compressed input read failed: ' + '; '.join(writer_error))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                result = dict(exit=124, timeout=True)
        result['seconds'] = time.monotonic() - started
    result.update(argv=argv, log=str(log))
    if compressed_input:
        result['stdin_gzip'] = str(compressed_input)
        (directory / (label + '.command.txt')).write_text(
            'gzip -dc ' + shlex.quote(str(compressed_input)) + ' | ' + shlex.join(argv) + '\n')
    write(directory / (label + '.result.json'), result)
    return result


def options(case, directory):
    """Use the existing toolchain profile and RuntimePlan options verbatim."""
    if case['boot']:
        shutil.copyfile(toolchain.BOOT_CONFIG_DIR / 'compiler-config.json', directory / 'boot-compiler-config.json')
    argv = toolchain.to_host_argv(toolchain.compile_argv(case['mode'], case['boot'], directory), directory)
    config = next(arg for arg in argv if arg.startswith('--ckks-config='))
    plan_options = [arg.removeprefix('--runtime-plan-') for arg in argv if arg.startswith('--runtime-plan-')]
    # The direct pass uses plan-format, while the top-level option is runtime-plan-format.
    plan_options = [('plan-' + arg) if arg.startswith(('format=', 'id=')) else arg for arg in plan_options]
    plan_options += ['pretty=false', 'report-io=false', 'prefix=' + str(directory / 'lowered')]
    return config, ' '.join(plan_options)


def command(source, config, pipeline, output):
    return [str(toolchain.host_paths()['hecate_opt']), str(source), config,
            '--mlir-disable-threading', '--verify-each', '--pass-pipeline=' + pipeline, '-o', str(output)]


def pipelines(plan_options):
    before = 'cse,canonicalize,func.func(convert-earth-to-ckks,convert-upscale-to-mulcp,cse)'
    after = 'func.func(canonicalize,emit-runtime-plan{' + plan_options + '})'
    return 'builtin.module(' + before + ')', 'builtin.module(' + after + ')', \
        'builtin.module(' + before + ',' + after + ')'


def plan_path(directory):
    found = list(Path(directory).glob('lowered.*.runtime-plan.json'))
    if len(found) != 1:
        raise ValueError(f'Expected one RuntimePlan in {directory}, found {len(found)}')
    return found[0]


def stats(plan):
    steps = plan['initialization'] + plan['execution'] + plan['finalization']
    counts = dict(Counter(step.get('op') or step['kind'] for step in steps))
    return dict(steps=len(steps), ops=counts, boots=counts.get('boot', 0), values=len(plan['values']),
                inputs=len(plan['external_inputs']), outputs=len(plan['final_outputs']))


def step_states(plan):
    values = {v['id']: v for v in plan['values']}

    def state(name):
        value = values[name]
        return {key: value.get(key) for key in ('kind', 'level', 'scale_log2', 'components', 'ntt')}

    return [dict(op=step.get('op') or step['kind'],
                 inputs=[state(v) for v in step.get('inputs', [])],
                 output=state(step['output']) if 'output' in step else None,
                 attrs=step.get('attrs', {}))
            for step in plan['initialization'] + plan['execution'] + plan['finalization']]


def bundle(path):
    return path.with_name(path.name.removesuffix('.runtime-plan.json') + '.bundle')


def constants(plan, path):
    import numpy as np
    for step in plan['initialization']:
        payload = step.get('payload')
        if payload is None:
            continue
        if payload['kind'] == 'inline':
            yield np.asarray(payload['values'], dtype=np.float64)
        elif payload['kind'] == 'bundle':
            # Content identifiers locate blobs; no digest is computed or verified.
            name = payload['content'].split(':', 1)[1] + '.bin'
            yield np.fromfile(bundle(path) / 'data' / name, dtype='<f8')
        else:
            raise ValueError('Unknown constant payload kind: ' + payload['kind'])


def compare_plans(old_path, new_path):
    import numpy as np
    from itertools import zip_longest
    old, new = read(old_path), read(new_path)
    a, b = step_states(old), step_states(new)
    different = [i for i, (x, y) in enumerate(zip_longest(a, b)) if x != y]
    equal_constants = [x is not None and y is not None and bool(np.array_equal(x, y))
                       for x, y in zip_longest(constants(old, old_path), constants(new, new_path))]
    blob_counts = []
    for path in (old_path, new_path):
        manifest = bundle(path) / 'manifest.json'
        blob_counts.append(len(read(manifest)['blobs']) if manifest.exists() else 0)
    result = dict(original=stats(old), candidate=stats(new), step_states_equal=not different,
                  differing_step_indices=different, constant_comparisons=equal_constants,
                  constants_equal=all(equal_constants), bundle_constant_counts=blob_counts,
                  stats_equal=stats(old) == stats(new),
                  old_plan=str(old_path), new_plan=str(new_path))
    result['equal'] = (result['stats_equal'] and result['step_states_equal'] and result['constants_equal']
                       and blob_counts[0] == blob_counts[1])
    return result


def ir_operations(path, dialect):
    pattern = re.compile(r'"(' + re.escape(dialect) + r'\.[a-zA-Z0-9_]+)"')
    result = []
    with Path(path).open() as stream:
        for line in stream:
            match = pattern.search(line)
            if match:
                result.append(match.group(1))
    return result


def earth_management(path):
    """Compare operand links, management attributes and types without materializing dense constants."""
    result = []
    with Path(path).open() as stream:
        for line in stream:
            if '"earth.constant"' in line:
                result.append(line.split(' = ', 1)[0].strip() + ' = earth.constant : ' + line.rsplit(' : ', 1)[-1].strip())
            elif '"earth.' in line or 'func.func ' in line or line.lstrip().startswith('return '):
                result.append(line.strip())
    return result


def checker_diagnostics(path):
    examples = []
    count = 0
    with Path(path).open() as stream:
        for lineno, line in enumerate(stream, 1):
            if re.fullmatch(r'\s*-?\d+\s+-?\d+\s+-?\d+\s+-?\d+\s*', line):
                values = [int(word) for word in line.split()]
                if values[:2] != values[2:]:
                    count += 1
                    if len(examples) < 10:
                        examples.append(dict(line=lineno, recorded_level_scale=values[:2], current_level_scale=values[2:]))
    return dict(mismatch_diagnostics=count, examples=examples, log=str(path),
                interpretation='SMUChecker.cpp prints recorded level/scale and current level/scale when they differ')


def q1(case, directory):
    directory.mkdir(parents=True)
    config, plan_options = options(case, directory)
    first, second, _ = pipelines(plan_options)
    source = Path(case['directory']) / 'lowered.earth.mlir'
    ckks = directory / 'converted.ckks.mlir'
    result = dict(conversion=run(directory, 'convert', command(source, config, first, ckks)))
    if result['conversion']['exit'] == 0:
        old_ops = ir_operations(Path(case['directory']) / 'lowered.ckks.mlir', 'ckks')
        new_ops = ir_operations(ckks, 'ckks')
        result.update(ckks_ops_equal=old_ops == new_ops, old_ckks_ops=len(old_ops), new_ckks_ops=len(new_ops))
        result['export'] = run(directory, 'export', command(ckks, config, second, os.devnull))
        if result['export']['exit'] == 0:
            result['comparison'] = compare_plans(plan_path(Path(case['directory'])), plan_path(directory))
    result['equal'] = result.get('ckks_ops_equal', False) and result.get('comparison', {}).get('equal', False)
    write(directory / 'REPORT.json', result)
    return result


def variant(source, destination, kind):
    """Change one managed op, streaming large inline constants without copying the whole file in RAM."""
    changed = None
    replacement = None
    op_pattern = re.compile(r'^\s*(%[\w]+) = "earth\.(' +
                            ('rescale' if kind == 'delete_rescale' else 'bootstrap') + r')"\((%[\w]+)\)')
    with Path(source).open() as src, gzip.open(destination, 'xt', compresslevel=1) as dst:
        for lineno, line in enumerate(src, 1):
            if changed is None:
                if kind.startswith('delete_'):
                    match = op_pattern.search(line)
                    if match:
                        replacement = (re.compile(re.escape(match[1]) + r'(?![\w])'), match[3])
                        changed = dict(line=lineno, deleted_op=match[2], result=match[1], replacement=match[3])
                        continue
                elif kind == 'modswitch_factor' and '"earth.modswitch"' in line:
                    match = re.search(r'downFactor = (\d+) : i64', line)
                    if match:
                        line = line[:match.start(1)] + str(int(match[1]) + 1) + line[match.end(1):]
                        changed = dict(line=lineno, old_factor=int(match[1]), new_factor=int(match[1]) + 1)
                elif kind == 'result_level' and '"earth.' in line and '"earth.constant"' not in line:
                    position = line.find(' -> ')
                    match = re.search(r'!earth\.(?:ci|pl)<\d+ \* (\d+)>', line[position:])
                    if position >= 0 and match:
                        start, end = position + match.start(1), position + match.end(1)
                        line = line[:start] + str(int(match[1]) + 1) + line[end:]
                        changed = dict(line=lineno, old_level=int(match[1]), new_level=int(match[1]) + 1)
            if replacement:
                line = replacement[0].sub(replacement[1], line)
            dst.write(line)
    return changed


def q2(case, directory):
    directory.mkdir(parents=True)
    config, _ = options(case, directory)
    source = Path(case['directory']) / 'lowered.earth.mlir'
    result = {}
    for kind in ('original', 'delete_rescale', 'modswitch_factor', 'delete_bootstrap', 'result_level'):
        target = directory / kind
        target.mkdir()
        test_input = source
        change = None
        if kind != 'original':
            test_input = target / 'variant.earth.mlir.gz'
            change = variant(source, test_input, kind)
            if change is None:
                result[kind] = dict(applicable=False, reason='Input has no matching operation')
                write(directory / 'REPORT.json', result)
                continue
        cfg, plan_options = options(case, target)
        parse = run(target, 'verify', command(test_input, cfg, 'builtin.module()', os.devnull))
        checker = run(target, 'check-smu', command(test_input, cfg, 'builtin.module(func.func(check-smu))', os.devnull))
        direct = (run(target, 'direct', command(test_input, cfg, pipelines(plan_options)[2], os.devnull))
                  if kind != 'original' else None)
        exported = list(target.glob('lowered.*.runtime-plan.json'))
        result[kind] = dict(applicable=True, change=change, verifier=parse, check_smu=checker,
                            verifier_rejected=parse['exit'] != 0,
                            check_smu_rejected_after_verifier=parse['exit'] == 0 and checker['exit'] != 0,
                            direct=direct,
                            silently_exported=(direct['exit'] == 0 and len(exported) == 1) if direct else None)
        write(directory / 'REPORT.json', result)
    return result


def q3(case, directory):
    directory.mkdir(parents=True)
    source = Path(case['directory']) / 'lowered.earth.mlir'
    original_ops = ir_operations(source, 'earth')
    original_management = earth_management(source)
    result = {}
    for mode in ('eva', 'pars'):
        target = directory / mode
        target.mkdir()
        # EVA's existing toolchain entry accepts only its ordinary profile.
        boot = case['boot'] if mode == 'pars' else None
        if boot:
            shutil.copyfile(toolchain.BOOT_CONFIG_DIR / 'compiler-config.json', target / 'boot-compiler-config.json')
        argv = toolchain.to_host_argv(toolchain.compile_argv(mode, boot, target), target)
        argv[1] = str(source)
        # Keep the requested Earth/CKKS snapshots and RuntimePlan. The additional final IR is unused.
        (target / 'lowered.mlir').symlink_to(os.devnull)
        invocation = run(target, 'compile', argv)
        row = dict(run=invocation, requested_mode=mode, boot_profile=boot, original_boot_profile=case['boot'])
        earth = target / 'lowered.earth.mlir'
        if earth.exists():
            ops = ir_operations(earth, 'earth')
            row.update(earth_ops_equal=ops == original_ops, original_ops=dict(Counter(original_ops)),
                       candidate_ops=dict(Counter(ops)),
                       earth_management_equal=earth_management(earth) == original_management)
        if invocation['exit'] == 0 and list(target.glob('lowered.*.runtime-plan.json')):
            row['comparison'] = compare_plans(plan_path(Path(case['directory'])), plan_path(target))
        result[mode] = row
        write(directory / 'REPORT.json', result)
    return result


def prepare_model(item, target):
    target.mkdir(parents=True)
    package = Path(item['package'])
    trace_dir, compiled = target / 'trace', target / 'compiled'
    trace = run(target, 'trace', [sys.executable, '-B', str(package / 'trace_model.py'),
                '--weights', item['weights'], '--output-dir', str(trace_dir)], trusted=True,
                env=toolchain.package_trace_env())
    if trace['exit'] != 0:
        raise RuntimeError('P2 trace failed; see ' + trace['log'])
    compiled_run = run(target, 'compile', [sys.executable, '-B', str(package / 'compile_model.py'),
                '--hecate-opt', str(toolchain.host_paths()['hecate_opt']), '--input', str(trace_dir / 'trace_model.mlir'),
                '--output-dir', str(compiled)], trusted=True)
    if compiled_run['exit'] != 0:
        raise RuntimeError('P2 compile failed; see ' + compiled_run['log'])
    return dict(directory=str(compiled), mode='dacapo-depth-dp', boot='qwen48',
                preparation=dict(trace=trace, compile=compiled_run, package=str(package)))


def inside(args):
    toolchain.setup_env()
    manifest = read(args.manifest)
    out = args.out
    report = dict(round='S0', complete=False, phase=args.phase, cases={}, llm_calls=0, report_images=0)
    try:
        if args.phase == 'help':
            result = run(out, 'help', [str(toolchain.host_paths()['hecate_opt']), '--help'])
            help_text = Path(result['log']).read_text()
            report.update(help=result, passes={name: '--' + name in help_text for name in PASSES},
                          pass_pipeline='--pass-pipeline=' in help_text)
        elif args.phase == 'archive':
            report['archives'] = []
            report['checker_diagnostics'] = {}
            logs = sorted(args.manifest.parent.glob('q2-*/*/original/check-smu.log'))
            for log in logs:
                report['checker_diagnostics'][log.parent.parent.name] = checker_diagnostics(log)
            sources = sorted(args.manifest.parent.glob('q1-*/*/converted.ckks.mlir')) + logs
            for source in sources:
                if source.stat().st_size < 32 * 1024**2:
                    continue
                destination = source.with_suffix(source.suffix + '.gz')
                original_size = source.stat().st_size
                with source.open('rb') as src, gzip.open(destination, 'xb', compresslevel=1) as dst:
                    shutil.copyfileobj(src, dst, length=1024**2)
                source.unlink()
                report['archives'].append(dict(original=str(source), archive=str(destination),
                                                original_bytes=original_size, archive_bytes=destination.stat().st_size))
            report['scope'] = 'Only this run Q1 conversion copies and Q2 diagnostic logs; all original and preparation artifacts retained'
        elif args.phase == 'prepare':
            case = manifest['cases']['p1-eva']
            target = out / 'p1-optimized'
            target.mkdir()
            # The extra qwen48 attempt was explicitly approved after the ordinary-profile attempt failed.
            shutil.copyfile(toolchain.BOOT_CONFIG_DIR / 'compiler-config.json', target / 'boot-compiler-config.json')
            argv = toolchain.to_host_argv(toolchain.compile_argv('dacapo-optimized', 'qwen48', target), target)
            argv[1] = str(Path(case['directory']) / 'candidate_trace.mlir')
            invocation = run(target, 'compile', argv, trusted=True)
            report['cases']['p1-optimized'] = dict(directory=str(target), mode='dacapo-optimized', boot='qwen48',
                                                     preparation=invocation)
            if invocation['exit'] != 0:
                raise RuntimeError('P1 optimized preparation failed')
            for item in manifest['models']:
                report['cases'][item['model']] = prepare_model(item, out / item['model'])
                write(out / 'REPORT.json', report)
        else:
            for name in args.cases.split(','):
                case = manifest['cases'][name]
                result = globals()[args.phase](case, out / name)
                report['cases'][name] = result
                write(out / 'REPORT.json', report)
                print(args.phase + ' ' + name + ' recorded', flush=True)
                if args.phase == 'q1' and not result['equal']:
                    raise RuntimeError('Q1 comparison requires diagnosis: ' + name)
        report['complete'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    write(out / 'REPORT.json', report)
    return 0 if report['complete'] else 1


def inventory(root):
    cases = {name: dict(directory=str(work_result(directory)), mode=mode, boot=boot)
             for name, (directory, mode, boot) in SMALL.items()}
    p2 = read(pt_result('p2-export-20261023T075648', 'REPORT.json'))
    models = [{key: item[key] for key in ('model', 'package', 'weights', 'source')}
              for item in p2['programs'] if item['program'] in ('mlp', 'resnet20')]
    manifest = dict(cases=cases, models=models,
                    approved_input_extension='User approved R2b pars/depth-dp reuse and one P1 optimized compile',
                    p1_directories=[str(path) for path in sorted(work_result().glob('pt-agent-r5d-p1-20261023T0*'))])
    write(root / 'inputs.json', manifest)
    return root / 'inputs.json'


def summarize(root):
    groups = {phase: {} for phase in ('q1', 'q2', 'q3')}
    runs = []
    diagnostics = {}
    for path in sorted(root.glob('*/REPORT.json')):
        row = read(path)
        phase = row.get('phase')
        if phase == 'archive':
            diagnostics.update(row.get('checker_diagnostics', {}))
        runs.append(dict(phase=phase, report=str(path), complete=row.get('complete'),
                         exit=row.get('outer_exit'), seconds=row.get('seconds'), error=row.get('error')))
        if phase in groups:
            for name, case in row['cases'].items():
                if name in groups[phase]:
                    raise ValueError(f'Duplicate {phase} evidence for {name}')
                groups[phase][name] = case
    expected = set(SMALL) | {'p1-optimized', 'mlp', 'resnet20'}
    all_recorded = all(set(group) == expected for group in groups.values())
    q1_pass = set(groups['q1']) == expected and all(row['equal'] for row in groups['q1'].values())
    mandatory = {}
    for name, rows in groups['q2'].items():
        mandatory[name] = all(rows.get(kind, {}).get('applicable') and
                              (rows[kind]['verifier_rejected'] or rows[kind]['check_smu_rejected_after_verifier'])
                              for kind in ('delete_rescale', 'result_level'))
    originals = {name: row['original']['verifier']['exit'] == row['original']['check_smu']['exit'] == 0
                 for name, row in groups['q2'].items()}
    q2_pass = set(mandatory) == expected and all(mandatory.values()) and all(originals.values())
    result = dict(round='S0', complete=all_recorded, q1_all_equal=q1_pass,
                  q2_mandatory_detected=mandatory, q2_originals_pass=originals,
                  q2_checker_diagnostics=diagnostics,
                  decision=('pass' if q2_pass else 'partial') if all_recorded and q1_pass else 'pending',
                  llm_calls=0, report_images=0, compiler_changes=0, existing_module_changes=0,
                  cases={name: {phase: groups[phase].get(name) for phase in groups} for name in sorted(expected)},
                  runs=runs, note='Q3 EVA uses the toolchain ordinary profile; qwen48 profile differences are explicit.')
    write(root / 'SUMMARY.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--phase', choices=('help', 'prepare', 'q1', 'q2', 'q3', 'archive', 'summary'), default='help')
    parser.add_argument('--cases', default='p1-eva,r2b-depth-dp,r2b-pars')
    args = parser.parse_args()
    if args.inside:
        if args.out is None or args.manifest is None:
            parser.error('--inside requires explicit --out and --manifest')
        return inside(args)
    from ..registry import append, git_rev, new_run_id
    root = args.out or pt_result('s0-pipeline-spike-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == 'summary':
        report = summarize(root)
        os.environ['PT_RESULTS'] = str(pt_result())
        append(dict(run_id=new_run_id('S0', 'six-inputs', 'summary'), round='S0', model='six-inputs',
                    mode='summary', git_rev=git_rev(), metrics=report, artifacts_dir=str(root),
                    ts=datetime.now(timezone.utc).isoformat()))
        print('SUMMARY=' + str(root / 'SUMMARY.json'), flush=True)
        return 0 if report['complete'] else 1
    manifest = args.manifest or (root / 'inputs.json' if (root / 'inputs.json').exists() else inventory(root))
    destination = root / (args.phase + '-' + args.cases.replace(',', '_'))
    destination.mkdir()
    os.environ['PT_RESULTS'] = str(pt_result())
    started = time.monotonic()
    argv = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '14400', '--',
            '-m', 'pt_agent.experiments.s0_pipeline_spike', '--inside', '--out', str(destination),
            '--manifest', str(manifest), '--phase', args.phase, '--cases', args.cases]
    write(destination / 'outer.argv.json', argv)
    with (destination / 'stdout.txt').open('w') as stdout, (destination / 'stderr.txt').open('w') as stderr:
        result = subprocess.run(argv, stdout=stdout, stderr=stderr)
    report = read(destination / 'REPORT.json') if (destination / 'REPORT.json').exists() else dict(complete=False)
    report.update(outer_exit=result.returncode, seconds=time.monotonic() - started)
    write(destination / 'REPORT.json', report)
    if args.phase == 'prepare':
        data = read(manifest)
        data['cases'].update(report.get('cases', {}))
        write(root / 'prepared-inputs.json', data)
    append(dict(run_id=new_run_id('S0', args.cases, args.phase), round='S0', model=args.cases,
                mode=args.phase, git_rev=git_rev(), metrics=report, artifacts_dir=str(destination),
                ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT=' + str(destination / 'REPORT.json'), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
