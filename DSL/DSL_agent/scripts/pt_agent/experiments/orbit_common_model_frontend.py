"""Bounded original Orbit CNN trace and TDAG preparation, without optimization.

The official model functions and checkpoints are selected explicitly. This outer
process owns results/registry; each inner worker receives all paths as argv.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import threading
import time

from .. import paths
from . import orbit_common_model_worker as worker
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

ROUND = 'ORBIT-COMMON-FRONTEND'
MODE = 'original-orbit-common-model-trace-tdag'
MODULE = 'pt_agent.experiments.orbit_common_model_worker'
GIB = 1024 ** 3
MODEL_RECORDS = {
    'resnet20': dict(factory='pt_agent.experiments.r13_common_models:resnet20_silu',
                    checkpoint='resnet20.silu.model', r13='r13-common-resnet20-silu-20261115',
                    r16='orbit-resnet20_silu'),
    'vgg16': dict(factory='pt_agent.experiments.r13_common_models:vgg16_silu_avgpool',
                 checkpoint='vgg16_silu_avgpool_model', r13='r13-common-vgg16-silu-avgpool-20261115',
                 r16='orbit-vgg16_silu_avgpool'),
    'alexnet': dict(factory='pt_agent.experiments.r13_common_models:alexnet_silu_avgpool',
                   checkpoint='alexNet_silu_avgpool_model', r13='r13-common-alexnet-silu-avgpool-20261115',
                   r16='orbit-alexnet_silu_avgpool'),
}


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Expected an object source record')
    return value


def selected_paths(args):
    selection = MODEL_RECORDS[args.model]
    factory = selection['factory']
    record = worker.MODEL_RECORDS[factory]
    root = Path(args.orbit_root or paths.ORBIT_ROOT).resolve()
    old = paths.RESULTS_DIR / 'orbit-resnet20-trace-tdag-r22-20261115/work'
    return dict(model=factory, scripts=paths.REPO / 'DSL/DSL_agent/scripts', repo=paths.REPO,
                orbit_root=root, frontend=root / 'frontend/dacapo',
                benchmark=root / 'frontend/dacapo_patch/benchmarks' / (record['benchmark'] + '.py'),
                model_source=root / 'frontend/dacapo/python/poly/poly/models' / record['source'],
                checkpoint=Path(args.checkpoint or paths.DACAPO_UPSTREAM / 'examples/data' / selection['checkpoint']).resolve(),
                r13_result=Path(args.r13_result or paths.RESULTS_DIR / selection['r13'] / 'result.json').resolve(),
                r16_result=Path(args.r16_result or paths.RESULTS_DIR / 'r16-baseline-compatibility-20261115' / selection['r16'] / 'result.json').resolve(),
                poly_dir=Path(args.poly_dir or old / 'local-poly/poly').resolve(),
                deps_dir=Path(args.deps_dir or old / 'python-deps').resolve(),
                intake_python=Path(args.intake_python or paths.INTAKE_PYTHON).absolute(),
                orbit_python=Path(args.orbit_python or root.parent / 'orbit-venv/bin/python').absolute(),
                profile=root / worker.PROFILE, activations=record['activations'])


def source_preflight(locations):
    files = ('benchmark', 'model_source', 'checkpoint', 'r13_result', 'r16_result', 'profile')
    for name in files:
        value = locations[name]
        if not value.is_file() or value.stat().st_size <= 0:
            raise FileNotFoundError('Missing original source: ' + name)
    # Intake venv symlinks can resolve only inside the existing pure Nix shell.
    for name in ('intake_python', 'orbit_python'):
        value = locations[name]
        if not (value.is_file() or value.is_symlink()):
            raise FileNotFoundError('Missing existing interpreter: ' + name)
    text_files = [locations['poly_dir'] / name for name in ('MPCB.py', 'Poly.py', 'Func.py')]
    text_files += [locations['poly_dir'] / 'data' / name for name in worker.DATA_FILES]
    if any(not p.is_file() or p.stat().st_size <= 0 for p in text_files):
        raise FileNotFoundError('Missing existing isolated original polynomial source')
    if not (locations['deps_dir'] / 'einops/__init__.py').is_file():
        raise FileNotFoundError('The existing isolated einops dependency is missing; installation is outside this driver')
    degree = _read(locations['profile']).get('polynomialDegree')
    if degree != 131072:
        raise ValueError('The original profile must retain polynomialDegree=131072')
    source = worker._check_sources(argparse.Namespace(**locations))
    return source | dict(model=locations['model'], benchmark=str(locations['benchmark']),
                frontend=str(locations['frontend']), original_profile=str(locations['profile']),
                polynomial_degree=131072, logical_slots=65536,
                original_poly_files=[dict(path=str(p), size_bytes=p.stat().st_size) for p in text_files],
                original_model_modified=False, weights_modified=False,
                polynomial_modified=False, prior_times_included=False,
                coefficient_equivalence_status='Original local Poly sources reused; complete model numerical equivalence remains pending',
                launcher_check_scope='Only the previously approved existing launcher object checks; no additional integrity checks',
                native_parameter_or_security_validation=False)


def child_environment(work, locations, env=None):
    inherited = dict(os.environ if env is None else env)
    removed = sorted(k for k in inherited if k.startswith(('LD_', 'NIX_', '__NIX_', 'HECATE_'))
                     or k in ('HECATE', 'IN_NIX_SHELL', 'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV'))
    for key in removed:
        inherited.pop(key, None)
    inherited.update(PYTHONPATH=str(locations['scripts']), PYTHONDONTWRITEBYTECODE='1',
                     PYTHONNOUSERSITE='1', TMPDIR=str(work / 'tmp'), MPLBACKEND='Agg')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        inherited[key] = '1'
    return inherited, dict(removed_environment_keys=removed,
                           controlled_pythonpath=[str(locations['scripts'])], home_preserved=True)


def phase_command(locations, work, phase, args, mlir=None):
    if phase == 'trace':
        argv = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds',
                str(math.ceil(args.trace_seconds + 60)), '--python', str(locations['intake_python']), '--', '-m', MODULE]
        options = ('frontend', 'model_source', 'benchmark', 'checkpoint', 'poly_dir', 'deps_dir',
                   'orbit_root', 'r13_result', 'r16_result')
        memory, seconds = args.memory_gib, args.trace_seconds
    else:
        if mlir is None:
            raise ValueError('Parse requires the actual trace-returned MLIR path')
        argv = [str(locations['orbit_python']), '-B', '-m', MODULE]
        options = ('orbit_root', 'profile')
        memory, seconds = 4, args.parse_seconds
    argv += ['--phase', phase, '--model', locations['model'], '--work-dir', str(work / (phase + '-worker')),
             '--memory-gib', str(memory), '--case-gib', str(args.case_gib), '--phase-seconds', str(seconds)]
    for key in options:
        argv += ['--' + key.replace('_', '-'), str(locations[key])]
    if phase == 'parse':
        argv += ['--mlir', str(mlir)]
    return argv


def validate_phase(record, phase, locations, work):
    if (record.get('round') != ROUND or record.get('model') != locations['model']
            or record.get('phase') != phase or record.get('status') != 'ok'
            or not isinstance(record.get('elapsed_ns'), int)):
        raise ValueError('Inner phase identity or terminal completion differs')
    if any(record.get(key) != 0 for key in worker.ZERO_FIELDS):
        raise ValueError('Inner phase performed an excluded action')
    if any(record.get(key) is not None for key in worker.UNMEASURED_FIELDS):
        raise ValueError('Unmeasured formal results must remain null')
    if any(record.get(key) is not False for key in
           ('optimizer_ready', 'backend_ready', 'minimal_ckks_passed', 'formal_matrix_complete')):
        raise ValueError('Inner preparation claimed an unmeasured capability')
    if phase == 'trace':
        exact = ('checkpoint_load_attempts', 'checkpoint_load_completed', 'model_build_attempts',
                 'model_build_completed', 'expr_load_attempts', 'expr_load_completed',
                 'function_registration_attempts', 'function_registrations', 'save_attempts',
                 'save_completed', 'finalize_attempts', 'finalize_completed', 'get_model_calls', 'trace_runs')
        if (any(record.get(k) != 1 for k in exact)
                or record.get('source_checked') is not True
                or record.get('environment_restored') is not True
                or record.get('orbit_model_trace_passed') is not True
                or record.get('activation_calls') != locations['activations']
                or len(record.get('activation_names', [])) != locations['activations']
                or record.get('parse_runs') != 0):
            raise ValueError('The complete original model trace or finalize count differs')
        artifact = record.get('mlir') or {}
        path = Path(artifact.get('path', '')).resolve()
        if (not path.is_relative_to((work / 'trace-worker/trace').resolve())
                or not path.is_file() or path.stat().st_size <= 0
                or path.stat().st_size != artifact.get('size_bytes')):
            raise ValueError('Actual trace output must be a nonempty trace-local MLIR')
        return path
    if (any(record.get(k) != 1 for k in ('params_calls', 'params_completed', 'build_from_mlir_calls',
                                        'build_from_mlir_completed', 'parse_runs'))
            or record.get('trace_runs') != 0 or record.get('tdag_parse_passed') is not True):
        raise ValueError('Original TDAG parser count differs')
    check, metadata = record.get('graph_check') or {}, record.get('parser_metadata') or {}
    if (any(check.get(k) is not True for k in ('passed', 'acyclic', 'input_reaches_output', 'compile_level_scale_unassigned'))
            or metadata.get('poly_deg') != 131072 or metadata.get('max_slot') != 65536
            or Path(metadata.get('profile', '')).resolve() != locations['profile'].resolve()):
        raise ValueError('Original TDAG acceptance or metadata-only profile alias differs')
    path = Path(record.get('tdag', '')).resolve()
    if path != (work / 'parse-worker/tdag.json').resolve() or not path.is_file():
        raise ValueError('TDAG must be the actual parse-worker output')
    # JSON shape and graph semantics are independently rechecked, without parsing MLIR again.
    actual = worker._check_graph(_read(path))
    for key in ('node_count', 'edge_count', 'op_counts'):
        if actual[key] != check.get(key):
            raise ValueError('Saved TDAG metadata differs: ' + key)
    return path


def _snapshot(locations, out):
    target = out / 'source'
    target.mkdir()
    sources = [Path(__file__), Path(worker.__file__), locations['benchmark'], locations['model_source']]
    sources += [locations['poly_dir'] / name for name in ('MPCB.py', 'Poly.py', 'Func.py')]
    sources += [locations['poly_dir'] / 'data' / name for name in worker.DATA_FILES]
    snapshots = []
    for index, source in enumerate(sources):
        destination = target / (str(index).zfill(2) + '-' + source.name)
        shutil.copyfile(source, destination)
        snapshots.append(dict(source=str(source), snapshot=str(destination), size_bytes=destination.stat().st_size))
    return snapshots


def _clean(observation):
    return (observation.get('exit_code') == 0 and not observation.get('timed_out')
            and observation.get('signal') is None and not observation.get('signals_sent')
            and not observation.get('remaining_pids'))


def _terminal(out, result):
    for _ in range(20):
        result['metrics']['status'] = result['status']
        result['metrics']['case_artifact_bytes'] = result.get('case_artifact_bytes', 0)
        write_json(out / 'result.json', result)
        amount = artifact_bytes(out)
        if amount == result.get('case_artifact_bytes'):
            return
        result['case_artifact_bytes'] = amount
    raise RuntimeError('Terminal result artifact size did not stabilize')


def completion_reserve_bytes(timing):
    """Bound the known completion payload before persisting a success registry row."""
    rows = [{key: row.get(key) for key in ('event_id', 'stage_id', 'parent_id',
             'inclusive_ns', 'exclusive_ns', 'status')} for row in timing['spans']]
    # Rows dominate the variable payload; fixed fields and integer endpoints
    # use less than 8 KiB for these fixed round/status/clock names.
    return len(json.dumps(rows, ensure_ascii=False, indent=2).encode('utf-8')) + 8192


def observed_action_counts(result):
    """Missing or interrupted worker evidence leaves excluded action totals unknown."""
    totals = {}
    for key in worker.ZERO_FIELDS:
        amount = 0
        for phase in ('trace', 'parse'):
            if not result['attempted'][phase]:
                continue
            record = result['phase_results'].get(phase) or {}
            if (not isinstance(record.get('elapsed_ns'), int)
                    or record.get('status') in (None, 'running')
                    or type(record.get(key)) is not int or record[key] < 0):
                amount = None
                break
            amount += record[key]
        totals[key] = amount
    return totals


def append_registry(registry, result, out):
    registry.parent.mkdir(parents=True, exist_ok=True)
    row = dict(run_id=out.name, round=ROUND, model=result['model'], mode=MODE,
               git_rev=result.get('git_rev'), metrics=result['metrics'], artifacts_dir=str(out),
               ts=datetime.now(timezone.utc).isoformat())
    with registry.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise ValueError('Preserving an existing registry run_id')
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def execute(args):
    started = time.perf_counter_ns()
    locations = selected_paths(args)
    out = Path(args.out).absolute()
    if out.exists() or out.is_symlink():
        raise ValueError('A fresh output directory is required')
    out.mkdir(parents=True)
    work = out / 'work'
    (work / 'tmp').mkdir(parents=True)
    recorder = StageRecorder(out / 'events.jsonl')
    deadline = time.monotonic() + args.overall_seconds
    result = dict(round=ROUND, mode=MODE, model=locations['model'], model_key=args.model,
                  status='running', error=None, failed_stage=None, git_rev=None,
                  attempted=dict(trace=0, parse=0), completed=dict(trace=0, parse=0),
                  phase_results={}, processes=[], worker_record_errors=[], source_records=None,
                  source_snapshots=[], orbit_model_trace_passed=False, tdag_parse_passed=False,
                  optimizer_ready=False, backend_ready=False, formal_matrix_complete=False,
                  extra_integrity_checks=0, retry_count=0, case_artifact_bytes=0,
                  resource_monitor_error=None, completion=None)
    result['case_artifact_bytes_scope'] = 'Required artifacts through result persistence; completion.json records the final self-count including its own marker'
    result.update(dict.fromkeys(worker.ZERO_FIELDS, 0))
    result.update(dict.fromkeys(worker.UNMEASURED_FIELDS, None))
    env, result['environment'] = child_environment(work, locations)
    result['limits'] = dict(trace_seconds=args.trace_seconds, trace_nix_seconds=math.ceil(args.trace_seconds + 60),
                            trace_supervisor_seconds=args.trace_seconds + 240, trace_memory_gib=args.memory_gib,
                            parse_seconds=args.parse_seconds, parse_supervisor_seconds=args.parse_seconds + 60,
                            parse_memory_gib=4, overall_seconds=args.overall_seconds, case_gib=args.case_gib,
                            file_gib=8, free_reserve_gib=10, single_concurrency=True,
                            memory_limit_semantics='Per-process RLIMIT_AS; sampled RSS is reported separately')
    stop = threading.Event()
    child_events = []
    saved_handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def interrupted(signum, frame):
        raise RuntimeError('Outer signal ' + str(signum))

    def checkpoint():
        if result['resource_monitor_error']:
            raise RuntimeError(result['resource_monitor_error'])
        if artifact_bytes(out) > args.case_gib * GIB:
            raise RuntimeError('Case artifacts exceed explicit allowance')
        if shutil.disk_usage(out).free < 10 * GIB:
            raise RuntimeError('Remaining disk is below the 10 GiB reserve')
        if time.monotonic() >= deadline:
            raise TimeoutError('Outer overall allowance exhausted')

    def monitor():
        while not stop.wait(.5):
            try:
                amount, free = artifact_bytes(out), shutil.disk_usage(out).free
                with (out / 'artifact-resources.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(ts_ns=time.perf_counter_ns(), artifact_bytes=amount, disk_free_bytes=free)) + '\n')
                if amount > args.case_gib * GIB or free < 10 * GIB:
                    raise RuntimeError('Periodic artifact or disk reserve allowance exceeded')
            except BaseException as exc:
                result['resource_monitor_error'] = f'{type(exc).__name__}: {exc}'
                os.kill(os.getpid(), signal.SIGTERM)
                return

    watcher = threading.Thread(target=monitor, daemon=True)

    def launch(phase, mlir=None):
        checkpoint()
        result['failed_stage'] = phase
        argv = phase_command(locations, work, phase, args, mlir)
        result['attempted'][phase] += 1
        process = dict(phase=phase, argv=argv, cwd=str(locations['scripts']), observation=None,
                       native_termination_signal=None)
        result['processes'].append(process)
        write_json(out / 'progress.json', result)
        with recorder.span('orbit.' + phase, argv=argv) as parent:
            try:
                observation = run_process(argv, cwd=locations['scripts'], log=out / (phase + '.log'),
                                          resources=out / (phase + '-resources.jsonl'),
                                          timeout=min(args.trace_seconds + 240 if phase == 'trace' else args.parse_seconds + 60,
                                                      max(.01, deadline - time.monotonic())),
                                          mem_gib=args.memory_gib if phase == 'trace' else 4, fsize_gib=8, env=env)
                process['observation'] = observation
                code = observation.get('exit_code')
                process['native_termination_signal'] = -code if isinstance(code, int) and code < 0 else None
            finally:
                phase_dir = work / (phase + '-worker')
                try:
                    worker_events = read_events(phase_dir / 'events.jsonl')
                    if not worker_events:
                        raise ValueError('Inner phase event evidence is missing')
                    for event in worker_events:
                        if event.get('type') in ('begin', 'end') and event.get('parent_id') is None:
                            event['parent_id'] = parent['event_id']
                        child_events.append(event)
                    record_path = phase_dir / 'phase_result.json'
                    if record_path.is_file():
                        result['phase_results'][phase] = _read(record_path)
                except BaseException as exc:
                    result['worker_record_errors'].append(dict(phase=phase, error=f'{type(exc).__name__}: {exc}'))
            if not _clean(observation):
                raise RuntimeError('Supervised ' + phase + ' failed')
            if any(row['phase'] == phase for row in result['worker_record_errors']):
                raise ValueError('The inner phase result is invalid')
            artifact = validate_phase(result['phase_results'].get(phase, {}), phase, locations, work)
            result['completed'][phase] += 1
        checkpoint()
        return artifact

    try:
        for signum in saved_handlers:
            signal.signal(signum, interrupted)
        signal.setitimer(signal.ITIMER_REAL, args.overall_seconds)
        watcher.start()
        with recorder.span('orbit.common.frontend'):
            result['failed_stage'] = 'source_preflight'
            with recorder.span('source_preflight'):
                checkpoint()
                result['source_records'] = source_preflight(locations)
                write_json(out / 'source-metadata.json', result['source_records'])
                result['source_snapshots'] = _snapshot(locations, out)
                write_json(out / 'source-snapshots.json', result['source_snapshots'])
            result['failed_stage'] = 'git_metadata'
            with recorder.span('git_metadata'):
                obs = run_process(['git', 'rev-parse', '--short', 'HEAD'], cwd=locations['repo'],
                                  log=out / 'git-revision.log', resources=out / 'git-resources.jsonl',
                                  timeout=min(15, max(.01, deadline-time.monotonic())), mem_gib=.5, fsize_gib=8, env=env)
                result['git_process'] = obs
                if not _clean(obs):
                    raise RuntimeError('Git source metadata failed')
                result['git_rev'] = (out / 'git-revision.log').read_text().strip()
            actual_mlir = launch('trace')
            launch('parse', actual_mlir)
            result.update(status='ok', error=None, failed_stage=None,
                          orbit_model_trace_passed=True, tdag_parse_passed=True)
    except BaseException as exc:
        result.update(status='stopped_on_error', error=f'{type(exc).__name__}: {exc}'[:4000])
    finally:
        # Supervised children have already been reaped. Persist failures also in the gaps between children.
        stop.set()
        if watcher.is_alive():
            watcher.join()
        try:
            ended = time.perf_counter_ns()
            events = read_events(out / 'events.jsonl') + child_events
            with (out / 'combined-events.jsonl').open('w') as stream:
                for event in events:
                    stream.write(json.dumps(event) + '\n')
            timing = summarize_events(events, termination_observed_ns=ended)
            timing['outer_interval'] = dict(start_ns=started, end_ns=ended, elapsed_ns=ended-started,
                                            boundary='Pre-persistence observation; complete endpoint is in completion.json')
            write_json(out / 'timing.json', timing)
            if result['resource_monitor_error']:
                result.update(status='stopped_on_error', error=result['resource_monitor_error'])
            if not timing['attribution_ready'] or timing['diagnostics']:
                result.update(status='stopped_on_error', timing_record_error=True)
            trace, parsed = (result['phase_results'].get(k, {}) for k in ('trace', 'parse'))
            actions = observed_action_counts(result)
            result.update(actions)
            result['metrics'] = dict(status=result['status'], attempted=dict(result['attempted']), completed=dict(result['completed']),
                                     orbit_model_trace_passed=result['orbit_model_trace_passed'], tdag_parse_passed=result['tdag_parse_passed'],
                                     optimizer_ready=False, backend_ready=False, formal_matrix_complete=False,
                                     trace_worker_counts={k: trace.get(k, None if result['attempted']['trace'] else 0) for k in worker.COUNTER_FIELDS},
                                     parse_worker_counts={k: parsed.get(k, None if result['attempted']['parse'] else 0) for k in worker.COUNTER_FIELDS},
                                     recorded_pre_persistence_elapsed_seconds=(ended-started)/1e9,
                                     full_elapsed_artifact=str(out / 'completion.json'),
                                     timing_attribution_ready=timing['attribution_ready'],
                                     **actions, **dict.fromkeys(worker.UNMEASURED_FIELDS, None))
            result['outer_ru_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result['artifacts'] = dict(timing=str(out / 'timing.json'), events=str(out / 'combined-events.jsonl'),
                                       trace_result=str(work / 'trace-worker/phase_result.json'),
                                       parse_result=str(work / 'parse-worker/phase_result.json'), completion=str(out / 'completion.json'))
            result['completion_reserved_bytes'] = completion_reserve_bytes(timing)
            _terminal(out, result)
            # Include terminal result and registry persistence in the measured completion boundary.
            try:
                checkpoint()
                if artifact_bytes(out) + result['completion_reserved_bytes'] > args.case_gib * GIB:
                    raise RuntimeError('Case allowance must also cover the completion marker before registry persistence')
            except BaseException as exc:
                result.update(status='stopped_on_error', terminal_budget_error=f'{type(exc).__name__}: {exc}')
                _terminal(out, result)
            append_registry(Path(args.registry or paths.RESULTS_DIR / 'registry.jsonl'), result, out)
            marker = persist_case_completion(out, round_label=ROUND, started_ns=started,
                                             status=result['status'], timing=timing,
                                             required_artifact_bytes=artifact_bytes(out))
            if marker['case_artifact_bytes'] > args.case_gib * GIB:
                raise RuntimeError('Completion marker exhausted the case artifact allowance')
            result['completion'] = marker
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for signum, handler in saved_handlers.items():
                signal.signal(signum, handler)
            if previous_timer[0]:
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=tuple(MODEL_RECORDS))
    parser.add_argument('--out', required=True, type=Path)
    for name in ('registry', 'orbit-root', 'checkpoint', 'r13-result', 'r16-result', 'poly-dir',
                 'deps-dir', 'intake-python', 'orbit-python'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--trace-seconds', type=float, default=1800)
    parser.add_argument('--parse-seconds', type=float, default=300)
    parser.add_argument('--overall-seconds', type=float, default=3000)
    parser.add_argument('--memory-gib', type=float, default=12)
    parser.add_argument('--case-gib', type=float, default=32)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (not 0 < args.trace_seconds <= 1800 or not 0 < args.parse_seconds <= 300
            or not 0 < args.memory_gib <= 12 or not 0 < args.case_gib <= 32
            or not 0 < args.overall_seconds <= 3600):
        parser.error('Limits must be positive and within 1800s trace, 300s parse, 12GiB memory, 32GiB artifacts, 3600s overall')
    return 0 if execute(args)['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
