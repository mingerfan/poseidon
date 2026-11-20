"""F5: one isolated Orbit expression trace and one official TDAG text parse."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

from .r10_paper_metrics import (StageRecorder, UnsupportedSupervision, read_events,
                               run_process, summarize_events, write_json)


MODE = 'orbit-frontend-f5-preparation'
MODEL = 'Orbit arithmetic expression'
TRACE_SECONDS = 60
NIX_SECONDS = 90
TRACE_CHILD_SECONDS = 120
PARSE_SECONDS = 30
OVERALL_SECONDS = 180
MEMORY_GIB = 4
FILE_GIB = 8
CASE_BYTES = 4 * 1024**3
FREE_BYTES = 10 * 1024**3
MODULE = 'pt_agent.experiments.r14_orbit_frontend'
PROFILE = Path('cost_models/profiled_LATTIGONEW_CPU16k_3_16.json')
CLEANUP_PASSES = ['CSE', 'Canonicalizer', 'ElideConstant',
                  'PrivatizeConstant', 'Canonicalizer']


class PhaseStopped(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


@contextmanager
def _wall_limit(seconds):
    """Bound work in this process; R10 separately supervises child descendants."""
    def expired(signum, frame):
        raise PhaseStopped('timeout', f'The {seconds}-second phase budget expired')

    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _load_expr(frontend):
    source = Path(frontend) / 'python/hecate/hecate/expr.py'
    spec = importlib.util.spec_from_file_location('orbit_expr_check', source)
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load the official expression file: ' + str(source))
    module = importlib.util.module_from_spec(spec)
    sys.modules['orbit_expr_check'] = module
    spec.loader.exec_module(module)
    return module


def trace(args, recorder, result):
    """Use only the official expr file; hc.save evaluates the one registered body."""
    work = Path(args.work_dir).resolve()
    frontend = Path(args.frontend).resolve()
    trace_dir, constants = work / 'trace', work / 'constants'
    trace_dir.mkdir()
    constants.mkdir()
    os.environ['HECATE'] = str(frontend)
    result.update(frontend=str(frontend), expr_source=str(frontend / 'python/hecate/hecate/expr.py'),
                  hecate_environment=str(frontend), trace_runs=1,
                  expr_module_exec_attempts=0, expr_module_loaded=False,
                  frontend_ir_context_initialized=False, function_registrations=0,
                  save_attempts=0, finalize_attempts=0, original_save_cleanup_passes=CLEANUP_PASSES,
                  signature='c,c', input_ir_elements=1, arithmetic='a*b+a',
                  numerical_constants=0, mlir=None, constant_files=[])
    hc = None
    original_cwd = Path.cwd()
    os.chdir(work)
    try:
        with recorder.span('trace.load_expr'):
            result['expr_module_exec_attempts'] += 1
            hc = _load_expr(frontend)
            result['expr_module_loaded'] = True
            result['frontend_ir_context_initialized'] = bool(getattr(hc, 'ctxt', None))
            if not result['frontend_ir_context_initialized']:
                raise PhaseStopped('error', 'The official frontend returned no IR context')
        with recorder.span('trace.register_arithmetic'):
            @hc.func('c,c')
            def arithmetic(a, b):
                return a * b + a

            result['function_registrations'] = 1
        with recorder.span('trace.save'):
            result['save_attempts'] += 1
            returned = hc.save(dirs=str(trace_dir), cst_dirs=str(constants))
            result['save_returned_path'] = str(returned)
            mlir = Path(returned).resolve()
            if (not mlir.is_relative_to(trace_dir) or not mlir.is_file()
                    or mlir.stat().st_size == 0):
                raise PhaseStopped('artifact_invalid', 'save must return a nonempty file in trace/')
            result['mlir'] = dict(path=str(mlir), size_bytes=mlir.stat().st_size)
            result['constant_files'] = [dict(path=str(path.resolve()), size_bytes=path.stat().st_size)
                                        for path in sorted(constants.rglob('*')) if path.is_file()]
    finally:
        try:
            if hc is not None and result['expr_module_loaded'] and getattr(hc, 'ctxt', None):
                with recorder.span('trace.finalize'):
                    result['finalize_attempts'] += 1
                    hc.lt.finalize(hc.ctxt)
        finally:
            os.chdir(original_cwd)


def _load_parser(orbit_root):
    root = Path(orbit_root).resolve()
    sys.path.insert(0, str(root))
    params = importlib.import_module('scripts.params.params')
    parser = importlib.import_module('scripts.tdag.mlir2tdag')
    for module, relative in ((params, 'scripts/params/params.py'),
                             (parser, 'scripts/tdag/mlir2tdag.py')):
        if Path(module.__file__).resolve() != root / relative:
            raise ImportError('An unexpected scripts package shadows the official Orbit module')
    return params.Params, parser.build_from_mlir


def graph_record(graph):
    """Keep the official TDAG attributes and recover node names from its result."""
    return dict(inputs=sorted(graph.inputs), outputs=sorted(graph.outputs),
                nodes=[dict(id=node, **attributes) for node, attributes in graph.nodes(data=True)],
                edges=[dict(source=source, target=target, **attributes)
                       for source, target, attributes in graph.edges(data=True)])


def check_graph(record):
    nodes = record['nodes']
    by_id = {node['id']: node for node in nodes}
    inputs, outputs = record['inputs'], record['outputs']
    if len(nodes) != 4 or len(by_id) != 4 or len(inputs) != 2 or len(set(inputs)) != 2:
        raise ValueError('Expected four distinct nodes and two distinct inputs')
    operations = {op: [node['id'] for node in nodes if node.get('op') == op]
                  for op in ('input', 'mul', 'add')}
    if {op: len(ids) for op, ids in operations.items()} != {'input': 2, 'mul': 1, 'add': 1}:
        raise ValueError('Expected two input nodes, one mul and one add')
    if set(inputs) != set(operations['input']):
        raise ValueError('TDAG input identifiers differ from the input nodes')
    mul, add = operations['mul'][0], operations['add'][0]
    if outputs != [add]:
        raise ValueError('The single output must be the add node')
    if any('level' not in node or 'scale' not in node or node['level'] is not None
           or node['scale'] is not None for node in nodes):
        raise ValueError('Compile-mode node level and scale must both be None')
    edges = [(edge['source'], edge['target']) for edge in record['edges']]
    expected = {(inputs[0], mul), (inputs[1], mul), (mul, add), (inputs[0], add)}
    if len(edges) != 4 or set(edges) != expected:
        raise ValueError('The four edges must preserve a*b+a with the first input reused')
    pending = {node: 0 for node in by_id}
    adjacency = {node: [] for node in by_id}
    for source, target in edges:
        if source not in by_id or target not in by_id:
            raise ValueError('An edge references an unknown node')
        pending[target] += 1
        adjacency[source].append(target)
    ready = [node for node, count in pending.items() if count == 0]
    visited = 0
    while ready:
        source = ready.pop()
        visited += 1
        for target in adjacency[source]:
            pending[target] -= 1
            if pending[target] == 0:
                ready.append(target)
    if visited != len(nodes):
        raise ValueError('The parsed graph must be acyclic')
    return dict(passed=True, nodes=4, edges=4, input_order=inputs, mul=mul, add=add,
                output=add, acyclic=True, compile_level_scale_unassigned=True)


def parse(args, recorder, result):
    root = Path(args.orbit_root).resolve()
    profile = Path(args.profile).resolve()
    mlir = Path(args.mlir).resolve()
    if profile != root / PROFILE:
        raise ValueError('F5 uses only the existing official 16k profile for parser metadata')
    result.update(orbit_root=str(root), profile=str(profile), mlir=str(mlir), parse_runs=1,
                  params_calls=0, build_from_mlir_calls=0, graph_check=dict(passed=False))
    with recorder.span('parse.import_modules'):
        Params, build_from_mlir = _load_parser(root)
    with recorder.span('parse.params'):
        result['params_calls'] += 1
        params = Params(str(profile), 'Orbit', 'compile', threads=1)
        result['parser_metadata'] = dict(profile_use='parser_metadata_only', mode=params.mode,
                                         sysname=params.sysname, threads=params.threads,
                                         max_slot=params.max_slot, poly_deg=params.poly_deg)
    with recorder.span('parse.build_from_mlir'):
        result['build_from_mlir_calls'] += 1
        graph = build_from_mlir(str(mlir), params)
    with recorder.span('parse.save_graph'):
        record = graph_record(graph)
        path = Path(args.work_dir) / 'tdag.json'
        write_json(path, record)
        result['tdag'] = str(path.resolve())
        result['graph'] = record
    with recorder.span('parse.check_graph'):
        result['graph_check'] = check_graph(record)


def run_phase(args):
    """Inner phases consume argv only and never locate the outer result registry."""
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if (work / 'phase_result.json').exists() or (work / 'events.jsonl').exists():
        raise ValueError('Existing inner phase evidence is preserved')
    result = dict(phase=args.phase, status='error', error=None, model_runs=0,
                  optimizer_runs=0, keygen_runs=0, real_ckks_attempts=0,
                  real_ckks_completed=0, llm_calls=0, builds=0, downloads=0, installations=0)
    recorder = StageRecorder(work / 'events.jsonl')
    started = time.perf_counter_ns()
    try:
        with recorder.span('f5.' + args.phase):
            with _wall_limit(TRACE_SECONDS if args.phase == 'trace' else PARSE_SECONDS):
                (trace if args.phase == 'trace' else parse)(args, recorder, result)
        result['status'] = 'ok'
    except PhaseStopped as exc:
        result.update(status=exc.status, error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    except (ImportError, FileNotFoundError) as exc:
        result.update(status='dependency_missing', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        result.update(elapsed_ns=time.perf_counter_ns() - started,
                      single_pid_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(work / 'phase_result.json', result)
    return result


def _locations():
    # Kept inside the outer entry: inner phases receive every location through argv.
    from .. import paths

    return dict(results=paths.RESULTS_DIR.resolve(), repo=paths.REPO,
                scripts=paths.REPO / 'DSL/DSL_agent/scripts', orbit=paths.ORBIT_ROOT.resolve(),
                frontend=(paths.ORBIT_ROOT / 'frontend/dacapo').resolve(),
                intake_python=paths.INTAKE_PYTHON.absolute(),
                orbit_python=paths.WORKSPACE / 'baselines/orbit-venv/bin/python',
                profile=(paths.ORBIT_ROOT / PROFILE).resolve())


def _git_state(repo):
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=repo, capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=repo, capture_output=True,
                           text=True, check=True).stdout
    return dict(branch=branch, working_tree_changed=bool(dirty))


def _case_size(out):
    return sum(path.stat().st_size for path in Path(out).rglob('*') if path.is_file())


def _check_budget(out):
    size = _case_size(out)
    if size > CASE_BYTES:
        raise PhaseStopped('artifact_budget_exceeded', 'The case artifacts exceed 4 GiB')
    if shutil.disk_usage(out).free < FREE_BYTES:
        raise PhaseStopped('dependency_missing', 'At least 10 GiB free disk space is required')
    return size


def _write_terminal_result(out, result):
    """Include the terminal JSON's serialized size in the complete case metadata."""
    path = Path(out) / 'result.json'
    base = _case_size(out) - (path.stat().st_size if path.exists() else 0)
    for _ in range(8):
        total = base + len((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
        within = total <= CASE_BYTES
        changed = (result.get('case_artifact_bytes') != total
                   or result['metrics'].get('case_artifact_bytes') != total
                   or result['metrics'].get('case_artifact_budget_passed') != within
                   or result['metrics'].get('status') != result['status'])
        result['case_artifact_bytes'] = total
        result['metrics'].update(case_artifact_bytes=total, case_artifact_budget_passed=within)
        if not within and result['status'] != 'artifact_budget_exceeded':
            result['status_before_artifact_budget'] = result['status']
            result.update(status='artifact_budget_exceeded',
                          error='The complete F5 case exceeds its 4 GiB artifact allowance')
            changed = True
        result['metrics']['status'] = result['status']
        if not changed:
            write_json(path, result)
            return
    raise RuntimeError('F5 terminal artifact accounting did not settle')


def _phase_commands(locations, work, mlir=None):
    if mlir is None:
        return [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun',
                '--seconds', str(NIX_SECONDS), '--python', str(locations['intake_python']), '--',
                '-m', MODULE, '--phase', 'trace', '--work-dir', str(work),
                '--frontend', str(locations['frontend'])]
    return [str(locations['orbit_python']), '-B', '-m', MODULE, '--phase', 'parse',
            '--work-dir', str(work), '--orbit-root', str(locations['orbit']),
            '--profile', str(locations['profile']), '--mlir', str(mlir)]


def execute(args):
    locations = _locations()
    out = Path(args.out).absolute()
    if out.resolve() == locations['results'] or not out.resolve().is_relative_to(locations['results']):
        raise ValueError('--out must name a new child directory of RESULTS_DIR')
    if out.exists():
        raise ValueError('Existing run directories are preserved')
    out.mkdir(parents=True)
    (out / 'resources.jsonl').touch()
    recorder = StageRecorder(out / 'events.jsonl')
    started = time.perf_counter_ns()
    processes, phase_results, child_events = [], {}, []
    launch_attempts = dict(trace=0, parse=0)
    status, error, git_state = 'ok', None, None
    try:
        with recorder.span('f5_preparation'):
            with _wall_limit(OVERALL_SECONDS):
                with recorder.span('preflight'):
                    git_state = _git_state(locations['repo'])
                    required = [locations['frontend'] / 'python/hecate/hecate/expr.py',
                                locations['frontend'] / 'build/lib/libHecateFrontend.so',
                                locations['orbit'] / 'scripts/params/params.py',
                                locations['orbit'] / 'scripts/tdag/mlir2tdag.py',
                                locations['profile']]
                    missing = [str(path) for path in required if not path.is_file()]
                    missing += [str(locations[key]) for key in ('intake_python', 'orbit_python')
                                if not locations[key].is_file() and not locations[key].is_symlink()]
                    if missing:
                        raise PhaseStopped('dependency_missing', 'Missing local prerequisites: ' + ', '.join(missing))
                    _check_budget(out)
                for phase in ('trace', 'parse'):
                    work = out / (phase + '-worker')
                    work.mkdir()
                    mlir = phase_results['trace']['mlir']['path'] if phase == 'parse' else None
                    argv = _phase_commands(locations, work, mlir)
                    timeout = TRACE_CHILD_SECONDS if phase == 'trace' else PARSE_SECONDS
                    remaining = OVERALL_SECONDS - (time.perf_counter_ns() - started) / 1e9
                    if remaining <= 0:
                        raise PhaseStopped('timeout', 'The overall 180-second budget expired')
                    process = None
                    with recorder.span('supervise.' + phase, wall_seconds=timeout) as parent:
                        try:
                            launch_attempts[phase] += 1
                            process = run_process(
                                argv, cwd=locations['scripts'], log=out / (phase + '.log'),
                                resources=out / 'resources.jsonl', timeout=min(timeout, remaining),
                                mem_gib=MEMORY_GIB, fsize_gib=FILE_GIB,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                                         OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                                         MKL_NUM_THREADS='1'))
                            processes.append(dict(phase=phase, command=argv, **process))
                        finally:
                            for event in read_events(work / 'events.jsonl'):
                                if event.get('type') in ('begin', 'end') and event.get('parent_id') is None:
                                    event['parent_id'] = parent['event_id']
                                child_events.append(event)
                            result_path = work / 'phase_result.json'
                            if result_path.is_file():
                                phase_results[phase] = json.loads(result_path.read_text())
                        _check_budget(out)
                        if process is None:
                            raise PhaseStopped('process_failed', phase + ': missing process observation')
                        if process['timed_out']:
                            raise PhaseStopped('timeout', phase + ': child time budget expired')
                        if process['signal'] or process['remaining_pids']:
                            raise PhaseStopped('process_failed', phase + ': signal or residual process')
                        worker = phase_results.get(phase)
                        if worker is None:
                            raise PhaseStopped('result_missing', phase + ': no phase result')
                        if worker.get('status') != 'ok':
                            raise PhaseStopped(worker.get('status', 'error'), str(worker.get('error')))
                        if process['exit_code'] != 0:
                            raise PhaseStopped('process_failed', phase + ': nonzero child exit')
                        if phase == 'parse' and worker.get('graph_check', {}).get('passed') is not True:
                            raise PhaseStopped('incomplete', 'The parsed graph acceptance is missing')
    except PhaseStopped as exc:
        status, error = exc.status, str(exc)
    except UnsupportedSupervision as exc:
        status, error = 'unsupported_process_supervision', str(exc)
    except BaseException as exc:
        status, error = 'error', f'{type(exc).__name__}: {exc}'
    finally:
        ended = time.perf_counter_ns()
        events = read_events(out / 'events.jsonl') + child_events
        timing = summarize_events(events, termination_observed_ns=ended)
        timing['outer_interval'] = dict(start_ns=started, end_ns=ended, elapsed_ns=ended - started)
        with (out / 'combined-events.jsonl').open('w') as stream:
            for event in events:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        with (out / 'stages.jsonl').open('w') as stream:
            for span in timing['spans']:
                stream.write(json.dumps(span, ensure_ascii=False) + '\n')
        trace_result, parse_result = phase_results.get('trace', {}), phase_results.get('parse', {})
        metrics = dict(status=status, phase_launch_attempts=launch_attempts,
                       trace_runs=trace_result.get('trace_runs', None if launch_attempts['trace'] else 0),
                       parse_runs=parse_result.get('parse_runs', None if launch_attempts['parse'] else 0),
                       save_attempts=trace_result.get('save_attempts'),
                       finalize_attempts=trace_result.get('finalize_attempts'),
                       graph_passed=parse_result.get('graph_check', {}).get('passed'),
                       model_runs=0, optimizer_runs=0, keygen_runs=0, real_ckks_attempts=0,
                       real_ckks_completed=0, llm_calls=0, builds=0, downloads=0, installations=0,
                       compiled_bootstraps=None, predicted_latency=None, CKKS_latency=None,
                       elapsed_ns=ended - started)
        result = dict(run_id=out.name, round='preparation', mode=MODE, model=MODEL, status=status,
                      error=error, git_rev=git_state, phase_results=phase_results, processes=processes,
                      timing=timing, metrics=metrics, artifact_bytes_before_result=_case_size(out))
        _write_terminal_result(out, result)
        with (locations['results'] / 'registry.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(run_id=out.name, round='preparation', model=MODEL, mode=MODE,
                                         git_rev=git_state, metrics=metrics, artifacts_dir=str(out),
                                         ts=datetime.now(timezone.utc).isoformat()), ensure_ascii=False) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('outer', 'trace', 'parse'), default='outer')
    for name in ('out', 'work-dir', 'frontend', 'orbit-root', 'profile', 'mlir'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    required = {'outer': ('out',), 'trace': ('work_dir', 'frontend'),
                'parse': ('work_dir', 'orbit_root', 'profile', 'mlir')}[args.phase]
    if any(getattr(args, name) is None for name in required):
        parser.error(args.phase + ' requires ' + ', '.join('--' + name.replace('_', '-') for name in required))
    result = execute(args) if args.phase == 'outer' else run_phase(args)
    print(json.dumps(dict(phase=args.phase, status=result['status'], error=result.get('error'))), flush=True)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
