"""R23: argv-only observation of one native Orbit optimizer and its output."""

import argparse
from collections import Counter, deque
from contextlib import contextmanager
import importlib
import json
import math
import numbers
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import threading
import time
import traceback
from types import SimpleNamespace

from .r10_paper_metrics import StageRecorder, write_json


ROUND = 'R23'
MODEL = 'pt_agent.experiments.r13_common_models:resnet20_silu'
MODE = 'orbit-resnet20-native-optimizer-preflight'
PROFILE = Path('cost_models/profiled_LATTIGONEW_CPU64k_3_16.json')
OPTIMIZE_SECONDS = 7200
VALIDATE_SECONDS = 300
GIB = 1024 ** 3
CASE_BYTES = 4 * GIB
FREE_BYTES = 10 * GIB
COUNT_FIELDS = (
    'optimize_attempts', 'optimize_completed', 'validate_attempts', 'validate_completed',
    'params_calls', 'params_completed', 'seed_calls', 'seed_completed',
    'build_from_mlir_calls', 'build_from_mlir_completed', 'check_tdag_calls',
    'check_tdag_completed', 'solve_ilp_attempts', 'solve_ilp_completed',
    'solve_ilp_feasible', 'solve_ilp_infeasible', 'solve_ilp_errors',
)
SOLVE_FIELDS = tuple(name for name in COUNT_FIELDS if name.startswith('solve_ilp_'))
ZERO_FIELDS = (
    'llm_calls', 'model_forward_runs', 'trace_runs', 'keygen_runs',
    'real_ckks_attempts', 'real_ckks_completed', 'downloads', 'installations',
    'confirm_samples', 'report_samples', 'freeze_attempts', 'heir_calls',
    'compiler_runs', 'checkpoint_load_attempts', 'model_build_attempts', 'sample_reads',
)
UNMEASURED_FIELDS = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency',
                     'accuracy_acceptance')
STAGE_FIELDS = ('source_checked', 'params_checked', 'native_optimizer_completed',
                'placement_artifact_checked', 'environment_restored')
_THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


class WorkerFailure(RuntimeError):
    def __init__(self, message, status='error'):
        super().__init__(message)
        self.status = status


def _initial_result(phase):
    result = dict(round=ROUND, model=MODEL, mode=MODE, phase=phase, status='running',
                  error=None, failed_phase=None, source_record=None,
                  params=None, profile_metadata=None, optimizer_runs=0, solver_runs=0,
                  model_optimize_calls=None, backend_ready=False,
                  formal_matrix_complete=False, security_validated=False,
                  accuracy_validated=False, phase_artifact_bytes=0,
                  case_artifact_bytes=0, placement_bootstrap_nodes=None,
                  placement_bootstrap_weight_sum=None, budget_monitor_completed=False)
    result.update(dict.fromkeys(STAGE_FIELDS, False))
    result.update(dict.fromkeys(COUNT_FIELDS, 0))
    result.update(dict.fromkeys(ZERO_FIELDS, 0))
    result.update(dict.fromkeys(UNMEASURED_FIELDS, None))
    return result


def _json_safe(value):
    """Represent budget metadata without altering the objects given to Orbit."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else repr(number)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _read_record(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise WorkerFailure('The explicit source record must be a JSON object', 'source_failed')
    return value


def _same_path(left, right):
    return isinstance(left, (str, Path)) and Path(left).resolve() == Path(right).resolve()


def _file_record(path):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise WorkerFailure('A required source/artifact file is absent or empty: ' + str(path),
                            'source_failed')
    return dict(path=str(path), size_bytes=path.stat().st_size)


def _selected_paths(args):
    root, work = Path(args.orbit_root).resolve(), Path(args.work_dir).resolve()
    profile, license_file = Path(args.profile).resolve(), Path(args.license_file).resolve()
    output, input_file = Path(args.output_mlir).resolve(), Path(args.input_mlir).resolve()
    expected_license = (root.parent / 'orbit-venv/lib/python3.11/site-packages/'
                        'gurobipy/.libs/gurobi.lic').resolve()
    if profile != root / PROFILE or license_file != expected_license:
        raise WorkerFailure('R23 uses the fixed original profile and bundled local license',
                            'source_failed')
    if output != work.parent / 'optimized.mlir' or input_file == output:
        raise WorkerFailure('R23 writes only the new work/optimized.mlir', 'source_failed')
    _file_record(profile)
    license_metadata = _file_record(license_file)
    return root, work, profile, input_file, output, license_metadata


def _phase_size(directory):
    total = 0
    for parent, _, files in os.walk(directory):
        for filename in files:
            try:
                total += (Path(parent) / filename).stat().st_size
            except FileNotFoundError:
                pass
    return total


def _budget_observation(args, started=None):
    work = Path(args.work_dir).resolve()
    case = work.parent.parent
    return dict(ts_ns=time.perf_counter_ns(), case_root=str(case),
                case_artifact_bytes=_phase_size(case),
                disk_free_bytes=shutil.disk_usage(work).free,
                elapsed_ns=None if started is None else time.perf_counter_ns() - started)


def _budget_error(observation, seconds=None):
    if observation['case_artifact_bytes'] > CASE_BYTES:
        return 'artifact_budget_exceeded', 'R23 case artifacts exceed 4 GiB'
    if observation['disk_free_bytes'] < FREE_BYTES:
        return 'disk_reserve', 'R23 disk reserve is below 10 GiB'
    if (seconds is not None and observation['elapsed_ns'] is not None
            and observation['elapsed_ns'] >= int(seconds * 1e9)):
        return 'timeout', 'R23 inner phase wall allowance exhausted'
    return None


def _checkpoint(args, result):
    work = Path(args.work_dir).resolve()
    observation = _budget_observation(args)
    result['phase_artifact_bytes'] = _phase_size(work)
    result['case_artifact_bytes'] = observation['case_artifact_bytes']
    result['last_budget_observation'] = observation
    write_json(work / 'phase_result.json', result)
    failure = _budget_error(observation)
    if failure:
        raise WorkerFailure(failure[1], failure[0])


@contextmanager
def _phase(args, recorder, result, name):
    result['failed_phase'] = name
    _checkpoint(args, result)
    with recorder.span(name):
        yield
    _checkpoint(args, result)


def _terminate_self():
    os.kill(os.getpid(), signal.SIGTERM)


@contextmanager
def _budget_guard(args, result, seconds):
    """Observe metadata independently of an optimizer blocked in native code."""
    work = Path(args.work_dir).resolve()
    stop = threading.Event()
    started = time.perf_counter_ns()

    def watch():
        while not stop.wait(1):
            try:
                observation = _budget_observation(args, started)
                with (work / 'budget-samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(observation) + '\n')
                    stream.flush()
                failure = _budget_error(observation, seconds)
                if failure is None:
                    continue
                try:
                    write_json(work / 'budget-fatal.json',
                               dict(status=failure[0], error=failure[1],
                                    termination_origin='worker_budget_monitor',
                                    **observation))
                finally:
                    _terminate_self()
                return
            except BaseException as exc:
                try:
                    write_json(work / 'budget-monitor-error.json',
                               dict(error=f'{type(exc).__name__}: {exc}'[:2000],
                                    ts_ns=time.perf_counter_ns(),
                                    termination_origin='worker_budget_monitor_error'))
                finally:
                    _terminate_self()
                return

    thread = threading.Thread(target=watch, name='r23-budget-observer', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)
        result['budget_monitor_completed'] = not thread.is_alive()
        if thread.is_alive():
            raise WorkerFailure('R23 metadata observer did not stop', 'record_error')


@contextmanager
def _worker_environment(args, result):
    """Keep HOME and restore temporary process-local imports and settings."""
    root, work, _, _, _, license_metadata = _selected_paths(args)
    saved_environment = dict(os.environ)
    saved_modules, saved_path, saved_cwd = dict(sys.modules), list(sys.path), Path.cwd()
    saved_bytecode = sys.dont_write_bytecode
    saved_sigterm = signal.getsignal(signal.SIGTERM)
    removed = sorted(name for name in os.environ if name.startswith(('GRB_', 'GUROBI_')))
    try:
        for name in removed:
            del os.environ[name]
        os.environ['GRB_LICENSE_FILE'] = license_metadata['path']
        for name, relative in (('TMPDIR', 'tmp'), ('TMP', 'tmp'), ('TEMP', 'tmp'),
                               ('XDG_CACHE_HOME', 'cache'), ('XDG_CONFIG_HOME', 'config'),
                               ('MPLCONFIGDIR', 'matplotlib')):
            directory = work / 'environment' / relative
            directory.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(directory)
        for name in _THREAD_ENV:
            os.environ[name] = '1'
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.environ['MPLBACKEND'] = 'Agg'
        sys.dont_write_bytecode = True
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.chdir(work)
        sys.path.insert(0, str(root))
        result['environment'] = dict(
            removed_environment_keys=removed, bundled_license=license_metadata,
            home_preserved=True, writable_environment={name: os.environ[name] for name in
                ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'MPLCONFIGDIR')},
            thread_environment={name: '1' for name in _THREAD_ENV},
            native_sigterm_default=True, global_isolation_claimed=False)
        yield
    finally:
        for name in tuple(sys.modules):
            if name not in saved_modules:
                del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_bytecode
        os.chdir(saved_cwd)
        for name in set(os.environ).union(saved_environment):
            if name == 'HOME':
                continue
            if name in saved_environment:
                os.environ[name] = saved_environment[name]
            else:
                os.environ.pop(name, None)
        signal.signal(signal.SIGTERM, saved_sigterm)
        result['environment_restored'] = True


def make_params(profile, Params, *, mode) -> tuple:
    if mode not in ('compile', 'execute'):
        raise WorkerFailure('R23 Params mode must be compile or execute', 'params_failed')
    declaration = _read_record(profile)
    expected = dict(polynomialDegree=131072, rescalingFactor=51,
                    levelLowerBound=1, levelUpperBound=16,
                    bootstrapLevelLowerBound=3, bootstrapLevelUpperBound=16,
                    runtime='Lattigo')
    if any(declaration.get(name) != value for name, value in expected.items()):
        raise WorkerFailure('The original R23 64k profile metadata differs', 'params_failed')
    params = Params(str(profile), 'Orbit', mode, Sw=40, CSw=40, bpsdepth=15,
                    threads=1, comp=True, part=True, reqbp=False, netname='ResNetSiLU64k')
    metadata = dict(profile=str(Path(profile).resolve()),
                    profile_use='external_native_optimization_profile',
                    declared_field='polynomialDegree', declared_polynomial_degree=131072,
                    original_poly_deg=params.poly_deg, original_max_slot=params.max_slot,
                    alias_adapted_in_memory=True, security_validated=False)
    params.poly_deg, params.max_slot = declaration['polynomialDegree'], 65536
    actual = {name: getattr(params, name) for name in
              ('poly_deg', 'max_slot', 'Sf', 'Sw', 'Csw', 'lvl_lb', 'lvl_ub',
               'bts_lb', 'bts_ub', 'backend', 'sysname', 'mode', 'bpsdepth',
               'threads', 'comp', 'part', 'reqbp', 'netname', 'trunc_val',
               'dacapo_mlir_in', 'dacapo_mlir_out')}
    required = dict(poly_deg=131072, max_slot=65536, Sf=51, Sw=40, Csw=40,
                    lvl_lb=1, lvl_ub=16, bts_lb=3, bts_ub=16, backend='Lattigo',
                    sysname='Orbit', mode=mode, bpsdepth=15, threads=1,
                    comp=True, part=True, reqbp=False, netname='ResNetSiLU64k',
                    trunc_val=1, dacapo_mlir_in=True, dacapo_mlir_out=True)
    if actual != required:
        raise WorkerFailure('R23 native Params did not retain the fixed configuration',
                            'params_failed')
    metadata.update(actual)
    return params, metadata


def _check_r22_source(args):
    source_path = Path(args.source_record).resolve()
    source = _read_record(source_path)
    if (source.get('round') != 'R22' or source.get('model') != MODEL
            or source.get('mode') != 'orbit-resnet20-trace-tdag-preflight'
            or source.get('status') != 'ok'
            or source.get('orbit_model_trace_passed') is not True
            or source.get('tdag_parse_passed') is not True):
        raise WorkerFailure('R23 requires the successful complete R22 source', 'source_failed')
    for field in ('attempted', 'completed'):
        counts = source.get(field) or {}
        if any(counts.get(name) != 1 for name in
               ('metadata_download', 'wheel_download', 'dependency_install', 'tests', 'trace', 'parse')):
            raise WorkerFailure('The saved R22 actual attempt counts differ', 'source_failed')
    phases = source.get('phase_results') or {}
    trace, parse = phases.get('trace') or {}, phases.get('parse') or {}
    graph, profile = parse.get('graph_check') or {}, parse.get('parser_metadata') or {}
    if (trace.get('status') != 'ok' or parse.get('status') != 'ok'
            or trace.get('activation_calls') != 19
            or len(trace.get('activation_names') or []) != 19
            or graph.get('passed') is not True or graph.get('node_count') != 10113
            or graph.get('edge_count') != 13342 or graph.get('acyclic') is not True
            or graph.get('input_reaches_output') is not True
            or profile.get('poly_deg') != 131072 or profile.get('max_slot') != 65536
            or not _same_path(profile.get('profile'), args.profile)):
        raise WorkerFailure('The complete R22 graph/profile metadata differs', 'source_failed')
    mlir = trace.get('mlir') or {}
    actual = _file_record(args.input_mlir)
    trace_directory = source_path.parent / 'work/trace-worker/trace'
    if (not _same_path(mlir.get('path'), args.input_mlir)
            or not Path(args.input_mlir).resolve().is_relative_to(trace_directory)
            or actual['size_bytes'] != mlir.get('size_bytes')):
        raise WorkerFailure('R23 input must be the preserved R22 trace file', 'source_failed')
    snapshot = Path(args.work_dir) / 'source-record.json'
    write_json(snapshot, source)
    return dict(source_record=str(source_path), source_snapshot=str(snapshot.resolve()),
                source_round='R22', source_status='ok', input_mlir=actual,
                original_graph_nodes=10113, original_graph_edges=13342,
                activation_count=19, inherited_profile=profile,
                source_times_included=False, weights_reopened=False, samples_reopened=False,
                training_preprocessing_verified=False,
                coefficient_source_equivalence_verified=False)


def _validate_source(args):
    source_path = Path(args.source_record).resolve()
    source = _read_record(source_path)
    if (source.get('round') != ROUND or source.get('model') != MODEL
            or source.get('mode') != MODE or source.get('phase') != 'optimize'
            or source.get('status') != 'ok' or source.get('native_optimizer_completed') is not True
            or source.get('optimize_attempts') != 1 or source.get('optimize_completed') != 1
            or source.get('source_checked') is not True
            or not _same_path(source.get('input_mlir', {}).get('path'), args.input_mlir)
            or not _same_path(source.get('output_mlir', {}).get('path'), args.output_mlir)):
        raise WorkerFailure('R23 validation requires the single successful optimization',
                            'source_failed')
    if (source_path.parent / 'first-fatal.json').exists():
        raise WorkerFailure('An optimizer with a caught fatal exception is not validated',
                            'source_failed')
    parameters = source.get('params') or {}
    for name, value in dict(mode='compile', poly_deg=131072, max_slot=65536, Sf=51,
                            Sw=40, Csw=40, lvl_lb=1, lvl_ub=16, bts_lb=3, bts_ub=16,
                            threads=1, comp=True, part=True, reqbp=False, bpsdepth=15,
                            dacapo_mlir_in=True, dacapo_mlir_out=True).items():
        if parameters.get(name) != value:
            raise WorkerFailure('Saved R23 optimization parameters differ', 'source_failed')
    output = _file_record(args.output_mlir)
    if output['size_bytes'] != source['output_mlir'].get('size_bytes'):
        raise WorkerFailure('Saved optimized MLIR metadata differs', 'source_failed')
    snapshot = Path(args.work_dir) / 'source-record.json'
    write_json(snapshot, source)
    return dict(source_record=str(source_path), source_snapshot=str(snapshot.resolve()),
                source_round=ROUND, source_phase='optimize', source_status='ok',
                source_params=parameters, output_mlir=output, source_times_included=False)


def _checked_import(root, name, relative):
    module = importlib.import_module(name)
    if not _same_path(getattr(module, '__file__', None), Path(root) / relative):
        raise ImportError('Unexpected Orbit module origin: ' + name)
    return module


def _load_optimizer(root):
    optimizer = _checked_import(root, 'scripts.optimizer.orbit.optimizer',
                                'scripts/optimizer/orbit/optimizer.py')
    worker = _checked_import(root, 'scripts.optimizer.orbit.ilp_worker',
                             'scripts/optimizer/orbit/ilp_worker.py')
    _checked_import(root, 'scripts.optimizer.orbit.ilp_core',
                    'scripts/optimizer/orbit/ilp_core.py')
    return optimizer, worker


def _load_validation(root):
    from .r14_orbit_frontend import _load_parser
    Params, build_from_mlir = _load_parser(root)
    checks = _checked_import(root, 'scripts.tdag.check_tdag', 'scripts/tdag/check_tdag.py')
    return Params, build_from_mlir, checks.check_tdag


def _set_seed(optimizer_module):
    optimizer_module.set_global_seed(42)


def _argument(arguments, keywords, name, position):
    return keywords[name] if name in keywords else arguments[position] if len(arguments) > position else None


@contextmanager
def install_observers(ilp_worker, recorder, work_dir) -> object:
    """Wrap Python delegation and this module's traceback name, not Gurobi C APIs."""
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    events_path, summary_path = work / 'solve-ilp-events.jsonl', work / 'solve-ilp-summary.json'
    fatal_path = work / 'first-fatal.json'
    if any(path.exists() for path in (events_path, summary_path, fatal_path)):
        raise WorkerFailure('R23 preserves existing optimizer observation records', 'record_error')
    original_solve, original_traceback = ilp_worker.solve_ilp, ilp_worker.traceback
    original_print_exc = original_traceback.print_exc
    expected_code = ilp_worker.ILP_Worker._lsabts_worker.__code__
    counters = dict.fromkeys(SOLVE_FIELDS, 0)
    lock = threading.RLock()
    state = dict(counters=counters, events_path=str(events_path), summary_path=str(summary_path),
                 first_fatal_path=str(fatal_path), restored=False,
                 model_optimize_calls=None, solve_clock='perf_counter_ns; separate ILP event stream')

    def summary():
        write_json(summary_path, dict(round=ROUND, **counters, model_optimize_calls=None,
                                       observed_ns=time.perf_counter_ns()))

    def emit(event):
        with events_path.open('a') as stream:
            stream.write(json.dumps(_json_safe(event), ensure_ascii=False) + '\n')
            stream.flush()
        summary()

    def observed_solve(*args, **kwargs):
        dag = _argument(args, kwargs, 'tdag', 0)
        budget = _argument(args, kwargs, 'io_budgets', 1)
        task = _argument(args, kwargs, 'task_name', 3)
        started = time.perf_counter_ns()
        with lock:
            counters['solve_ilp_attempts'] += 1
            call_index = counters['solve_ilp_attempts']
            event = dict(call_index=call_index, task_name=task, pid=os.getpid(),
                         thread_id=threading.get_ident(), thread_name=threading.current_thread().name,
                         start_ns=started, partition_nodes=len(dag.nodes) if hasattr(dag, 'nodes') else None,
                         partition_edges=len(dag.edges) if hasattr(dag, 'edges') else None,
                         io_budget=_json_safe(budget))
            emit(dict(event, type='begin'))
        try:
            answer = original_solve(*args, **kwargs)
        except BaseException as exc:
            with lock:
                counters['solve_ilp_errors'] += 1
                emit(dict(event, type='error', end_ns=time.perf_counter_ns(),
                          error_type=type(exc).__name__, error=str(exc)[:2000],
                          traceback=''.join(traceback.format_exception(*sys.exc_info()))))
            raise
        else:
            infeasible = isinstance(answer, tuple) and len(answer) == 2 and answer[0] is None and answer[1] is None
            with lock:
                counters['solve_ilp_completed'] += 1
                counters['solve_ilp_infeasible' if infeasible else 'solve_ilp_feasible'] += 1
                emit(dict(event, type='end', end_ns=time.perf_counter_ns(),
                          result='infeasible' if infeasible else 'solution',
                          solution_present=not infeasible))
            return answer

    def caught_exception(*args, **kwargs):
        caller = sys._getframe(1)
        if caller.f_code is not expected_code:
            return original_print_exc(*args, **kwargs)
        exception = sys.exc_info()
        try:
            with lock:
                if not fatal_path.exists():
                    write_json(fatal_path, dict(
                        round=ROUND, termination_origin='worker_caught_exception',
                        error_type=None if exception[0] is None else exception[0].__name__,
                        error=None if exception[1] is None else str(exception[1])[:2000],
                        traceback=''.join(traceback.format_exception(*exception)),
                        task_name=caller.f_locals.get('task_name'), pid=os.getpid(),
                        thread_id=threading.get_ident(), thread_name=threading.current_thread().name,
                        observed_ns=time.perf_counter_ns(), counters=dict(counters),
                        model_optimize_calls=None))
                summary()
            original_print_exc(*args, **kwargs)
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            _terminate_self()

    summary()
    ilp_worker.solve_ilp = observed_solve
    ilp_worker.traceback = SimpleNamespace(print_exc=caught_exception)
    try:
        yield state
    finally:
        ilp_worker.solve_ilp = original_solve
        ilp_worker.traceback = original_traceback
        state['restored'] = True
        with lock:
            summary()


def _copy_observer_counts(state, result):
    result.update(state['counters'])
    result['solver_runs'] = state['counters']['solve_ilp_attempts']
    result['solve_ilp_records'] = {key: state[key] for key in
                                  ('events_path', 'summary_path', 'first_fatal_path')}
    result['observer_restored'] = state['restored']


def optimize_model(args, recorder, result) -> None:
    root, work, profile, input_file, output, _ = _selected_paths(args)
    result['source_record'] = str(Path(args.source_record).resolve())
    with _phase(args, recorder, result, 'optimize.source_records'):
        result['source_summary'] = _check_r22_source(args)
        result['source_checked'] = True
        result['input_mlir'] = _file_record(input_file)
        if output.exists():
            raise WorkerFailure('R23 never replaces an existing optimized MLIR', 'source_failed')
    with _phase(args, recorder, result, 'optimize.import_modules'):
        optimizer, ilp_worker = _load_optimizer(root)
    with _phase(args, recorder, result, 'optimize.params'):
        result['params_calls'] += 1
        _checkpoint(args, result)
        params, metadata = make_params(profile, optimizer.Params, mode='compile')
        result.update(params=metadata, profile_metadata=metadata, params_checked=True)
        result['params_completed'] += 1
    with _phase(args, recorder, result, 'optimize.seed'):
        result['seed_calls'] += 1
        _checkpoint(args, result)
        _set_seed(optimizer)
        result['seed_completed'] += 1
        result['seed_scope'] = 'official set_global_seed(42), including optional Torch/CUDA seeding; no model forward'
    observer = None
    try:
        with install_observers(ilp_worker, recorder, work) as observer:
            result['solve_ilp_records'] = {key: observer[key] for key in
                                          ('events_path', 'summary_path', 'first_fatal_path')}
            with _phase(args, recorder, result, 'optimizer.run'):
                result['optimize_attempts'] += 1
                result['optimizer_runs'] = result['optimize_attempts']
                _checkpoint(args, result)
                optimizer.run(str(input_file), str(output), params)
                result['optimize_completed'] += 1
                result['native_optimizer_completed'] = True
                _copy_observer_counts(observer, result)
                _checkpoint(args, result)
    finally:
        if observer is not None:
            _copy_observer_counts(observer, result)
    with _phase(args, recorder, result, 'optimize.output_metadata'):
        result['output_mlir'] = _file_record(output)
        if (work / 'first-fatal.json').exists():
            raise WorkerFailure('The optimizer recorded a caught fatal exception', 'optimizer_failed')
        result['external_profile_estimate'] = dict(
            source=str(profile), source_kind='original external latencyTable',
            values_source='native optimizer stdout/stderr only; not recomputed',
            native_cost_unit='microseconds', native_seconds_conversion='cost / 1000000',
            local_latency_calibrated=False, predicted_latency=None, CKKS_latency=None)


def _placement_record(record):
    nodes, edges = record['nodes'], record['edges']
    by_id = {node['id']: node for node in nodes}
    inputs, outputs = record['inputs'], record['outputs']
    if (len(by_id) != len(nodes) or len(inputs) != 1 or len(outputs) != 1
            or inputs[0] not in by_id or outputs[0] not in by_id
            or by_id[inputs[0]].get('op') != 'input'):
        raise WorkerFailure('Placed TDAG must have unique nodes and one input/output', 'placement_failed')
    following, previous = {name: set() for name in by_id}, {name: set() for name in by_id}
    indegree = dict.fromkeys(by_id, 0)
    for edge in edges:
        source, target = edge['source'], edge['target']
        if source not in by_id or target not in by_id:
            raise WorkerFailure('Placed TDAG references an absent node', 'placement_failed')
        if target not in following[source]:
            following[source].add(target)
            previous[target].add(source)
            indegree[target] += 1
    queue, visited = deque(name for name in by_id if indegree[name] == 0), 0
    while queue:
        current = queue.popleft()
        visited += 1
        for target in following[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(nodes):
        raise WorkerFailure('Placed TDAG contains a cycle', 'placement_failed')
    reachable, pending = set(), [inputs[0]]
    while pending:
        current = pending.pop()
        if current not in reachable:
            reachable.add(current)
            pending.extend(following[current])
    if outputs[0] not in reachable:
        raise WorkerFailure('Placed output is unreachable from its input', 'placement_failed')
    bootstraps = []
    for node in nodes:
        if node.get('level') is None or node.get('scale') is None:
            raise WorkerFailure('Placed TDAG must retain assigned level and scale', 'placement_failed')
        if node.get('op') != 'bootstrap':
            continue
        weight = node.get('weight')
        if type(weight) is not int or weight < 0:
            raise WorkerFailure('Bootstrap weight must be a nonnegative integer', 'placement_failed')
        bootstraps.append(dict(id=node['id'], predecessors=sorted(previous[node['id']]),
                               successors=sorted(following[node['id']]),
                               targetLevel=node.get('op_descr', {}).get('targetLevel'),
                               level=node['level'], scale=node['scale'], weight=weight))
    return dict(passed=True, node_count=len(nodes), edge_count=len(edges),
                acyclic=True, input_reaches_output=True, level_scale_assigned=True,
                op_counts=dict(Counter(node.get('op') for node in nodes)),
                inputs=inputs, outputs=outputs, bootstraps=bootstraps,
                placement_bootstrap_nodes=len(bootstraps),
                placement_bootstrap_weight_sum=sum(node['weight'] for node in bootstraps))


def validate_model(args, recorder, result) -> None:
    root, work, profile, input_file, output, _ = _selected_paths(args)
    result['source_record'] = str(Path(args.source_record).resolve())
    with _phase(args, recorder, result, 'validate.source_records'):
        result['source_summary'] = _validate_source(args)
        result.update(source_checked=True, native_optimizer_completed=True,
                      input_mlir=_file_record(input_file), output_mlir=_file_record(output))
    with _phase(args, recorder, result, 'validate.import_modules'):
        Params, build_from_mlir, check_tdag = _load_validation(root)
    with _phase(args, recorder, result, 'validate.params'):
        result['params_calls'] += 1
        _checkpoint(args, result)
        params, metadata = make_params(profile, Params, mode='execute')
        result.update(params=metadata, profile_metadata=metadata, params_checked=True)
        result['params_completed'] += 1
    with _phase(args, recorder, result, 'validate.build_from_mlir'):
        result['validate_attempts'] += 1
        result['build_from_mlir_calls'] += 1
        _checkpoint(args, result)
        graph = build_from_mlir(str(output), params)
        result['build_from_mlir_completed'] += 1
    with _phase(args, recorder, result, 'validate.save_graph'):
        from .r14_orbit_frontend import graph_record
        record = graph_record(graph)
        destination = work / 'tdag.json'
        write_json(destination, record)
        result['tdag'] = str(destination)
    with _phase(args, recorder, result, 'validate.check_tdag'):
        result['check_tdag_calls'] += 1
        _checkpoint(args, result)
        check_tdag(graph)
        result['check_tdag_completed'] += 1
    with _phase(args, recorder, result, 'validate.placement_record'):
        placement = _placement_record(record)
        destination = work / 'placement.json'
        write_json(destination, placement)
        result.update(placement=str(destination), graph_check=placement,
                      placement_bootstrap_nodes=placement['placement_bootstrap_nodes'],
                      placement_bootstrap_weight_sum=placement['placement_bootstrap_weight_sum'],
                      placement_artifact_checked=True)
        result['validate_completed'] += 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('optimize', 'validate'))
    for name in ('work-dir', 'orbit-root', 'profile', 'input-mlir', 'output-mlir',
                 'license-file', 'source-record'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if any((args.work_dir / filename).exists() for filename in ('phase_result.json', 'events.jsonl')):
        raise ValueError('R23 preserves existing inner phase records')
    resource.setrlimit(resource.RLIMIT_AS, (4 * GIB, 4 * GIB))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * GIB, 8 * GIB))
    result = _initial_result(args.phase)
    result['source_record'] = str(args.source_record.resolve())
    recorder = StageRecorder(args.work_dir / 'events.jsonl')
    started = time.perf_counter_ns()
    write_json(args.work_dir / 'phase_result.json', result)
    try:
        with recorder.span('r23.' + args.phase):
            with _worker_environment(args, result):
                (args.work_dir / 'worker-source.py').write_text(Path(__file__).read_text())
                with _budget_guard(args, result,
                                   OPTIMIZE_SECONDS if args.phase == 'optimize' else VALIDATE_SECONDS):
                    (optimize_model if args.phase == 'optimize' else validate_model)(args, recorder, result)
        result.update(status='ok', failed_phase=None)
    except WorkerFailure as exc:
        result.update(status=exc.status, error=f'{type(exc).__name__}: {exc}'[:2000])
        traceback.print_exc()
    except (ImportError, FileNotFoundError) as exc:
        result.update(status='dependency_missing', error=f'{type(exc).__name__}: {exc}'[:2000])
        traceback.print_exc()
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}'[:2000])
        traceback.print_exc()
    finally:
        result.update(elapsed_ns=time.perf_counter_ns() - started,
                      resource_ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      elapsed_scope='worker entry through phase context restoration; before terminal write')
        try:
            _checkpoint(args, result)
        except BaseException as exc:
            result['terminal_budget_error'] = f'{type(exc).__name__}: {exc}'[:2000]
            if result['status'] == 'ok':
                result.update(status=getattr(exc, 'status', 'error'),
                              error=result['terminal_budget_error'])
            write_json(args.work_dir / 'phase_result.json', result)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
