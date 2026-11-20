"""Generic original Orbit CNN frontend; no optimizer or CKKS execution.

Derived from the own R22 frontend worker. Original benchmark function bodies, Poly,
weights and model definitions are used unchanged; only explicit dispatch and
DataParallel wrapping for the original plain-module R16 adapters are added.
"""
import argparse
import ast
from collections import Counter, deque
from contextlib import contextmanager
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import traceback
from types import ModuleType

from .r10_paper_metrics import StageRecorder, write_json


MODEL = 'pt_agent.experiments.r13_common_models:resnet20_silu'
PROFILE = Path('cost_models/profiled_LATTIGONEW_CPU64k_3_16.json')
BENCHMARK_NAME = 'ResNetSiLU64k'
EXPECTED_ACTIVATIONS = 19
MODEL_RECORDS = {
    'pt_agent.experiments.r13_common_models:resnet20_silu': dict(
        benchmark='ResNetSiLU64k', source='ResNetSiLU.py', activations=19, checkpoint_envelope=True,
        mpcb=('CascadeConv','CascadeDS','CascadePool','shapeClosure'),
        functions=('HE_ConvBN','HE_DS','HE_Pool','HE_Linear')),
    'pt_agent.experiments.r13_common_models:vgg16_silu_avgpool': dict(
        benchmark='VGG16SiLU64k', source='VGG16SiLU.py', activations=15, checkpoint_envelope=False,
        mpcb=('CascadeConv','CascadeMax','shapeClosure'),
        functions=('HE_ConvBN','HE_Avg','HE_Linear','HE_MPBN')),
    'pt_agent.experiments.r13_common_models:alexnet_silu_avgpool': dict(
        benchmark='AlexNetSiLU64k', source='AlexNetSiLU.py', activations=7, checkpoint_envelope=False,
        mpcb=('CascadeConv','CascadeMax','shapeClosure'),
        functions=('HE_ConvBN','HE_Avg','HE_ReshapeLinear','HE_Linear')),
}


def configure_model(model):
    global MODEL, BENCHMARK_NAME, EXPECTED_ACTIVATIONS
    if model not in MODEL_RECORDS:
        raise ValueError('Unknown original common-model factory')
    record=MODEL_RECORDS[model]
    MODEL=model;BENCHMARK_NAME=record['benchmark'];EXPECTED_ACTIVATIONS=record['activations']
    return dict(record)

TRACE_SECONDS = 1800
PARSE_SECONDS = 300
GIB = 1024 ** 3
ZERO_FIELDS = ('llm_calls', 'optimizer_runs', 'solver_runs', 'compiler_runs',
               'keygen_runs', 'real_ckks_attempts', 'real_ckks_completed',
               'model_forward_runs', 'sample_reads', 'downloads', 'installations',
               'confirm_samples', 'report_samples', 'freeze_attempts', 'heir_calls')
UNMEASURED_FIELDS = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency',
                     'accuracy_acceptance')
COUNTER_FIELDS = ('checkpoint_load_attempts', 'checkpoint_load_completed',
                  'model_build_attempts', 'model_build_completed',
                  'expr_load_attempts', 'expr_load_completed',
                  'function_registration_attempts', 'function_registrations',
                  'save_attempts', 'save_completed',
                  'finalize_attempts', 'finalize_completed', 'get_model_calls',
                  'activation_calls', 'params_calls', 'params_completed',
                  'build_from_mlir_calls', 'build_from_mlir_completed',
                  'trace_runs', 'parse_runs')
DATA_FILES = ('treeStr.txt', 'coeffStr.txt', 'sgn151527.txt', 'tree15.txt', 'tree27.txt')
_THREADS_INITIALIZED = False


class WorkerFailure(RuntimeError):
    def __init__(self, message, status='error'):
        super().__init__(message)
        self.status = status


def _initial_result(phase):
    result = dict(round='ORBIT-COMMON-FRONTEND', model=MODEL, phase=phase, status='running', error=None,
                  failed_phase=None, activation_names=[], source_checked=False,
                  orbit_model_trace_passed=False, tdag_parse_passed=False,
                  optimizer_ready=False, backend_ready=False, minimal_ckks_passed=False,
                  formal_matrix_complete=False, environment_restored=False,
                  phase_artifact_bytes=0)
    result.update(dict.fromkeys(ZERO_FIELDS, 0))
    result.update(dict.fromkeys(UNMEASURED_FIELDS, None))
    result.update(dict.fromkeys(COUNTER_FIELDS, 0))
    return result


def _initialize_environment():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def _numeric_modules():
    global _THREADS_INITIALIZED
    _initialize_environment()
    import numpy as np
    import torch
    if not _THREADS_INITIALIZED:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _THREADS_INITIALIZED = True
    return np, torch


def _phase_size(work):
    amount = 0
    for parent, directories, files in os.walk(work):
        for name in files:
            try:
                amount += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                pass
    return amount


def _checkpoint(args, result):
    work = Path(args.work_dir)
    result['phase_artifact_bytes'] = _phase_size(work)
    if result['phase_artifact_bytes'] > args.case_gib * GIB:
        raise WorkerFailure('Original Orbit frontend phase artifacts exceed the explicit case allowance', 'artifact_budget_exceeded')
    if shutil.disk_usage(work).free < 10 * GIB:
        raise WorkerFailure('Original Orbit common frontend disk reserve is below 10 GiB', 'disk_reserve')
    write_json(work / 'phase_result.json', result)


@contextmanager
def _phase(args, recorder, result, name):
    result['failed_phase'] = name
    _checkpoint(args, result)
    with recorder.span(name):
        yield
    _checkpoint(args, result)


@contextmanager
def _wall_limit(seconds):
    def expired(signum, frame):
        raise WorkerFailure('Original Orbit common frontend inner phase time allowance exhausted', 'timeout')
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)


def _read_record(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise WorkerFailure('Source record must be a JSON object', 'source_failed')
    return value


def _same_path(left, right):
    return isinstance(left, (str, Path)) and Path(left).resolve() == Path(right).resolve()


def _check_sources(args):
    selected=configure_model(args.model)
    original_root=Path(args.orbit_root).resolve()/'frontend/dacapo'
    if (Path(args.model_source).resolve()!=original_root/'python/poly/poly/models'/selected['source']
            or Path(args.benchmark).resolve()!=Path(args.orbit_root).resolve()/'frontend/dacapo_patch/benchmarks'/(selected['benchmark']+'.py')):
        raise WorkerFailure('Only the exact official model definition and benchmark are accepted','source_failed')
    r13 = _read_record(args.r13_result)
    r16 = _read_record(args.r16_result)
    first, second = r13.get('worker') or {}, r16.get('worker') or {}
    if (r13.get('round') != 'R13' or r13.get('status') != 'ok'
            or r13.get('model') != MODEL or first.get('model') != MODEL
            or not all(first.get(key) is True for key in
                       ('import_completed', 'real_probes_passed', 'graph_math_passed'))):
        raise WorkerFailure('Original Orbit frontend requires the saved successful R13 model', 'source_failed')
    if (r16.get('round') != 'R16' or r16.get('status') != 'ok'
            or r16.get('model') != MODEL or r16.get('baseline') != 'orbit'
            or second.get('model') != MODEL or second.get('baseline') != 'orbit'
            or not all(second.get(key) is True for key in
                       ('compatibility_passed', 'source_checked', 'strict_load_passed',
                        'structure_passed', 'math_probes_passed'))
            or second.get('forward_attempts') != 3 or second.get('forward_completed') != 3):
        raise WorkerFailure('Original Orbit common frontend requires the saved three R16 Orbit reference comparisons', 'source_failed')
    original = first.get('model_source') or {}
    summary = second.get('source_summary') or {}
    report = second.get('state_report') or {}
    if (not _same_path(original.get('checkpoint'), args.checkpoint)
            or not _same_path(summary.get('checkpoint'), args.checkpoint)
            or not _same_path(summary.get('model_source'), args.model_source)
            or not _same_path(summary.get('r13_result'), args.r13_result)
            or report.get('passed') is not True or report.get('strict_load') is not True
            or (report.get('structure') or {}).get('passed') is not True
            or report.get('extra_state') != [] or not report.get('original_state')):
        raise WorkerFailure('Original Orbit common frontend source paths or original state provenance differ', 'source_failed')
    return dict(r13_result=str(Path(args.r13_result).resolve()),
                r16_result=str(Path(args.r16_result).resolve()),
                checkpoint=str(Path(args.checkpoint).resolve()),
                model_source=str(Path(args.model_source).resolve()),
                original_state_count=len(report['original_state']), prior_reference_count=3,
                prior_reference_passed=True, references_reexecuted=False,
                prior_times_included=False)


def _load_model(args, recorder, result):
    _, torch = _numeric_modules()
    with _phase(args, recorder, result, 'trace.source_records'):
        result['source_summary'] = _check_sources(args)
        result['source_checked'] = True
    with _phase(args, recorder, result, 'trace.checkpoint_load'):
        result['checkpoint_load_attempts'] += 1
        _checkpoint(args, result)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        result['checkpoint_load_completed'] += 1
        state = (checkpoint.get('state_dict') if MODEL_RECORDS[MODEL]['checkpoint_envelope']
                 and isinstance(checkpoint,dict) else checkpoint)
        if not isinstance(state, dict) or not state:
            raise WorkerFailure('Original checkpoint needs the selected official state-dictionary format', 'state_failed')
    from . import r16_orbit_models
    with _phase(args, recorder, result, 'trace.model_build'):
        result['model_build_attempts'] += 1
        _checkpoint(args, result)
        prior = r16_orbit_models._RECORDER
        r16_orbit_models._RECORDER = recorder
        try:
            module, report = r16_orbit_models.build_model(MODEL, args.model_source, state)
        finally:
            r16_orbit_models._RECORDER = prior
        result['model_build_completed'] += 1
        result['model_report'] = report
        if (report.get('passed') is not True or report.get('strict_load') is not True
                or (report.get('structure') or {}).get('passed') is not True
                or report.get('extra_state') != []):
            raise WorkerFailure('Original Orbit strict loading or structure failed', 'state_failed')
    wrapper_added=not isinstance(module,torch.nn.DataParallel)
    if wrapper_added:
        # Original VGG/Alex benchmark bodies access model.module. The wrapper
        # retains the strict-loaded original module and every original value.
        module=torch.nn.DataParallel(module).eval()
    result['benchmark_wrapper']=dict(kind='torch.nn.DataParallel',added=wrapper_added,
        original_state_values_changed=False,reason='Unchanged official benchmark expects model.module')
    return module


def _load_expr(frontend):
    from .r14_orbit_frontend import _load_expr as load
    return load(frontend)


def _load_parser(orbit_root):
    from .r14_orbit_frontend import _load_parser as load
    return load(orbit_root)


def _load_einops(deps_dir):
    module = importlib.import_module('einops')
    source = getattr(module, '__file__', None)
    if (source is None or not Path(source).resolve().is_relative_to(Path(deps_dir).resolve())
            or getattr(module, '__version__', None) != '0.8.1'):
        raise WorkerFailure('einops must be the approved local 0.8.1 installation', 'dependency_missing')
    return module


def _load_poly(poly_dir, hc):
    poly_dir = Path(poly_dir).resolve()
    for filename in DATA_FILES:
        path = poly_dir / 'data' / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise WorkerFailure('Original polynomial data is missing: ' + filename, 'dependency_missing')
    package = ModuleType('poly')
    package.__path__ = [str(poly_dir)]
    package.__package__ = 'poly'
    sys.modules['poly'] = package
    sys.modules['hecate'] = hc
    modules = []
    for name in ('MPCB', 'Poly', 'Func'):
        source = poly_dir / (name + '.py')
        spec = importlib.util.spec_from_file_location('poly.' + name, source)
        if spec is None or spec.loader is None:
            raise ImportError('Cannot load the approved poly file: ' + str(source))
        module = importlib.util.module_from_spec(spec)
        sys.modules['poly.' + name] = module
        setattr(package, name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def load_benchmark_function(source, namespace) -> object:
    """Register the one original function; execute no other top-level nodes."""
    source = Path(source)
    parsed = ast.parse(source.read_text(), filename=str(source))
    found = [node for node in parsed.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == BENCHMARK_NAME]
    if len(found) != 1 or not isinstance(found[0], ast.FunctionDef):
        raise ValueError('Expected exactly one synchronous selected official benchmark function')
    node = found[0]
    if len(node.decorator_list) != 1:
        raise ValueError('Expected only the original hc.func decorator')
    decorator = node.decorator_list[0]
    if (not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute)
            or not isinstance(decorator.func.value, ast.Name)
            or decorator.func.value.id != 'hc' or decorator.func.attr != 'func'
            or len(decorator.args) != 1 or decorator.keywords
            or not isinstance(decorator.args[0], ast.Constant)
            or decorator.args[0].value != 'c'):
        raise ValueError('Expected the original hc.func("c") decorator')
    selected = ast.Module(body=[node], type_ignores=[])
    exec(compile(selected, str(source), 'exec'), namespace)
    return namespace[BENCHMARK_NAME]


def trace_model(args, recorder, result) -> None:
    selected=configure_model(args.model)
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    result['trace_runs'] += 1
    np, torch = _numeric_modules()
    model = _load_model(args, recorder, result)
    trace_dir, constants = work / 'trace', work / 'constants'
    trace_dir.mkdir()
    constants.mkdir()
    original_path, original_modules, original_cwd = list(sys.path), dict(sys.modules), Path.cwd()
    saved_environment = {name: os.environ.get(name) for name in ('HECATE', 'PATH')}
    hc = None
    primary_error = None
    try:
        os.chdir(work)
        os.environ['HECATE'] = str(Path(args.frontend).resolve())
        sys.path.insert(0, str(Path(args.deps_dir).resolve()))
        for name in tuple(sys.modules):
            if name.split('.')[0] in ('hecate', 'poly', 'einops'):
                del sys.modules[name]
        with _phase(args, recorder, result, 'trace.load_einops'):
            einops = _load_einops(args.deps_dir)
            result['einops'] = dict(version=getattr(einops, '__version__', None),
                                    source=getattr(einops, '__file__', None))
        with _phase(args, recorder, result, 'trace.load_expr'):
            result['expr_load_attempts'] += 1
            _checkpoint(args, result)
            hc = _load_expr(args.frontend)
            result['expr_load_completed'] += 1
            if not getattr(hc, 'ctxt', None):
                raise WorkerFailure('Official Orbit expression context was not initialized')
            sys.modules['hecate'] = hc
        with _phase(args, recorder, result, 'trace.load_poly'):
            mpcb, poly, functions = _load_poly(args.poly_dir, hc)
            result['poly_sources'] = [str(Path(args.poly_dir).resolve() / name)
                                      for name in ('MPCB.py', 'Poly.py', 'Func.py')]
        def get_model():
            result['get_model_calls'] += 1
            if result['get_model_calls'] != 1:
                raise WorkerFailure('The official trace requested its model more than once')
            return model
        original_silu = functions.HE_SiLU
        def silu(name, value):
            result['activation_calls'] += 1
            result['activation_names'].append(name)
            if result['activation_calls'] > EXPECTED_ACTIVATIONS:
                raise WorkerFailure('The original selected trace exceeded its exact activation count')
            return original_silu(name, value)
        namespace = dict(np=np, torch=torch, hc=hc, getModel=get_model)
        for name in selected['mpcb']:
            namespace[name] = getattr(mpcb, name)
        for name in selected['functions']:
            namespace[name] = getattr(functions, name)
        namespace['HE_SiLU'] = silu
        with _phase(args, recorder, result, 'trace.register_model'):
            result['function_registration_attempts'] += 1
            _checkpoint(args, result)
            load_benchmark_function(args.benchmark, namespace)
            result['function_registrations'] += 1
        with _phase(args, recorder, result, 'trace.save'):
            result['save_attempts'] += 1
            _checkpoint(args, result)
            returned = hc.save(dirs=str(trace_dir), cst_dirs=str(constants))
            result['save_completed'] += 1
            result['save_returned_path'] = str(returned)
            mlir = Path(returned).resolve()
            if (not mlir.is_relative_to(trace_dir) or mlir.suffix != '.mlir'
                    or not mlir.is_file() or mlir.stat().st_size == 0):
                raise WorkerFailure('The official save returned no nonempty trace-local MLIR', 'artifact_invalid')
            if result['get_model_calls'] != 1 or result['activation_calls'] != EXPECTED_ACTIVATIONS:
                raise WorkerFailure('The complete original selected model was not traced', 'coverage_failed')
            result['mlir'] = dict(path=str(mlir), size_bytes=mlir.stat().st_size)
            result['constant_files'] = [dict(path=str(path.resolve()), size_bytes=path.stat().st_size)
                                        for path in sorted(constants.rglob('*')) if path.is_file()]
            result['constant_file_count'] = len(result['constant_files'])
            result['constant_bytes'] = sum(item['size_bytes'] for item in result['constant_files'])
            result['benchmark_function'] = BENCHMARK_NAME
            result['orbit_model_trace_passed'] = True
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            if hc is None:
                candidate = sys.modules.get('orbit_expr_check')
                if candidate is not original_modules.get('orbit_expr_check'):
                    hc = candidate
            if hc is not None and getattr(hc, 'ctxt', None):
                try:
                    if primary_error is None:
                        result['failed_phase'] = 'trace.finalize'
                    # Resource failures must still release the live native context.
                    with recorder.span('trace.finalize'):
                        result['finalize_attempts'] += 1
                        hc.lt.finalize(hc.ctxt)
                        result['finalize_completed'] += 1
                except BaseException as exc:
                    result['cleanup_error'] = str(type(exc).__name__) + ': ' + str(exc)
                    result['orbit_model_trace_passed'] = False
                    if primary_error is None:
                        raise
        finally:
            for name in tuple(sys.modules):
                if name not in original_modules:
                    del sys.modules[name]
            sys.modules.update(original_modules)
            sys.path[:] = original_path
            os.chdir(original_cwd)
            for name, value in saved_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            result['environment_restored'] = True


def _profile_params(profile, Params):
    declaration = _read_record(profile)
    degree = declaration.get('polynomialDegree')
    if type(degree) is not int or degree != 131072:
        raise ValueError('Original Orbit common frontend requires the original 64k polynomialDegree=131072')
    params = Params(str(profile), 'Orbit', 'compile', threads=1)
    metadata = dict(profile=str(Path(profile).resolve()), profile_use='parser_metadata_only',
                    declared_field='polynomialDegree', declared_polynomial_degree=degree,
                    original_poly_deg=params.poly_deg, original_max_slot=params.max_slot,
                    mode=params.mode, sysname=params.sysname, threads=params.threads)
    params.poly_deg = degree
    params.max_slot = degree // 2
    if params.max_slot != 65536:
        raise ValueError('The original 64k slot metadata differs')
    metadata.update(poly_deg=params.poly_deg, max_slot=params.max_slot,
                    alias_adapted_in_memory=True, security_validated=False)
    return params, metadata


def _check_graph(record):
    nodes, edges = record['nodes'], record['edges']
    by_id = {node['id']: node for node in nodes}
    inputs, outputs = record['inputs'], record['outputs']
    if (len(by_id) != len(nodes) or len(inputs) != 1 or len(outputs) != 1
            or inputs[0] not in by_id or outputs[0] not in by_id):
        raise ValueError('The original complete model needs unique nodes and one input/output')
    if by_id[inputs[0]].get('op') != 'input':
        raise ValueError('The model input must have the official input operation')
    following = {name: set() for name in by_id}
    indegree = dict.fromkeys(by_id, 0)
    for edge in edges:
        source, target = edge['source'], edge['target']
        if source not in by_id or target not in by_id:
            raise ValueError('TDAG edge references a missing node')
        if target not in following[source]:
            following[source].add(target)
            indegree[target] += 1
    queue = deque(name for name in by_id if indegree[name] == 0)
    visited = 0
    while queue:
        name = queue.popleft()
        visited += 1
        for target in following[name]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(nodes):
        raise ValueError('The official TDAG contains a cycle')
    reachable, pending = set(), [inputs[0]]
    while pending:
        name = pending.pop()
        if name not in reachable:
            reachable.add(name)
            pending.extend(following[name])
    if outputs[0] not in reachable:
        raise ValueError('The model output is unreachable from its input')
    unassigned = all(node.get('level') is None and node.get('scale') is None for node in nodes)
    if not unassigned:
        raise ValueError('Official compile-mode TDAG levels/scales must remain unassigned')
    return dict(passed=True, node_count=len(nodes), edge_count=len(edges), acyclic=True,
                input_reaches_output=True, compile_level_scale_unassigned=unassigned,
                op_counts=dict(Counter(node.get('op') for node in nodes)),
                rotations=[dict(id=node['id'], **node.get('op_descr', {}))
                           for node in nodes if node.get('op') == 'rotate'])


def parse_model(args, recorder, result) -> None:
    configure_model(args.model)
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    _initialize_environment()
    result['parse_runs'] += 1
    root, profile, mlir = Path(args.orbit_root).resolve(), Path(args.profile).resolve(), Path(args.mlir).resolve()
    if profile != root / PROFILE:
        raise ValueError('Original Orbit common frontend uses only the original 64k CPU profile')
    if not mlir.is_file() or not mlir.stat().st_size:
        raise ValueError('Original Orbit common frontend requires the single original model trace output')
    original_path, original_modules = list(sys.path), dict(sys.modules)
    try:
        with _phase(args, recorder, result, 'parse.import_modules'):
            Params, build_from_mlir = _load_parser(root)
        with _phase(args, recorder, result, 'parse.params'):
            result['params_calls'] += 1
            _checkpoint(args, result)
            params, result['parser_metadata'] = _profile_params(profile, Params)
            result['params_completed'] += 1
        with _phase(args, recorder, result, 'parse.build_from_mlir'):
            result['build_from_mlir_calls'] += 1
            _checkpoint(args, result)
            graph = build_from_mlir(str(mlir), params)
            result['build_from_mlir_completed'] += 1
        with _phase(args, recorder, result, 'parse.save_graph'):
            from .r14_orbit_frontend import graph_record
            record = graph_record(graph)
            path = work / 'tdag.json'
            write_json(path, record)
            result.update(tdag=str(path), graph=record)
        with _phase(args, recorder, result, 'parse.check_graph'):
            result['graph_check'] = _check_graph(record)
            result['tdag_parse_passed'] = True
    finally:
        for name in tuple(sys.modules):
            if name not in original_modules:
                del sys.modules[name]
        sys.modules.update(original_modules)
        sys.path[:] = original_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('trace', 'parse'))
    parser.add_argument('--model',required=True,choices=tuple(MODEL_RECORDS))
    parser.add_argument('--memory-gib',type=float,default=4)
    parser.add_argument('--case-gib',type=float,default=4)
    parser.add_argument('--phase-seconds', type=float)
    parser.add_argument('--work-dir', type=Path, required=True)
    for name in ('frontend', 'model-source', 'benchmark', 'checkpoint', 'poly-dir',
                 'deps-dir', 'orbit-root', 'profile', 'mlir', 'r13-result', 'r16-result'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    configure_model(args.model)
    phase_cap = TRACE_SECONDS if args.phase == 'trace' else PARSE_SECONDS
    args.phase_seconds = phase_cap if args.phase_seconds is None else args.phase_seconds
    if not 0 < args.phase_seconds <= phase_cap:
        parser.error('Phase seconds must be positive and within the selected phase allowance')
    if not 0<args.memory_gib<=12 or not 0<args.case_gib<=32:
        parser.error('Frontend preparation is limited to12GiB memory and32GiB case artifacts')
    required = (('frontend', 'model_source', 'benchmark', 'checkpoint', 'poly_dir',
                 'deps_dir', 'orbit_root', 'r13_result', 'r16_result') if args.phase == 'trace'
                else ('orbit_root', 'profile', 'mlir'))
    if any(getattr(args, name) is None for name in required):
        parser.error('The selected phase requires explicit source and artifact paths')
    _initialize_environment()
    resource.setrlimit(resource.RLIMIT_AS, (int(args.memory_gib * GIB), int(args.memory_gib * GIB)))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * GIB, 8 * GIB))
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if (args.work_dir / 'phase_result.json').exists() or (args.work_dir / 'events.jsonl').exists():
        raise ValueError('Original Orbit common frontend preserves existing inner phase records')
    result = _initial_result(args.phase)
    recorder = StageRecorder(args.work_dir / 'events.jsonl')
    started = time.perf_counter_ns()
    write_json(args.work_dir / 'phase_result.json', result)
    try:
        with recorder.span('orbit_common.' + args.phase):
            with _wall_limit(args.phase_seconds):
                (trace_model if args.phase == 'trace' else parse_model)(args, recorder, result)
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
                      resource_ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        try:
            _checkpoint(args, result)
        except BaseException as exc:
            result['terminal_budget_error'] = f'{type(exc).__name__}: {exc}'[:2000]
            if result['status'] == 'ok':
                result.update(status=getattr(exc, 'status', 'error'),
                              error=result['terminal_budget_error'])
            result['orbit_model_trace_passed'] = False
            result['tdag_parse_passed'] = False
            write_json(args.work_dir / 'phase_result.json', result)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
