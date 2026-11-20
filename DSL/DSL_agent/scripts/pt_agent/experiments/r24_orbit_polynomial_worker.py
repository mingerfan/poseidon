"""R24: original Orbit SiLU arithmetic on three saved training probes."""

import argparse
import ast
import copy
from contextlib import contextmanager
import json
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


ROUND = 'R24'
MODEL = 'pt_agent.experiments.r13_common_models:resnet20_silu'
MODE = 'orbit-native-polynomial-physical-scale-check'
GIB = 1024 ** 3
CHECK_SECONDS = 300
EXPECTED_STATE_COUNT = 116
EXPECTED_SILU_MODULES = 10
EXPECTED_ACTIVATIONS = 19
CRITERION = '1e-12+1e-12*abs(reference)'
COUNTER_FIELDS = (
    'native_load_attempts', 'native_load_completed', 'checkpoint_load_attempts',
    'checkpoint_load_completed', 'probe_load_attempts', 'probe_load_completed',
    'model_build_attempts', 'model_build_completed', 'deepcopy_attempts', 'deepcopy_completed',
    'original_forward_attempts', 'original_forward_completed',
    'approx_forward_attempts', 'approx_forward_completed',
    'activation_attempts', 'activation_completed', 'math_check_attempts', 'math_check_completed',
    'original_reference_checks', 'silu_modules_replaced',
)
COUNT_FIELDS = COUNTER_FIELDS
ZERO_FIELDS = (
    'llm_calls', 'downloads', 'installations', 'new_weights', 'new_datasets',
    'fit_calls', 'agent_calls', 'tuner_calls', 'count_runs', 'emit_runs', 'trace_runs',
    'compiler_runs', 'optimizer_runs', 'solver_runs', 'keygen_runs',
    'real_ckks_attempts', 'real_ckks_completed', 'heir_calls',
    'heldout_samples', 'confirm_samples', 'report_samples', 'freeze_attempts',
    'raw_cifar_reads',
)
UNMEASURED_FIELDS = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency',
                     'accuracy_acceptance')
STAGE_FIELDS = ('source_checked', 'original_reference_passed',
                'native_polynomial_math_passed', 'physical_scaled_forward_completed')
_TORCH_INITIALIZED = False


class WorkerFailure(RuntimeError):
    def __init__(self, message, status='error'):
        super().__init__(message)
        self.status = status


def _initial_result():
    result = dict(round=ROUND, model=MODEL, mode=MODE, status='running', error=None,
                  failed_phase=None, original_probes=[], approximate_probes=[],
                  activation_rows=[], module_stats={}, native_metadata=None,
                  native_packed_equivalence_verified=False, backend_ready=False,
                  security_validated=False, accuracy_validated=False,
                  formal_matrix_complete=False, phase_artifact_bytes=0,
                  case_artifact_bytes=0, budget_monitor_completed=False)
    result.update(dict.fromkeys(STAGE_FIELDS, False))
    result.update(dict.fromkeys(COUNTER_FIELDS, 0))
    result.update(dict.fromkeys(ZERO_FIELDS, 0))
    result.update(dict.fromkeys(UNMEASURED_FIELDS, None))
    return result


def _numeric_modules():
    import numpy as np
    import torch
    return np, torch


def _initialize_torch():
    global _TORCH_INITIALIZED
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    np, torch = _numeric_modules()
    if not _TORCH_INITIALIZED:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _TORCH_INITIALIZED = True
    if np.__version__ != '1.25.2' or torch.__version__.split('+')[0] != '2.0.1':
        raise WorkerFailure('R24 requires NumPy 1.25.2 and Torch 2.0.1', 'dependency_missing')
    return dict(numpy=np.__version__, torch=torch.__version__, dtype='float64',
                torch_intra_threads=1, torch_interop_threads=1)


def _read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise WorkerFailure('An explicit source record must be a JSON object', 'source_failed')
    return value


def _same_path(left, right):
    return isinstance(left, (str, Path)) and Path(left).resolve() == Path(right).resolve()


def _file_record(path):
    path = Path(path).resolve()
    if not path.is_file() or not path.stat().st_size:
        raise WorkerFailure('Required source file is missing: ' + str(path), 'source_failed')
    return dict(path=str(path), size_bytes=path.stat().st_size)


def _size(path):
    total = 0
    for parent, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def _budget_observation(args, started=None):
    work = Path(args.work_dir).resolve()
    return dict(ts_ns=time.perf_counter_ns(), case_root=str(work.parent.parent),
                case_artifact_bytes=_size(work.parent.parent),
                disk_free_bytes=shutil.disk_usage(work).free,
                elapsed_ns=None if started is None else time.perf_counter_ns() - started)


def _budget_error(row):
    if row['case_artifact_bytes'] > 4 * GIB:
        return 'artifact_budget_exceeded', 'R24 case artifacts exceed 4 GiB'
    if row['disk_free_bytes'] < 10 * GIB:
        return 'disk_reserve', 'R24 disk reserve is below 10 GiB'
    if row['elapsed_ns'] is not None and row['elapsed_ns'] >= CHECK_SECONDS * 1_000_000_000:
        return 'timeout', 'R24 inner check allowance exhausted'
    return None


def _checkpoint(args, result):
    work = Path(args.work_dir).resolve()
    observation = _budget_observation(args)
    result.update(phase_artifact_bytes=_size(work),
                  case_artifact_bytes=observation['case_artifact_bytes'],
                  last_budget_observation=observation)
    write_json(work / 'phase_result.json', result)
    failure = _budget_error(observation)
    if failure:
        raise WorkerFailure(failure[1], failure[0])


@contextmanager
def _phase(args, recorder, result, name, **metadata):
    result['failed_phase'] = name
    _checkpoint(args, result)
    with recorder.span(name, **metadata):
        yield
    _checkpoint(args, result)


def _terminate_self():
    os.kill(os.getpid(), signal.SIGTERM)


@contextmanager
def _budget_guard(args, result):
    work = Path(args.work_dir).resolve()
    started, stop = time.perf_counter_ns(), threading.Event()

    def watch():
        while not stop.wait(1):
            try:
                observation = _budget_observation(args, started)
                with (work / 'budget-samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(observation) + '\n')
                    stream.flush()
                failure = _budget_error(observation)
                if failure is None:
                    continue
                record = dict(status=failure[0], error=failure[1], **observation)
            except BaseException as exc:
                record = dict(status='record_error', error=f'{type(exc).__name__}: {exc}'[:2000],
                              ts_ns=time.perf_counter_ns())
            try:
                write_json(work / 'budget-fatal.json',
                           dict(record, termination_origin='worker_budget_monitor'))
            finally:
                _terminate_self()
            return

    thread = threading.Thread(target=watch, name='r24-budget-observer', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)
        result['budget_monitor_completed'] = not thread.is_alive()
        if thread.is_alive():
            raise WorkerFailure('R24 metadata observer did not stop', 'record_error')


def _selected_functions(source, names, namespace):
    path = Path(source)
    parsed = ast.parse(path.read_text(), filename=str(path))
    selected = []
    for name in names:
        matches = [node for node in parsed.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name]
        if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
            raise WorkerFailure('Missing, duplicate or async native function: ' + name, 'source_failed')
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.decorator_list
               for node in ast.walk(matches[0])):
            raise WorkerFailure('Native polynomial functions must have no decorators', 'source_failed')
        selected.append(matches[0])
    selected.sort(key=lambda node: node.lineno)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return [dict(name=node.name, lineno=node.lineno, end_lineno=node.end_lineno,
                 complete_body_preserved=True, defaults_preserved=True) for node in selected]


def load_native_functions(poly_dir) -> tuple:
    import numpy as np

    directory = Path(poly_dir).resolve()
    tree_path, coefficient_path = directory / 'data/treeStr.txt', directory / 'data/coeffStr.txt'
    with tree_path.open() as stream:
        tree_lines = stream.readlines()
    with coefficient_path.open() as stream:
        coefficient_lines = stream.readlines()
    coefficients = np.asarray([float(line.strip()) for line in coefficient_lines], dtype=np.float64)
    if coefficients.shape != (96,) or not np.isfinite(coefficients).all():
        raise WorkerFailure('R24 requires the original 96 finite float64 coefficients', 'source_failed')
    tree = [[int(token) for token in line.strip().split(' ')] for line in tree_lines]
    if not tree or not tree[0]:
        raise WorkerFailure('The original polynomial tree is empty', 'source_failed')
    metadata = dict(
        sources=[_file_record(directory / name) for name in
                 ('MPCB.py', 'Poly.py', 'Func.py', 'data/treeStr.txt', 'data/coeffStr.txt')],
        tree=tree, tree_root=tree[0][0], coefficient_count=int(coefficients.size),
        coefficient_dtype=str(coefficients.dtype), c0=float(coefficients[0]),
        c0_independently_added=False, even_coefficients_ignored_in_reference=True,
        leaf_length=16, poly_closure_attempts=0, poly_closure_completed=0,
        closure_lengths=[], closure_scales=[], ast_selection={},
        fitted_target_known=False, approximation_domain_known=False,
        source_combination_equivalence_verified=False)
    odd_nonzero = [index for index in range(1, len(coefficients), 2) if coefficients[index] != 0]
    degree = max(odd_nonzero, default=0)
    metadata.update(odd_chebyshev_degree=degree, full_g_degree=degree + 1 if odd_nonzero else 1,
                    degree_basis='nonzero odd original Chebyshev coefficients; HE_SiLU multiplies by u')
    mpcb_namespace = {'np': np}
    metadata['ast_selection']['MPCB.py'] = _selected_functions(
        directory / 'MPCB.py', ('fint', 'GenPoly'), mpcb_namespace)
    original_genpoly = mpcb_namespace['GenPoly']

    def counted_genpoly(*args, **kwargs):
        metadata['poly_closure_attempts'] += 1
        length = kwargs.get('length', args[2] if len(args) > 2 else None)
        scale = kwargs.get('scale', args[3] if len(args) > 3 else 1.0)
        metadata['closure_lengths'].append(length)
        metadata['closure_scales'].append(scale)
        answer = original_genpoly(*args, **kwargs)
        metadata['poly_closure_completed'] += 1
        return answer

    mpcb = SimpleNamespace(fint=mpcb_namespace['fint'], GenPoly=counted_genpoly)
    poly_namespace = dict(np=np, MPCB=mpcb, tree_var=tree_lines, coeff_var=coefficient_lines)
    metadata['ast_selection']['Poly.py'] = _selected_functions(
        directory / 'Poly.py', ('GenPoly',), poly_namespace)
    poly = SimpleNamespace(GenPoly=poly_namespace['GenPoly'])
    func_namespace = dict(np=np, MPCB=mpcb, Poly=poly,
                          hc=SimpleNamespace(add_comment=lambda *args, **kwargs: None))
    metadata['ast_selection']['Func.py'] = _selected_functions(
        directory / 'Func.py', ('HE_SiLU',), func_namespace)
    return func_namespace['HE_SiLU'], coefficients, metadata


def odd_chebyshev_reference(coefficients, values) -> object:
    import numpy as np

    if (not isinstance(coefficients, np.ndarray) or coefficients.ndim != 1
            or coefficients.dtype != np.dtype('float64') or not np.isfinite(coefficients).all()
            or not isinstance(values, np.ndarray) or values.dtype != np.dtype('float64')
            or not np.isfinite(values).all()):
        raise WorkerFailure('Independent Chebyshev reference needs finite float64 arrays', 'nonfinite')
    odd = coefficients.copy()
    odd[::2] = 0
    return values * (np.polynomial.chebyshev.chebval(values, odd) + 0.5)


def _check_math(reference, observed, *, logits=False):
    import numpy as np

    shape_ok = (isinstance(reference, np.ndarray) and isinstance(observed, np.ndarray)
                and reference.shape == observed.shape and reference.size > 0
                and (not logits or reference.shape == (1, 10)))
    dtype_ok = (isinstance(reference, np.ndarray) and isinstance(observed, np.ndarray)
                and reference.dtype == np.dtype('float64') and observed.dtype == np.dtype('float64'))
    finite = bool(dtype_ok and np.isfinite(reference).all() and np.isfinite(observed).all())
    checked = dict(passed=False, shape_ok=bool(shape_ok), dtype_ok=dtype_ok, finite=finite,
                   max_absolute_error=None, max_error_ratio=None, criterion=CRITERION)
    if shape_ok and dtype_ok and finite:
        with np.errstate(over='ignore', invalid='ignore'):
            difference = np.abs(observed - reference)
            allowance = 1e-12 + 1e-12 * np.abs(reference)
            ratio = difference / allowance
        error_finite = bool(np.isfinite(difference).all() and np.isfinite(ratio).all())
        checked.update(passed=bool(error_finite and np.all(difference <= allowance)),
                       max_absolute_error=float(difference.max()) if error_finite else None,
                       max_error_ratio=float(ratio.max()) if error_finite else None)
    return checked


def _check_sources(args):
    records = {name: _read_json(getattr(args, name + '_result')) for name in ('r13', 'r16', 'r22', 'r23')}
    for name, record in records.items():
        if record.get('round') != name.upper() or record.get('model') != MODEL or record.get('status') != 'ok':
            raise WorkerFailure('R24 source identity/status differs: ' + name, 'source_failed')
    r13, r16, r22, r23 = (records[name] for name in ('r13', 'r16', 'r22', 'r23'))
    original, baseline = r13.get('worker') or {}, r16.get('worker') or {}
    if (r16.get('baseline') != 'orbit'
            or not all(baseline.get(name) is True for name in
                       ('source_checked', 'strict_load_passed', 'structure_passed', 'math_probes_passed'))
            or not all(original.get(name) is True for name in
                       ('import_completed', 'real_probes_passed', 'graph_math_passed'))
            or r22.get('orbit_model_trace_passed') is not True or r22.get('tdag_parse_passed') is not True
            or r23.get('native_optimizer_completed') is not True
            or r23.get('placement_artifact_checked') is not True):
        raise WorkerFailure('R24 requires all four saved source stages to have passed', 'source_failed')
    saved = baseline.get('source_summary') or {}
    artifacts = original.get('artifacts') or {}
    for key in ('model_source', 'checkpoint', 'probes', 'probe_source', 'r13_result'):
        if not _same_path(saved.get(key), getattr(args, key)):
            raise WorkerFailure('R24 original source path differs: ' + key, 'source_failed')
    for key, actual in (('training_probes', args.probes), ('training_probe_source', args.probe_source)):
        if not _same_path(artifacts.get(key), actual) or not Path(actual).resolve().is_relative_to(Path(args.r13_result).resolve().parent):
            raise WorkerFailure('R24 reuses only the original saved training probes', 'source_failed')
    if (Path(args.poly_dir).resolve() != Path(args.r22_result).resolve().parent / 'work/local-poly/poly'
            or not _same_path((r22.get('source_records') or {}).get('r13_result'), args.r13_result)
            or not _same_path((r22.get('source_records') or {}).get('r16_result'), args.r16_result)
            or not _same_path(((r23.get('phase_results') or {}).get('optimize') or {}).get('source_record'), args.r22_result)):
        raise WorkerFailure('R24 native polynomial provenance differs', 'source_failed')
    probes = baseline.get('math_probes')
    if not isinstance(probes, list) or len(probes) != 3:
        raise WorkerFailure('R24 requires three saved original R16 references', 'source_failed')
    paths = {name: _file_record(getattr(args, name)) for name in
             ('r13_result', 'r16_result', 'r22_result', 'r23_result', 'model_source',
              'checkpoint', 'probes', 'probe_source')}
    return dict(paths=paths, poly_dir=str(Path(args.poly_dir).resolve()),
                source_times_included=False, training_preprocessing_verified=False,
                coefficient_source_equivalence_verified=False, reference_source='R16 Orbit math_probes.reference'), r13, r16


def _load_probes(args, r13, r16, result):
    np, _ = _numeric_modules()
    source = _read_json(args.probe_source)
    if (source != (r13.get('worker') or {}).get('source') or source.get('split') != 'train'
            or source.get('indices') != [0, 1, 2]
            or source.get('sample_ids') != ['train:0', 'train:1', 'train:2']
            or source.get('normalization') != {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}):
        raise WorkerFailure('R24 training probe source/preprocessing differs', 'source_failed')
    with np.load(args.probes, allow_pickle=False) as archive:
        if set(archive.files) != {'input', 'labels', 'indices'}:
            raise WorkerFailure('Saved training NPZ fields differ', 'source_failed')
        images, labels, indices = (np.array(archive[name]) for name in ('input', 'labels', 'indices'))
    if (images.shape != (3, 3, 32, 32) or images.dtype != np.dtype('float64')
            or not np.isfinite(images).all() or labels.shape != (3,) or indices.shape != (3,)
            or labels.dtype.kind not in ('i', 'u') or indices.dtype.kind not in ('i', 'u')
            or labels.tolist() != source.get('labels') or indices.tolist() != [0, 1, 2]):
        raise WorkerFailure('Saved training probe tensors/identities differ', 'shape_failed')
    references = []
    for index, probe in enumerate(r16['worker']['math_probes']):
        if (probe.get('probe_index') != index or probe.get('sample_id') != source['sample_ids'][index]
                or probe.get('passed') is not True or probe.get('finite') is not True
                or probe.get('reference_dtype') != 'float64' or probe.get('reference_shape') != [1, 10]):
            raise WorkerFailure('R16 original reference metadata differs', 'source_failed')
        reference = np.asarray(probe.get('reference'), dtype=np.float64)
        if reference.shape != (1, 10) or not np.isfinite(reference).all():
            raise WorkerFailure('R16 original references must be finite float64 [1,10]', 'source_failed')
        references.append(reference)
    return images, labels, references, source


def _load_state(args, result):
    _, torch = _numeric_modules()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict') if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not state or any(not isinstance(name, str) for name in state):
        raise WorkerFailure('R24 requires the original state_dict envelope', 'state_failed')
    result['checkpoint_state'] = []
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or not bool(torch.isfinite(value).all()):
            raise WorkerFailure('Original checkpoint has a nonfinite/non-CPU tensor', 'state_failed')
        result['checkpoint_state'].append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype), finite=True))
    return state


def _build_model(args, state, recorder):
    from . import r16_orbit_models
    previous = r16_orbit_models._RECORDER
    r16_orbit_models._RECORDER = recorder
    try:
        return r16_orbit_models.build_model(MODEL, args.model_source, state)
    finally:
        r16_orbit_models._RECORDER = previous


def _clone_model(module):
    return copy.deepcopy(module)


def _state_isolated(original, approximate):
    _, torch = _numeric_modules()
    left, right = original.state_dict(), approximate.state_dict()
    if set(left) != set(right):
        raise WorkerFailure('The approximation changed original state keys', 'state_failed')
    for name in left:
        if (left[name].dtype != right[name].dtype or not torch.equal(left[name], right[name])
                or (left[name].numel() and left[name].data_ptr() == right[name].data_ptr())):
            raise WorkerFailure('The approximation must retain separate equal original state: ' + name, 'state_failed')
    return dict(passed=True, state_items=len(left), equal_values=True, separate_storage=True,
                integer_bn_preserved=True)


def _require_logits(output):
    _, torch = _numeric_modules()
    if (not isinstance(output, torch.Tensor) or output.dtype != torch.float64
            or tuple(output.shape) != (1, 10) or output.device.type != 'cpu'):
        raise WorkerFailure('R24 model output must be CPU float64 [1,10]', 'shape_failed')
    if not bool(torch.isfinite(output).all()):
        raise WorkerFailure('R24 model output contains nonfinite values', 'nonfinite')
    return output.detach().numpy()


def _replace_activations(module, native_silu, coefficients, args, recorder, result):
    np, torch = _numeric_modules()
    selected = [(name, child) for name, child in module.named_modules() if isinstance(child, torch.nn.SiLU)]
    if len(selected) != EXPECTED_SILU_MODULES:
        raise WorkerFailure('The original model must have exactly ten SiLU instances', 'structure_failed')

    class PhysicalSiLU(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.native_name = name

        def forward(self, z):
            name = self.native_name
            result['activation_attempts'] += 1
            _checkpoint(args, result)
            if (not isinstance(z, torch.Tensor) or z.device.type != 'cpu' or z.dtype != torch.float64
                    or not z.numel() or not bool(torch.isfinite(z).all())):
                raise WorkerFailure('Native activation input must be finite CPU float64', 'nonfinite')
            u = z.detach().numpy() / 32
            with recorder.span('native_activation', module=name,
                               probe_index=result.get('active_probe_index')):
                native = native_silu(name, u)
                result['activation_completed'] += 1
                _checkpoint(args, result)
                result['math_check_attempts'] += 1
                _checkpoint(args, result)
                reference = odd_chebyshev_reference(coefficients, u)
                checked = _check_math(reference, native)
                result['math_check_completed'] += 1
                row = dict(module=name, probe_index=result.get('active_probe_index'),
                           sample_id=result.get('active_sample_id'), call_index=result['activation_attempts'],
                           shape=list(u.shape), element_count=int(u.size), max_abs_u=float(np.abs(u).max()),
                           outside_standard_interval_elements=int(np.count_nonzero(np.abs(u) > 1)), **checked)
                result['activation_rows'].append(row)
                statistics = result['module_stats'].setdefault(name, dict(calls=0, element_count=0,
                    outside_standard_interval_elements=0, max_abs_u=0.0,
                    max_absolute_error=None, max_error_ratio=None, all_finite=True, all_math_passed=True))
                statistics['calls'] += 1
                statistics['element_count'] += row['element_count']
                statistics['outside_standard_interval_elements'] += row['outside_standard_interval_elements']
                statistics['max_abs_u'] = max(statistics['max_abs_u'], row['max_abs_u'])
                statistics['all_finite'] = statistics['all_finite'] and checked['finite']
                statistics['all_math_passed'] = statistics['all_math_passed'] and checked['passed']
                for key in ('max_absolute_error', 'max_error_ratio'):
                    if checked[key] is not None:
                        statistics[key] = max(statistics[key] or 0.0, checked[key])
                if result['math_check_completed'] == 3 * EXPECTED_ACTIVATIONS and all(
                        item['passed'] for item in result['activation_rows']):
                    result['native_polynomial_math_passed'] = True
                _checkpoint(args, result)
                if not checked['passed']:
                    raise WorkerFailure('Original tree and independent Chebyshev arithmetic differ',
                                        'math_failed' if checked['finite'] else 'nonfinite')
                physical = 32 * native
                if not np.isfinite(physical).all():
                    raise WorkerFailure('Physical activation contains nonfinite values', 'nonfinite')
                return torch.from_numpy(physical)

    names = []
    for name, _ in selected:
        parent_name, _, attribute = name.rpartition('.')
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attribute, PhysicalSiLU(name))
        names.append(name)
        result['silu_modules_replaced'] += 1
        _checkpoint(args, result)
    return names


def run_checks(args, recorder, result) -> None:
    with _phase(args, recorder, result, 'source_records'):
        summary, r13, r16 = _check_sources(args)
        result.update(source_checked=True, source_summary=summary)
        write_json(Path(args.work_dir) / 'source-summary.json', summary)
    with _phase(args, recorder, result, 'numeric_environment'):
        result['numeric_environment'] = _initialize_torch()
    np, torch = _numeric_modules()
    with _phase(args, recorder, result, 'native_function_load'):
        result['native_load_attempts'] += 1
        _checkpoint(args, result)
        native_silu, coefficients, metadata = load_native_functions(args.poly_dir)
        result.update(native_metadata=metadata)
        result['native_load_completed'] += 1
    with _phase(args, recorder, result, 'checkpoint_load'):
        result['checkpoint_load_attempts'] += 1
        _checkpoint(args, result)
        state = _load_state(args, result)
        result['checkpoint_load_completed'] += 1
    with _phase(args, recorder, result, 'model_build'):
        result['model_build_attempts'] += 1
        _checkpoint(args, result)
        module, report = _build_model(args, state, recorder)
        result['model_build_completed'] += 1
        result['state_report'] = report
        if (report.get('passed') is not True or report.get('strict_load') is not True
                or report.get('extra_state') != [] or len(state) != EXPECTED_STATE_COUNT
                or len(report.get('original_state') or []) != EXPECTED_STATE_COUNT
                or (report.get('structure') or {}).get('passed') is not True
                or (report.get('structure') or {}).get('activation_occurrences') != EXPECTED_ACTIVATIONS
                or any(row.get('finite') is not True or row.get('numerically_equal') is not True
                       for row in report.get('original_state', []))):
            raise WorkerFailure('Original state/structure must pass strict R16 loading', 'state_failed')
        del state
    with _phase(args, recorder, result, 'read_saved_training_probes'):
        result['probe_load_attempts'] += 1
        _checkpoint(args, result)
        images, labels, references, source = _load_probes(args, r13, r16, result)
        result['probe_load_completed'] += 1
        result['probe_source'] = source
    originals = []
    for index in range(3):
        with _phase(args, recorder, result, 'original_forward', probe_index=index, sample_id=source['sample_ids'][index]):
            result['original_forward_attempts'] += 1
            _checkpoint(args, result)
            with torch.no_grad():
                output = module(torch.from_numpy(images[index:index + 1]))
            result['original_forward_completed'] += 1
            observed = _require_logits(output)
            checked = _check_math(references[index], observed, logits=True)
            result['original_reference_checks'] += 1
            result['original_probes'].append(dict(probe_index=index, sample_id=source['sample_ids'][index],
                label=int(labels[index]), reference=references[index].tolist(), observed=observed.tolist(), **checked))
            _checkpoint(args, result)
            if not checked['passed']:
                raise WorkerFailure('Original model disagrees with its saved R16 reference', 'reference_failed')
            originals.append(observed.copy())
    result['original_reference_passed'] = True
    _checkpoint(args, result)
    with _phase(args, recorder, result, 'deepcopy_original'):
        result['deepcopy_attempts'] += 1
        _checkpoint(args, result)
        approximate = _clone_model(module)
        result['deepcopy_completed'] += 1
        result['copy_state_check'] = _state_isolated(module, approximate)
    with _phase(args, recorder, result, 'replace_ten_silu_instances'):
        result['replaced_modules'] = _replace_activations(approximate, native_silu, coefficients, args, recorder, result)
        approximate.eval()
        result['replacement_state_check'] = _state_isolated(module, approximate)
    for index in range(3):
        result.update(active_probe_index=index, active_sample_id=source['sample_ids'][index])
        with _phase(args, recorder, result, 'physical_scaled_forward', probe_index=index, sample_id=source['sample_ids'][index]):
            count_before = result['activation_attempts']
            result['approx_forward_attempts'] += 1
            _checkpoint(args, result)
            with torch.no_grad():
                output = approximate(torch.from_numpy(images[index:index + 1]))
            result['approx_forward_completed'] += 1
            observed = _require_logits(output)
            if result['activation_attempts'] - count_before != EXPECTED_ACTIVATIONS:
                raise WorkerFailure('Each original forward must call nineteen native activations', 'structure_failed')
            difference = observed - originals[index]
            if not np.isfinite(difference).all():
                raise WorkerFailure('Logit differences contain nonfinite values', 'nonfinite')
            result['approximate_probes'].append(dict(probe_index=index, sample_id=source['sample_ids'][index],
                label=int(labels[index]), original=originals[index].tolist(), approximate=observed.tolist(),
                difference=difference.tolist(), max_absolute_difference=float(np.abs(difference).max()),
                original_argmax=int(np.argmax(originals[index])), approximate_argmax=int(np.argmax(observed)),
                argmax_changed=bool(np.argmax(originals[index]) != np.argmax(observed)),
                scope='fixed training arithmetic diagnostic; no accuracy acceptance or selection'))
    result['physical_scaled_forward_completed'] = True
    _checkpoint(args, result)
    with _phase(args, recorder, result, 'final_counts_and_state'):
        if (result['activation_attempts'] != 3 * EXPECTED_ACTIVATIONS
                or result['activation_completed'] != 3 * EXPECTED_ACTIVATIONS
                or result['math_check_attempts'] != 3 * EXPECTED_ACTIVATIONS
                or result['math_check_completed'] != 3 * EXPECTED_ACTIVATIONS
                or result['native_polynomial_math_passed'] is not True
                or metadata['poly_closure_attempts'] != 3 * EXPECTED_ACTIVATIONS
                or metadata['poly_closure_completed'] != 3 * EXPECTED_ACTIVATIONS
                or any(length != 16 for length in metadata['closure_lengths'])
                or any(scale != 1.0 for scale in metadata['closure_scales'])):
            raise WorkerFailure('Native closure/activation/check counts differ', 'structure_failed')
        result['final_state_check'] = _state_isolated(module, approximate)
        write_json(Path(args.work_dir) / 'native-metadata.json', metadata)
        write_json(Path(args.work_dir) / 'activation-statistics.json',
                   dict(rows=result['activation_rows'], modules=result['module_stats']))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work-dir', 'model-source', 'checkpoint', 'probes', 'probe-source',
                 'r13-result', 'r16-result', 'r22-result', 'r23-result', 'poly-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if any((args.work_dir / name).exists() for name in ('phase_result.json', 'events.jsonl')):
        raise ValueError('R24 preserves existing worker records')
    resource.setrlimit(resource.RLIMIT_AS, (4 * GIB, 4 * GIB))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * GIB, 8 * GIB))
    saved_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    result, recorder = _initial_result(), StageRecorder(args.work_dir / 'events.jsonl')
    started = time.perf_counter_ns()
    write_json(args.work_dir / 'phase_result.json', result)
    try:
        with recorder.span('r24.checks'):
            (args.work_dir / 'worker-source.py').write_text(Path(__file__).read_text())
            with _budget_guard(args, result):
                run_checks(args, recorder, result)
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
        signal.signal(signal.SIGTERM, saved_sigterm)
        result.update(elapsed_ns=time.perf_counter_ns() - started,
                      resource_ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      elapsed_scope='worker entry through checks and monitor cleanup; before terminal write')
        try:
            _checkpoint(args, result)
        except BaseException as exc:
            result['terminal_budget_error'] = f'{type(exc).__name__}: {exc}'[:2000]
            if result['status'] == 'ok':
                result.update(status=getattr(exc, 'status', 'error'), error=result['terminal_budget_error'])
            write_json(args.work_dir / 'phase_result.json', result)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
