"""R26: fixed-protocol paired quality of the original native physical model."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import threading
import time
import traceback

from .r10_paper_metrics import StageRecorder, write_json
from . import r24_orbit_polynomial_worker as native


ROUND = 'R26'
MODEL = 'pt_agent.experiments.r13_common_models:resnet20_silu'
MODE = 'orbit-native-physical-heldout-quality-diagnostic'
SCOPE = 'R22 original unoptimized packed graph on train:0,train:1,train:2; finite-input numerical agreement only'
GIB, CHECK_SECONDS = 1024**3, 1800
EXPECTED_SAMPLES, EXPECTED_ACTIVATIONS, EXPECTED_SILU_MODULES = 1000, 19, 10
EXPECTED_STATE_COUNT, EXPECTED_INTEGER_STATE_COUNT = 116, 19
NORMALIZATION = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
COUNTER_FIELDS = (
    'native_load_attempts', 'native_load_completed',
    'checkpoint_load_attempts', 'checkpoint_load_completed',
    'model_build_attempts', 'model_build_completed',
    'deepcopy_attempts', 'deepcopy_completed',
    'heldout_load_attempts', 'heldout_load_completed',
    'original_forward_attempts', 'original_forward_completed',
    'approx_forward_attempts', 'approx_forward_completed',
    'activation_attempts', 'activation_completed',
    'math_check_attempts', 'math_check_completed',
    'native_closure_attempts', 'native_closure_completed',
    'silu_modules_replaced', 'heldout_rows_read', 'heldout_samples',
    'raw_cifar_reads', 'evaluated_pairs', 'completed_samples',
)
COUNT_FIELDS = COUNTER_FIELDS
ZERO_FIELDS = (
    'llm_calls', 'downloads', 'installations', 'new_weights', 'new_datasets',
    'fit_calls', 'agent_calls', 'tuner_calls', 'count_runs', 'emit_runs',
    'trace_runs', 'compiler_runs', 'optimizer_runs', 'solver_runs', 'keygen_runs',
    'real_ckks_attempts', 'real_ckks_completed', 'heir_calls', 'confirm_samples',
    'report_samples', 'freeze_attempts', 'training_samples', 'compiled_constant_reads',
    'packed_forward_attempts', 'packed_forward_completed',
)
FALSE_FIELDS = ('backend_ready', 'security_validated', 'accuracy_validated',
                'formal_matrix_complete', 'heldout_packed_equivalence_verified')
UNMEASURED_FIELDS = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency',
                     'accuracy_acceptance', 'domain_acceptance')
STAGE_FIELDS = ('source_checked', 'original_model_loaded', 'approximation_built',
                'native_activation_math_passed', 'paired_quality_completed')
_HELDOUT_PROGRESS = None


class WorkerFailure(native.WorkerFailure):
    pass


def _initial_result():
    result = dict(round=ROUND, model=MODEL, mode=MODE, status='running', error=None,
                  failed_phase=None, active_sample_id=None, active_index=None,
                  active_module=None, native_metadata=None, module_stats={},
                  sample_rows=[], current_sample_stats=None, validation_metrics=None,
                  paired_quality_policy_passed=None, all_finite=None, outputs_partial=True,
                  native_packed_equivalence_verified=False, native_packed_equivalence_scope=SCOPE,
                  inference_normalization_constants_source_verified=False,
                  training_preprocessing_verified=False,
                  official_default_dtype_equivalence_verified=False,
                  phase_artifact_bytes=0, case_artifact_bytes=0,
                  budget_monitor_completed=False, acc_drop_pp_unit='percentage points')
    for names, value in ((COUNTER_FIELDS, 0), (ZERO_FIELDS, 0), (FALSE_FIELDS, False),
                         (STAGE_FIELDS, False), (UNMEASURED_FIELDS, None)):
        result.update(dict.fromkeys(names, value))
    return result


def _numeric_modules():
    return native._numeric_modules()


def _initialize_torch():
    return native._initialize_torch()


def _read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise WorkerFailure('A source record must be a JSON object', 'source_failed')
    return value


def _same_path(left, right):
    return isinstance(left, (str, Path)) and Path(left).resolve() == Path(right).resolve()


def _file_record(path):
    path = Path(path).resolve()
    if not path.is_file() or not path.stat().st_size:
        raise WorkerFailure('Missing source file: ' + str(path), 'source_failed')
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


def _sync_native_counts(result):
    metadata = result.get('native_metadata')
    if metadata is not None:
        result['native_closure_attempts'] = metadata.get('poly_closure_attempts')
        result['native_closure_completed'] = metadata.get('poly_closure_completed')


def _budget_observation(args, started=None):
    work = Path(args.work_dir).resolve()
    if started is None:
        started = getattr(args, '_started_ns', None)
    now = time.perf_counter_ns()
    return dict(ts_ns=now, case_root=str(work.parent.parent),
                case_artifact_bytes=_size(work.parent.parent),
                disk_free_bytes=shutil.disk_usage(work).free,
                elapsed_ns=None if started is None else now - started)


def _budget_error(row):
    if row['case_artifact_bytes'] > 4 * GIB:
        return 'artifact_budget_exceeded', 'R26 case artifacts exceed 4 GiB'
    if row['disk_free_bytes'] < 10 * GIB:
        return 'disk_reserve', 'R26 disk reserve is below 10 GiB'
    if row['elapsed_ns'] is not None and row['elapsed_ns'] >= CHECK_SECONDS * 1_000_000_000:
        return 'timeout', 'R26 inner allowance exhausted'
    return None


def _checkpoint(args, result):
    _sync_native_counts(result)
    work = Path(args.work_dir).resolve()
    row = _budget_observation(args)
    result.update(phase_artifact_bytes=_size(work), case_artifact_bytes=row['case_artifact_bytes'],
                  last_budget_observation=row)
    write_json(work / 'phase_result.json', result)
    failure = _budget_error(row)
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
    work, stop = Path(args.work_dir).resolve(), threading.Event()
    started = getattr(args, '_started_ns', time.perf_counter_ns())

    def watch():
        while not stop.wait(1):
            try:
                row = _budget_observation(args, started)
                with (work / 'budget-samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                failure = _budget_error(row)
                if failure is None:
                    continue
                record = dict(status=failure[0], error=failure[1], **row)
            except BaseException as exc:
                record = dict(status='record_error', error=f'{type(exc).__name__}: {exc}'[:2000],
                              ts_ns=time.perf_counter_ns())
            record.update(termination_origin='worker_budget_monitor',
                          counters={name: result.get(name) for name in COUNTER_FIELDS},
                          active_sample_id=result.get('active_sample_id'),
                          active_module=result.get('active_module'))
            try:
                write_json(work / 'budget-fatal.json', record)
            finally:
                _terminate_self()
            return

    thread = threading.Thread(target=watch, name='r26-budget-observer', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)
        result['budget_monitor_completed'] = not thread.is_alive()
        if thread.is_alive():
            raise WorkerFailure('R26 resource observer did not stop', 'record_error')


def _check_sources(args):
    records = {name: _read_json(getattr(args, name + '_result'))
               for name in ('r13', 'r16', 'r22', 'r24', 'r25')}
    for name, record in records.items():
        if record.get('round') != name.upper() or record.get('model') != MODEL or record.get('status') != 'ok':
            raise WorkerFailure('Original source identity/status differs: ' + name, 'source_failed')
    r13, r16, r22, r24, r25 = (records[name] for name in ('r13', 'r16', 'r22', 'r24', 'r25'))
    w13, w16, w24, w25 = (record.get('worker') or {} for record in (r13, r16, r24, r25))
    saved = w16.get('source_summary') or {}
    if (r16.get('baseline') != 'orbit'
            or any(w16.get(name) is not True for name in
                   ('source_checked', 'strict_load_passed', 'structure_passed', 'math_probes_passed'))
            or any(not _same_path(saved.get(key), getattr(args, key))
                   for key in ('model_source', 'checkpoint', 'r13_result'))):
        raise WorkerFailure('Original Orbit source/strict-state provenance differs', 'source_failed')
    if (r22.get('mode') != 'orbit-resnet20-trace-tdag-preflight'
            or r22.get('orbit_model_trace_passed') is not True or r22.get('tdag_parse_passed') is not True
            or any(r22.get(key, {}).get(phase) != 1 for key in ('attempted', 'completed') for phase in ('trace', 'parse'))
            or Path(args.poly_dir).resolve() != Path(args.r22_result).resolve().parent / 'work/local-poly/poly'):
        raise WorkerFailure('Original R22 native polynomial source differs', 'source_failed')
    if (r24.get('mode') != 'orbit-native-polynomial-physical-scale-check'
            or any(w24.get(name) is not True for name in
                   ('original_reference_passed', 'native_polynomial_math_passed', 'physical_scaled_forward_completed'))
            or any(w24.get(name) != count for name, count in dict(original_forward_attempts=3,
                   original_forward_completed=3, approx_forward_attempts=3, approx_forward_completed=3,
                   activation_attempts=57, activation_completed=57, math_check_attempts=57,
                   math_check_completed=57).items())
            or any(not _same_path(r24.get('source_records', {}).get('records', {}).get(name),
                                  getattr(args, name + '_result')) for name in ('r13', 'r16', 'r22'))):
        raise WorkerFailure('Original R24 native physical-math evidence differs', 'source_failed')
    if (r25.get('mode') != 'orbit-saved-packed-graph-three-probe-check'
            or w25.get('source_checked') is not True or w25.get('native_packed_equivalence_verified') is not True
            or w25.get('native_packed_equivalence_scope') != SCOPE
            or any(w25.get(name) != count for name, count in dict(packed_forward_attempts=3,
                   packed_forward_completed=3, math_check_attempts=3, math_check_completed=3,
                   op_attempts=30336, op_completed=30336).items())
            or any(not _same_path(r25.get('source_records', {}).get('records', {}).get(name),
                                  getattr(args, name + '_result')) for name in ('r13', 'r22', 'r24'))):
        raise WorkerFailure('Original R25 three-probe packed evidence differs', 'source_failed')
    info = _read_json(args.model_info)
    original = info.get('source') or {}
    input_rows, split = info.get('inputs'), info.get('split') or {}
    artifact = (w13.get('artifacts') or {}).get('model_info')
    if (not isinstance(artifact, str) or not _same_path(Path(artifact) / 'model_info.json', args.model_info)
            or info.get('task') != 'classification' or info.get('metric') != 'classification_top1'
            or info.get('threshold') != {'acc_drop_pp': 0.5}
            or original.get('factory') != MODEL or original.get('normalization') != NORMALIZATION
            or original.get('training_preprocessing_verified') is not False
            or original.get('preprocessing_provenance') != 'official inference test'
            or not isinstance(input_rows, list) or len(input_rows) != 1
            or input_rows[0].get('kind') != 'features' or input_rows[0].get('dtype') != 'float64'
            or input_rows[0].get('shape') != [1, 3, 32, 32]
            or split.get('kind') != 'legacy_cifar' or split.get('heldout') != list(range(9000, 10000))
            or not _same_path((w13.get('source') or {}).get('records'), args.records)
            or not _same_path((original.get('calibration_source') or {}).get('records'), args.records)
            or original.get('normalization') != (w13.get('source') or {}).get('normalization')):
        raise WorkerFailure('Original ModelInfo policy/features/split/source differs', 'source_failed')
    paths = {name: _file_record(getattr(args, name)) for name in
             ('model_source', 'checkpoint', 'model_info', 'r13_result', 'r16_result',
              'r22_result', 'r24_result', 'r25_result')}
    summary = dict(paths=paths, records=str(Path(args.records).resolve()),
                   poly_dir=str(Path(args.poly_dir).resolve()), source_times_included=False,
                   policy=dict(task=info['task'], metric=info['metric'], threshold=info['threshold']),
                   normalization=original['normalization'],
                   preprocessing_provenance='official inference test',
                   inference_normalization_constants_source_verified=True,
                   training_preprocessing_verified=False,
                   official_default_dtype_equivalence_verified=False,
                   coefficient_source_equivalence_verified=False,
                   inherited_packed_scope=SCOPE)
    return summary, info


def read_heldout(args, source_metadata) -> tuple:
    """Read exactly the one original heldout shard, then apply float64 normalization."""
    np, _ = _numeric_modules()
    info_source = source_metadata.get('source') or {}
    if (info_source.get('normalization') != NORMALIZATION
            or not _same_path((info_source.get('calibration_source') or {}).get('records'), args.records)
            or source_metadata.get('split', {}).get('kind') != 'legacy_cifar'
            or source_metadata.get('split', {}).get('heldout') != list(range(9000, 10000))):
        raise WorkerFailure('Fixed heldout source/normalization differs', 'source_failed')
    root = Path(args.records).resolve()
    metadata = _read_json(root / 'dataset.json')
    expected = dict(format='poseidon-cifar10-binary-records-v1', record_bytes=3073,
                    row_label_byte=0, preprocessing_applied=False, train_count=50000,
                    test_count=10000, pixel_layout='uint8 planar RGB 3x32x32, C order')
    shard_name = 'test-09000.records'
    entries = [entry for entry in metadata.get('shards', []) if isinstance(entry, dict)
               and entry.get('file') == shard_name and entry.get('split') == 'test'
               and entry.get('start') == 9000 and entry.get('count') == 1000]
    shard = root / shard_name
    if (any(metadata.get(key) != value for key, value in expected.items())
            or len(entries) != 1 or not shard.is_file() or shard.stat().st_size != 1000 * 3073):
        raise WorkerFailure('Original heldout shard metadata/record count differs', 'source_failed')
    mapping = np.memmap(shard, mode='r', dtype=np.uint8, shape=(1000, 3073))
    try:
        records = np.array(mapping, copy=True)
    finally:
        del mapping
    if _HELDOUT_PROGRESS is not None:
        _HELDOUT_PROGRESS(dict(raw_cifar_reads=1, heldout_rows_read=1000, heldout_samples=1000))
    labels = records[:, 0].astype(np.int64)
    pixels = records[:, 1:].reshape(1000, 3, 32, 32)
    if labels.shape != (1000,) or bool(np.any(labels < 0) or np.any(labels > 9)):
        raise WorkerFailure('Heldout class labels must be integers in 0..9', 'shape_failed')
    mean = np.asarray(NORMALIZATION['mean'], dtype=np.float64).reshape(1, 3, 1, 1)
    std = np.asarray(NORMALIZATION['std'], dtype=np.float64).reshape(1, 3, 1, 1)
    images = (np.asarray(pixels, dtype=np.float64) / 255.0 - mean) / std
    if images.dtype != np.dtype('float64') or images.shape != (1000, 3, 32, 32) or not bool(np.isfinite(images).all()):
        raise WorkerFailure('Normalized heldout images must be finite float64', 'nonfinite')
    source = dict(records=str(root), dataset_metadata=expected, shard=shard_name,
                  shard_metadata={key: entries[0].get(key) for key in ('file', 'split', 'start', 'count', 'bytes')},
                  source_split='test', split='heldout', batch=1,
                  indices=list(range(9000, 10000)),
                  sample_ids=['test:' + str(index) for index in range(9000, 10000)],
                  labels=labels.tolist(), count=1000, normalization=NORMALIZATION,
                  normalization_arithmetic='float64(pixels)/255; subtract mean; divide std',
                  raw_cifar_reads=1, heldout_rows_read=1000, confirm_samples=0, report_samples=0)
    return images, labels, source


def load_original_model(args, recorder, result) -> object:
    _, torch = _numeric_modules()
    with _phase(args, recorder, result, 'checkpoint_load'):
        result['checkpoint_load_attempts'] += 1
        _checkpoint(args, result)
        state = native._load_state(args, result)
        result['checkpoint_load_completed'] += 1
        integer_count = sum(not value.is_floating_point() for value in state.values())
        if len(state) != EXPECTED_STATE_COUNT or integer_count != EXPECTED_INTEGER_STATE_COUNT:
            raise WorkerFailure('Original checkpoint state/BN inventory differs', 'state_failed')
    with _phase(args, recorder, result, 'model_build'):
        result['model_build_attempts'] += 1
        _checkpoint(args, result)
        module, report = native._build_model(args, state, recorder)
        result['model_build_completed'] += 1
        result['state_report'] = report
        if (report.get('passed') is not True or report.get('strict_load') is not True
                or report.get('extra_state') != []
                or len(report.get('original_state') or []) != EXPECTED_STATE_COUNT
                or report.get('structure', {}).get('passed') is not True
                or report.get('structure', {}).get('activation_occurrences') != EXPECTED_ACTIVATIONS
                or any(row.get('finite') is not True or row.get('numerically_equal') is not True
                       for row in report.get('original_state', []))):
            raise WorkerFailure('Original CPU model must retain strict state and structure', 'state_failed')
        del state
        if any(child.training for child in module.modules()):
            raise WorkerFailure('The original model must remain in eval mode', 'state_failed')
        result['original_model_loaded'] = True
    return module


def _new_statistics():
    return dict(calls_attempted=0, calls_completed=0, math_attempted=0, math_completed=0,
                element_count=0, max_abs_u=0.0, outside_standard_interval_elements=0,
                max_absolute_error=0.0, max_error_ratio=0.0, all_finite=True,
                all_math_passed=True, native_elapsed_ns=0, math_elapsed_ns=0)


def _merge_statistics(target, row):
    for key in ('element_count', 'outside_standard_interval_elements'):
        target[key] += row[key]
    for key in ('max_abs_u', 'max_absolute_error', 'max_error_ratio'):
        if row.get(key) is not None:
            target[key] = max(target[key], row[key])
    target['all_finite'] = target['all_finite'] and row['finite']
    target['all_math_passed'] = target['all_math_passed'] and row['passed']


def build_approximation(module, native_silu, coefficients, args, recorder, result) -> object:
    np, torch = _numeric_modules()
    with _phase(args, recorder, result, 'deepcopy_original'):
        result['deepcopy_attempts'] += 1
        _checkpoint(args, result)
        approximate = native._clone_model(module)
        result['deepcopy_completed'] += 1
        result['copy_state_check'] = native._state_isolated(module, approximate)
    selected = [(name, child) for name, child in approximate.named_modules()
                if isinstance(child, torch.nn.SiLU)]
    stem_name = 'module.mish' if isinstance(approximate, torch.nn.DataParallel) else 'mish'
    if len(selected) != EXPECTED_SILU_MODULES or stem_name not in dict(selected):
        raise WorkerFailure('The approximation must retain ten original SiLU instances', 'structure_failed')
    result['expected_module_calls_per_sample'] = {name: 1 if name == stem_name else 2 for name, _ in selected}
    result['module_stats'] = {name: _new_statistics() for name, _ in selected}

    class PhysicalSiLU(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.native_name = name

        def forward(self, z):
            name = self.native_name
            result['active_module'] = name
            statistics = result['module_stats'][name]
            current = result.get('current_sample_stats')
            if current is None:
                raise WorkerFailure('A native call needs explicit active sample accounting', 'structure_failed')
            result['activation_attempts'] += 1
            statistics['calls_attempted'] += 1
            current['calls_attempted'] += 1
            if (not isinstance(z, torch.Tensor) or z.device.type != 'cpu' or z.dtype != torch.float64
                    or not z.numel() or not bool(torch.isfinite(z).all())):
                statistics['all_finite'] = current['all_finite'] = False
                raise WorkerFailure('Native activation input must be finite CPU float64', 'nonfinite')
            u = z.detach().numpy() / 32.0
            result['active_activation'] = dict(module=name, shape=list(u.shape),
                element_count=int(u.size), max_abs_u=float(np.abs(u).max()),
                outside_standard_interval_elements=int(np.count_nonzero(np.abs(u) > 1)))
            started = time.perf_counter_ns()
            try:
                value = native_silu(name, u)
                result['activation_completed'] += 1
                statistics['calls_completed'] += 1
                current['calls_completed'] += 1
                current['module_calls'][name] = current['module_calls'].get(name, 0) + 1
            finally:
                elapsed = time.perf_counter_ns() - started
                statistics['native_elapsed_ns'] += elapsed
                current['native_elapsed_ns'] += elapsed
                _sync_native_counts(result)
            if (not isinstance(value, np.ndarray) or value.dtype != np.dtype('float64')
                    or value.shape != u.shape or not bool(np.isfinite(value).all())):
                statistics['all_finite'] = current['all_finite'] = False
                raise WorkerFailure('Native activation output must preserve finite float64 shape', 'nonfinite')
            result['math_check_attempts'] += 1
            statistics['math_attempted'] += 1
            current['math_attempted'] += 1
            started = time.perf_counter_ns()
            try:
                reference = native.odd_chebyshev_reference(coefficients, u)
                checked = native._check_math(reference, value)
                result['math_check_completed'] += 1
                statistics['math_completed'] += 1
                current['math_completed'] += 1
            finally:
                elapsed = time.perf_counter_ns() - started
                statistics['math_elapsed_ns'] += elapsed
                current['math_elapsed_ns'] += elapsed
            row = dict(element_count=int(u.size), max_abs_u=float(np.abs(u).max()),
                       outside_standard_interval_elements=int(np.count_nonzero(np.abs(u) > 1)), **checked)
            _merge_statistics(statistics, row)
            _merge_statistics(current, row)
            if not checked['passed']:
                raise WorkerFailure('Native tree differs from the independent full g_ref',
                                    'math_failed' if checked['finite'] else 'nonfinite')
            with np.errstate(over='ignore', invalid='ignore'):
                physical = 32.0 * value
            if not bool(np.isfinite(physical).all()):
                statistics['all_finite'] = current['all_finite'] = False
                raise WorkerFailure('Physical native activation is nonfinite', 'nonfinite')
            return torch.from_numpy(physical)

    with _phase(args, recorder, result, 'replace_ten_silu_instances'):
        for name, _ in selected:
            parent_name, _, attribute = name.rpartition('.')
            parent = approximate.get_submodule(parent_name) if parent_name else approximate
            setattr(parent, attribute, PhysicalSiLU(name))
            result['silu_modules_replaced'] += 1
        approximate.eval()
        result['replaced_modules'] = [name for name, _ in selected]
        result['replacement_state_check'] = native._state_isolated(module, approximate)
        result['approximation_built'] = True
    return approximate


def _snapshot_state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _check_unchanged_state(module, snapshot):
    _, torch = _numeric_modules()
    observed = module.state_dict()
    if set(observed) != set(snapshot) or any(observed[name].dtype != snapshot[name].dtype
            or not torch.equal(observed[name], snapshot[name]) for name in snapshot):
        raise WorkerFailure('Original parameter/BN state changed during the paired run', 'state_failed')
    return dict(passed=True, original_state_items=len(snapshot), all_values_unchanged=True)


def _native_functions(args):
    return native.load_native_functions(args.poly_dir)


def _require_logits(output):
    return native._require_logits(output)


def _append_record(path, record):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def _run_validation(args, recorder, result):
    global _HELDOUT_PROGRESS
    with _phase(args, recorder, result, 'source_records'):
        summary, info = _check_sources(args)
        result.update(source_checked=True, source_summary=summary,
                      inference_normalization_constants_source_verified=True,
                      native_packed_equivalence_verified=True)
        write_json(Path(args.work_dir) / 'source-summary.json', summary)
    with _phase(args, recorder, result, 'numeric_environment'):
        result['numeric_environment'] = _initialize_torch()
    np, torch = _numeric_modules()
    with _phase(args, recorder, result, 'native_function_load'):
        result['native_load_attempts'] += 1
        _checkpoint(args, result)
        native_silu, coefficients, metadata = _native_functions(args)
        result['native_metadata'] = metadata
        result['native_load_completed'] += 1
    module = load_original_model(args, recorder, result)
    approximate = build_approximation(module, native_silu, coefficients, args, recorder, result)
    initial_state = _snapshot_state(module)
    with _phase(args, recorder, result, 'read_heldout'):
        result['heldout_load_attempts'] += 1
        _checkpoint(args, result)
        previous = _HELDOUT_PROGRESS

        def progress(row):
            result.update(row)
            _checkpoint(args, result)

        _HELDOUT_PROGRESS = progress
        try:
            images, labels, source = read_heldout(args, info)
        finally:
            _HELDOUT_PROGRESS = previous
        result['heldout_load_completed'] += 1
        result.update(sample_source=source, raw_cifar_reads=source['raw_cifar_reads'],
                      heldout_rows_read=source['heldout_rows_read'], heldout_samples=source['heldout_rows_read'])
        if (images.shape != (EXPECTED_SAMPLES, 3, 32, 32) or images.dtype != np.dtype('float64')
                or labels.shape != (EXPECTED_SAMPLES,) or labels.dtype.kind not in ('i', 'u')
                or not bool(np.isfinite(images).all())
                or bool(np.any(labels < 0) or np.any(labels > 9))):
            raise WorkerFailure('Heldout arrays/labels differ from the fixed protocol', 'shape_failed')
        write_json(Path(args.work_dir) / 'sample-source.json', source)
    originals = np.empty((EXPECTED_SAMPLES, 10), dtype=np.float64)
    approximations = np.empty_like(originals)
    sample_file = Path(args.work_dir) / 'samples.jsonl'
    for index in range(EXPECTED_SAMPLES):
        sample_id = 'test:' + str(9000 + index)
        result.update(active_index=9000 + index, active_sample_id=sample_id,
                      active_module=None, current_sample_stats=dict(_new_statistics(), module_calls={}))
        result['failed_phase'] = 'original_forward'
        result['original_forward_attempts'] += 1
        with recorder.span('original_forward', sample_id=sample_id, index=9000 + index):
            with torch.no_grad():
                output = module(torch.from_numpy(images[index:index + 1]))
            result['original_forward_completed'] += 1
            original = _require_logits(output)
        result['failed_phase'] = 'approx_forward'
        result['approx_forward_attempts'] += 1
        with recorder.span('approx_forward', sample_id=sample_id, index=9000 + index):
            with torch.no_grad():
                output = approximate(torch.from_numpy(images[index:index + 1]))
            result['approx_forward_completed'] += 1
            observed = _require_logits(output)
        current = result['current_sample_stats']
        if (current['calls_completed'] != EXPECTED_ACTIVATIONS
                or current['math_completed'] != EXPECTED_ACTIVATIONS
                or current['module_calls'] != result['expected_module_calls_per_sample']
                or not current['all_math_passed'] or not current['all_finite']):
            raise WorkerFailure('Each paired approximation needs nineteen valid native calls', 'structure_failed')
        with np.errstate(over='ignore', invalid='ignore'):
            difference = observed - original
        if not bool(np.isfinite(difference).all()):
            raise WorkerFailure('Paired logit differences must be finite', 'nonfinite')
        row = dict(index=9000 + index, sample_id=sample_id, label=int(labels[index]),
                   original=original.tolist(), approximate=observed.tolist(),
                   original_argmax=int(np.argmax(original)), approximate_argmax=int(np.argmax(observed)),
                   max_logit_absolute_difference=float(np.abs(difference).max()),
                   activation_math=dict(current), partial=True)
        result['failed_phase'] = 'save_pair'
        _append_record(sample_file, row)
        result['sample_rows'].append(row)
        originals[index], approximations[index] = original[0], observed[0]
        result['evaluated_pairs'] += 1
        result['completed_samples'] += 1
        del output, original, observed, difference, current, row
        if result['evaluated_pairs'] % 10 == 0:
            _checkpoint(args, result)
    with _phase(args, recorder, result, 'final_state_and_counts'):
        _sync_native_counts(result)
        result['final_original_state_check'] = _check_unchanged_state(module, initial_state)
        result['final_state_check'] = native._state_isolated(module, approximate)
        total_calls = EXPECTED_SAMPLES * EXPECTED_ACTIVATIONS
        counts = dict(native_load_attempts=1, native_load_completed=1,
                      checkpoint_load_attempts=1, checkpoint_load_completed=1,
                      model_build_attempts=1, model_build_completed=1,
                      deepcopy_attempts=1, deepcopy_completed=1,
                      heldout_load_attempts=1, heldout_load_completed=1,
                      original_forward_attempts=EXPECTED_SAMPLES, original_forward_completed=EXPECTED_SAMPLES,
                      approx_forward_attempts=EXPECTED_SAMPLES, approx_forward_completed=EXPECTED_SAMPLES,
                      activation_attempts=total_calls, activation_completed=total_calls,
                      math_check_attempts=total_calls, math_check_completed=total_calls,
                      native_closure_attempts=total_calls, native_closure_completed=total_calls,
                      silu_modules_replaced=EXPECTED_SILU_MODULES,
                      heldout_rows_read=1000, heldout_samples=1000, raw_cifar_reads=1,
                      evaluated_pairs=EXPECTED_SAMPLES, completed_samples=EXPECTED_SAMPLES)
        if (any(result.get(name) != value for name, value in counts.items())
                or len(result['sample_rows']) != EXPECTED_SAMPLES
                or any(length != 16 for length in metadata.get('closure_lengths', []))
                or any(scale != 1.0 for scale in metadata.get('closure_scales', []))
                or len(metadata.get('closure_lengths', [])) != total_calls
                or len(metadata.get('closure_scales', [])) != total_calls
                or any(statistics['calls_completed'] != EXPECTED_SAMPLES * result['expected_module_calls_per_sample'][name]
                       or statistics['math_completed'] != statistics['calls_completed']
                       or not statistics['all_math_passed'] or not statistics['all_finite']
                       for name, statistics in result['module_stats'].items())):
            raise WorkerFailure('Full paired/native/closure/state counts differ', 'structure_failed')
        result['native_activation_math_passed'] = True
    with _phase(args, recorder, result, 'paired_quality'):
        from ..metrics import compare, passes
        measured = compare(dict(values=originals, task='classification'), approximations, labels)
        accepted = bool(passes(measured, info['threshold']))
        write_json(Path(args.work_dir) / 'paired-quality.json',
                   dict(validation_metrics=measured, paired_quality_policy_passed=accepted,
                        policy=info['threshold'], count=EXPECTED_SAMPLES,
                        acc_drop_pp_unit='percentage points', formal_accuracy_acceptance=None))
        write_json(Path(args.work_dir) / 'native-statistics.json',
                   dict(module_stats=result['module_stats'], native_metadata=metadata,
                        standard_interval_is_descriptive=True))
        result.update(validation_metrics=measured, paired_quality_policy_passed=accepted,
                      paired_quality_completed=True, all_finite=True, outputs_partial=False,
                      status='ok' if accepted else 'quality_rejected')


def _invalidate_quality(result):
    result.update(validation_metrics=None, paired_quality_policy_passed=None,
                  paired_quality_completed=False, outputs_partial=True, all_finite=None)


def run_validation(args, recorder, result) -> None:
    try:
        _run_validation(args, recorder, result)
    except BaseException:
        _invalidate_quality(result)
        try:
            _checkpoint(args, result)
        except BaseException:
            pass
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work-dir', 'model-source', 'checkpoint', 'poly-dir', 'records', 'model-info',
                 'r13-result', 'r16-result', 'r22-result', 'r24-result', 'r25-result'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if any((args.work_dir / name).exists() for name in ('phase_result.json', 'events.jsonl')):
        raise ValueError('R26 preserves existing worker evidence')
    resource.setrlimit(resource.RLIMIT_AS, (4 * GIB, 4 * GIB))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * GIB, 8 * GIB))
    saved_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    result, recorder = _initial_result(), StageRecorder(args.work_dir / 'events.jsonl')
    args._started_ns = time.perf_counter_ns()
    write_json(args.work_dir / 'phase_result.json', result)
    try:
        with recorder.span('r26.validation'):
            (args.work_dir / 'worker-source.py').write_text(Path(__file__).read_text())
            with _budget_guard(args, result):
                run_validation(args, recorder, result)
        result['failed_phase'] = None
    except native.WorkerFailure as exc:
        _invalidate_quality(result)
        result.update(status=exc.status, error=f'{type(exc).__name__}: {exc}'[:2000])
        traceback.print_exc()
    except (ImportError, FileNotFoundError) as exc:
        _invalidate_quality(result)
        result.update(status='dependency_missing', error=f'{type(exc).__name__}: {exc}'[:2000])
        traceback.print_exc()
    except MemoryError as exc:
        _invalidate_quality(result)
        result.update(status='resource_limit', error=f'{type(exc).__name__}: {exc}'[:2000])
        traceback.print_exc()
    except BaseException as exc:
        _invalidate_quality(result)
        result.update(status='error', error=f'{type(exc).__name__}: {exc}'[:2000])
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM, saved_sigterm)
        _sync_native_counts(result)
        result.update(elapsed_ns=time.perf_counter_ns() - args._started_ns,
                      resource_ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      elapsed_scope='worker entry through paired checks and monitor cleanup; before terminal write')
        try:
            _checkpoint(args, result)
        except BaseException as exc:
            result['terminal_budget_error'] = f'{type(exc).__name__}: {exc}'[:2000]
            if result['status'] in ('ok', 'quality_rejected'):
                _invalidate_quality(result)
                result.update(status=getattr(exc, 'status', 'error'), error=result['terminal_budget_error'])
            write_json(args.work_dir / 'phase_result.json', result)
    return 0 if result['status'] in ('ok', 'quality_rejected') else 1


if __name__ == '__main__':
    raise SystemExit(main())
