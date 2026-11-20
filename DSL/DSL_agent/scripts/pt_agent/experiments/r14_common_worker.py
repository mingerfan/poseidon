"""R14: one bounded rule-program preparation from an already saved R13 model."""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import resource
import signal

from .r10_paper_metrics import StageRecorder, read_events, write_json


INTERNAL_SECONDS = 1800
CASE_BYTES = 4 * 1024**3
EMIT_BYTES = 3 * 1024**3
GRAPH_CRITERION = '1e-12+1e-12*abs(reference)'
RULE_OPTIONS = dict(deg=63, margin=1.25, slots=16384, placement='depth-dp',
                    refresh='compiler', act_method='cheb')
REQUIRED_STAGES = ('source_checked', 'reference_passed', 'calibration_completed',
                   'plan_completed', 'typecheck_passed', 'signature_passed',
                   'count_completed', 'stats_equal', 'emit_completed', 'lowering_ready')


class WorkerFailure(RuntimeError):
    def __init__(self, message, *, status='error'):
        super().__init__(message)
        self.status = status


def check_saved_source(info, r13_result) -> dict:
    """Check the original source and inferred interface without altering ModelInfo."""
    from ..data import cifar
    from .r13_common_models import FACTORIES

    errors = []
    worker = r13_result.get('worker') or {}
    model = r13_result.get('model')
    metadata = FACTORIES.get(model)
    if (r13_result.get('round') != 'R13' or r13_result.get('status') != 'ok'
            or r13_result.get('mode') != 'common-model-intake-preparation'
            or worker.get('round') != 'R13' or worker.get('status') != 'ok'
            or worker.get('model') != model or metadata is None):
        errors.append('Expected one of the three successful original R13 preparations')
    required = ('import_completed', 'real_probes_passed', 'graph_math_passed',
                'roundtrip_passed', 'static_coverage')
    if any(worker.get(key) is not True for key in required):
        errors.append('The R13 intake, original references, readback and coverage must have passed')
    source = info.source
    if source.get('factory') != model or source.get('path') != model or source.get('type') != 'torch':
        errors.append('Saved source does not identify the original factory')
    if metadata is not None:
        for field, key in (('original_source', 'source'), ('checkpoint', 'checkpoint'),
                           ('inference_preprocessing_source', 'inference_preprocessing_source')):
            if Path(source.get(field, '')).resolve() != Path(metadata[key]).resolve():
                errors.append('Original source path differs: ' + field)
    expected_norm = dict(mean=list(cifar.MEAN), std=list(cifar.STD))
    if (source.get('spec_revision') != 'R13' or source.get('normalization') != expected_norm
            or source.get('preprocessing_provenance') != 'official inference test'
            or source.get('training_preprocessing_verified') is not False):
        errors.append('Saved preprocessing provenance differs from R13')
    if source != worker.get('model_source') or info.split != worker.get('split'):
        errors.append('Saved source or split differs from the R13 observation')
    expected_split = dict(kind='legacy_cifar', heldout=list(cifar.HELDOUT),
                          confirm=list(cifar.CONFIRM), report=list(cifar.REPORT),
                          calibration='frozen training selection')
    if info.split != expected_split:
        errors.append('The original fixed CIFAR split must be retained')
    graph = info.graph
    graph_inputs = list(graph.inputs) if graph is not None else []
    graph_outputs = list(graph.outputs) if graph is not None else []
    if len(graph_inputs) != 1 or len(graph_outputs) != 1:
        errors.append('Expected one original graph input and output')
    else:
        if tuple(graph.tensors[graph_inputs[0]].shape) != (1, 3, 32, 32):
            errors.append('Original graph input shape differs')
        if tuple(graph.tensors[graph_outputs[0]].shape) != (1, 10):
            errors.append('Original graph output shape differs')
    if len(info.inputs) != 1:
        errors.append('Expected one inferred ModelInfo input')
    else:
        entry = info.inputs[0]
        if (entry.get('name') != (graph_inputs[0] if graph_inputs else None)
                or entry.get('kind') != 'features' or entry.get('dtype') != 'float64'
                or entry.get('shape') != [1, 3, 32, 32]):
            errors.append('Saved input name, features kind, dtype or shape differs')
    if info.input_handling.get('internal_regression_source') != 'legacy_cifar':
        errors.append('Original internal sample source differs')
    return dict(passed=not errors, model=model, errors=errors,
                normalization=source.get('normalization'), graph_inputs=graph_inputs,
                graph_outputs=graph_outputs, input_metadata=info.inputs)


def check_program_signature(info, program) -> dict:
    """Check the declared features interface locally; retain its original kind."""
    expected = [dict(name=item['name'], kind='cipher', shape=list(item['shape']))
                for item in info.inputs]
    actual = [dict(name=item.name, kind=item.kind, shape=list(item.shape))
              for item in program.params]
    passed = (len(expected) == len(actual) == 1 and expected == actual
              and expected[0]['shape'] == [1, 3, 32, 32]
              and [item['name'] for item in expected] == list(info.graph.inputs)
              and len(program.outputs) == len(info.graph.outputs) == 1)
    return dict(passed=passed, expected=expected, actual=actual,
                output_count=len(program.outputs), expected_output_count=len(info.graph.outputs))


def _json_array(array):
    import numpy as np
    return np.where(np.isfinite(array), array, None).tolist()


def compare_reference(graph, inputs, r13_probes) -> dict:
    """Compare three NumPy forwards with the saved original-model reference fields."""
    import numpy as np
    from ..analyzer import reference_forward

    if len(graph.inputs) != 1 or len(graph.outputs) != 1 or len(r13_probes) != 3:
        raise ValueError('R14 requires the original single I/O and three R13 reference records')
    name, output_name = graph.inputs[0], graph.outputs[0]
    if set(inputs) != {name} or inputs[name].shape != (3, 3, 32, 32):
        raise ValueError('The original three training inputs are required')
    observations = []
    recorder = getattr(inputs, 'recorder', None)
    for index, record in enumerate(r13_probes):
        if (record.get('probe_index') != index or record.get('input_name') != name
                or record.get('output_name') != output_name or 'reference' not in record):
            raise ValueError('R13 reference identity or order differs')
        reference = np.asarray(record['reference'], dtype=np.float64)
        context = recorder.span('reference_sample', probe_index=index) if recorder else _null_span()
        with context:
            output = np.asarray(reference_forward(graph, {name: inputs[name][index:index + 1]})[output_name],
                                dtype=np.float64)
        shape_ok = reference.shape == output.shape == (1, 10)
        finite = bool(np.isfinite(reference).all() and np.isfinite(output).all())
        difference = np.abs(output - reference) if shape_ok and finite else None
        passed = bool(difference is not None and
                      np.all(difference <= 1e-12 + 1e-12 * np.abs(reference)))
        observations.append(dict(probe_index=index, input_name=name, output_name=output_name,
                                 passed=passed, finite=finite, shape_ok=shape_ok,
                                 reference_shape=list(reference.shape), output_shape=list(output.shape),
                                 reference=_json_array(reference), output=_json_array(output),
                                 max_absolute_error=float(difference.max()) if difference is not None else None))
        if not passed:
            break
    return dict(passed=len(observations) == 3 and all(item['passed'] for item in observations),
                criterion=GRAPH_CRITERION, samples=observations)


@contextmanager
def _null_span():
    yield


class _ObservedInputs(dict):
    pass


def _case_size(directory):
    return sum(path.stat().st_size for path in Path(directory).rglob('*') if path.is_file())


def _observe_budget(args, result, stage, boundary, *, enforce=True):
    size = _case_size(args.work_dir.parent)
    result['case_artifact_bytes'] = size
    result['case_budget_observations'].append(dict(stage=stage, boundary=boundary, bytes=size,
                                                  limit_bytes=CASE_BYTES, within_budget=size <= CASE_BYTES))
    if enforce and size > CASE_BYTES:
        raise WorkerFailure('The R14 case exceeded 4 GiB of saved artifacts', status='artifact_budget_exceeded')


@contextmanager
def _phase(args, recorder, result, name):
    _observe_budget(args, result, name, 'before')
    write_json(args.work_dir / 'worker_result.json', result)
    try:
        with recorder.span(name):
            yield
    except BaseException:
        _observe_budget(args, result, name, 'after', enforce=False)
        write_json(args.work_dir / 'worker_result.json', result)
        raise
    else:
        _observe_budget(args, result, name, 'after')
        write_json(args.work_dir / 'worker_result.json', result)


def _read_calibration(args, info):
    import numpy as np
    from ..data import cifar
    from ..data.samples import LegacyCifar

    if Path(args.inputs).resolve() != cifar.RECORDS.resolve() or Path(args.selections).resolve() != cifar.SELECTIONS.resolve():
        raise WorkerFailure('The argv paths must identify the existing frozen CIFAR source', status='source_failed')
    metadata = json.loads((Path(args.inputs) / 'dataset.json').read_text())
    expected = dict(format='poseidon-cifar10-binary-records-v1', record_bytes=3073,
                    row_label_byte=0, preprocessing_applied=False, train_count=50000,
                    test_count=10000, pixel_layout='uint8 planar RGB 3x32x32, C order')
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise WorkerFailure('CIFAR dataset metadata differs', status='source_failed')
    entries = json.loads(Path(args.selections).read_text()).get('calibration')
    if not isinstance(entries, list) or len(entries) != 1000:
        raise WorkerFailure('Expected exactly 1000 frozen calibration selections', status='source_failed')
    clean = []
    for entry in entries:
        index, label = entry.get('index'), entry.get('label')
        if (type(index) is not int or not 0 <= index < 50000 or type(label) is not int
                or not 0 <= label < 10 or entry.get('shard') != f'train-{index // 1000 * 1000:05d}.records'
                or entry.get('row') != index % 1000):
            raise WorkerFailure('Calibration must identify original training rows and labels', status='source_failed')
        clean.append(dict(index=index, label=label, shard=entry['shard'], row=entry['row']))
    if len({item['index'] for item in clean}) != 1000:
        raise WorkerFailure('Frozen calibration indices must be unique', status='source_failed')
    normalization = info.source['normalization']
    arrays, labels = LegacyCifar(info.graph, normalization=(normalization['mean'], normalization['std'])).get('calibration')
    name = info.graph.inputs[0]
    if (set(arrays) != {name} or arrays[name].shape != (1000, 3, 32, 32)
            or arrays[name].dtype != np.dtype('float64') or not np.isfinite(arrays[name]).all()
            or labels.shape != (1000,) or any(int(label) != entry['label'] for label, entry in zip(labels, clean))):
        raise WorkerFailure('Original calibration arrays or labels differ', status='source_failed')
    source = dict(records=str(Path(args.inputs).resolve()), selections=str(Path(args.selections).resolve()),
                  split='train', entries=clean, indices=[item['index'] for item in clean],
                  sample_ids=['train:' + str(item['index']) for item in clean],
                  normalization=normalization, count=1000, batch=1, dataset_metadata=expected)
    return arrays, labels, source


def _check_calibration(graph, calibration):
    from ..analyzer import approximation_points
    if not calibration:
        raise WorkerFailure('Calibration returned no activation observations')
    for name, row in calibration.items():
        if (not isinstance(row, dict) or any(key not in row for key in ('min', 'max', 'absmax', 'p999', 'count'))
                or any(not math.isfinite(float(row[key])) for key in ('min', 'max', 'absmax', 'p999'))
                or row['min'] > row['max'] or type(row['count']) is not int or row['count'] <= 0):
            raise WorkerFailure('Invalid calibration observation: ' + name)
    if any(name not in calibration for _, name in approximation_points(graph)):
        raise WorkerFailure('Calibration is missing a rule approximation point')


def _emit_budget(text, constants):
    array_bytes = sum(array.nbytes for array in constants.values())
    size = len(text.encode('utf-8')) + array_bytes + 1024 * len(constants) + 1024**2
    return dict(text_bytes=len(text.encode('utf-8')), array_bytes=array_bytes,
                entry_overhead_bytes=1024 * len(constants), global_overhead_bytes=1024**2,
                estimated_bytes=size, limit_bytes=EMIT_BYTES, within_budget=size <= EMIT_BYTES)


def _readback_emit(directory, constants):
    import numpy as np
    text = (directory / 'candidate_trace.py').read_text()
    ast.parse(text)
    rows = []
    with np.load(directory / 'constants.npz', allow_pickle=False) as archive:
        if set(archive.files) != set(constants) or len(archive.files) != len(constants):
            raise WorkerFailure('Emitted constant names differ')
        for name in archive.files:
            array = archive[name]
            original = constants[name]
            finite = bool(np.isfinite(array).all())
            row = dict(name=name, dtype=str(array.dtype), shape=list(array.shape), finite=finite,
                       bytes=int(array.nbytes))
            rows.append(row)
            if array.dtype != original.dtype or array.shape != original.shape or not finite:
                raise WorkerFailure('Emitted constant metadata or finite-value check failed: ' + name)
            del array
    return dict(source_parsed=True, source_executed=False, constants=rows,
                source_bytes=(directory / 'candidate_trace.py').stat().st_size,
                constants_bytes=(directory / 'constants.npz').stat().st_size)


def run(args, recorder, result) -> None:
    import numpy as np
    from ..analyzer import calibrate
    from ..backends import CountBackend, EmitBackend
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from ..kernels.layout import layout_record
    from ..lower import lower
    from ..models.info import ModelInfo
    from ..planners.rule import plan

    def save(name, value):
        path = args.work_dir / (name + '.json')
        write_json(path, value)
        result['artifacts'][name] = str(path)

    with _phase(args, recorder, result, 'load_saved_model'):
        r13 = json.loads(Path(args.r13_result).read_text())
        artifact = (r13.get('worker') or {}).get('artifacts') or {}
        for field, actual in (('model_info', args.model_info), ('training_probes', args.r13_probes),
                              ('training_probe_source', args.r13_probe_source)):
            if field not in artifact or Path(artifact[field]).resolve() != Path(actual).resolve():
                raise WorkerFailure('R13 artifact path differs: ' + field, status='source_failed')
        info = ModelInfo.load(args.model_info)
        result['model'] = r13.get('model')
        source_check = check_saved_source(info, r13)
        save('source_check', source_check)
        result['source_checked'] = source_check['passed']
        if not source_check['passed']:
            raise WorkerFailure('Saved R13 source checks failed', status='source_failed')
        result['model_source'] = info.source
        result['input_metadata'] = info.inputs
        result['split'] = info.split
        result['assumptions'] = info.assumptions
        result['task'] = info.task
        result['metric'] = info.metric
        result['threshold'] = info.threshold
        result['r13_source_result'] = str(Path(args.r13_result).resolve())
        result['r13_timing'] = r13.get('timing')
    with _phase(args, recorder, result, 'read_calibration_inputs'):
        arrays, labels, source = _read_calibration(args, info)
        save('calibration_source', source)
        result['calibration_source'] = source
        original_source = json.loads(Path(args.r13_probe_source).read_text())
        saved_source = info.source.get('calibration_source')
        if original_source != saved_source or original_source != r13['worker'].get('source'):
            raise WorkerFailure('R13 training probe provenance differs', status='source_failed')
        if (original_source.get('indices') != source['indices'][:3]
                or original_source.get('sample_ids') != source['sample_ids'][:3]
                or original_source.get('labels') != [int(x) for x in labels[:3]]):
            raise WorkerFailure('Original probe IDs differ from the calibration prefix', status='source_failed')
        with np.load(args.r13_probes, allow_pickle=False) as archive:
            if set(archive.files) != {'input', 'labels', 'indices'}:
                raise WorkerFailure('R13 probe archive fields differ', status='source_failed')
            probe_inputs = np.array(archive['input'])
            if (probe_inputs.dtype != np.dtype('float64')
                    or not np.array_equal(probe_inputs, arrays[info.graph.inputs[0]][:3])
                    or not np.array_equal(archive['labels'], labels[:3])
                    or not np.array_equal(archive['indices'], source['indices'][:3])):
                raise WorkerFailure('Original probe tensors differ from the calibration prefix', status='source_failed')
    with _phase(args, recorder, result, 'reference_check'):
        observed_inputs = _ObservedInputs({info.graph.inputs[0]: probe_inputs})
        observed_inputs.recorder = recorder
        references = r13['worker']['graph_math']['samples']
        observation = compare_reference(info.graph, observed_inputs, references)
        result['reference_passed'] = observation['passed']
        save('reference_check', observation)
        if not observation['passed']:
            raise WorkerFailure('NumPy reference differs from the saved original model', status='math_failed')
        del observed_inputs, probe_inputs
    with _phase(args, recorder, result, 'calibrate'):
        result['calibration_attempts'] = 1
        write_json(args.work_dir / 'worker_result.json', result)
        calibration = calibrate(info.graph, arrays, batch=1)
        _check_calibration(info.graph, calibration)
        save('calibration', calibration)
        result['calibration_completed'] = True
        result['calibration_summary'] = dict(count=1000, batch=1,
            p999='maximum of each batch 99.9th absolute percentile; batch=1', fields=list(next(iter(calibration.values()))))
        del arrays, labels
    with _phase(args, recorder, result, 'rule_plan'):
        result['plan_attempts'] = 1
        write_json(args.work_dir / 'worker_result.json', result)
        text = plan(info.graph, calibration, **RULE_OPTIONS)
        program_path = args.work_dir / 'program.ptdsl'
        program_path.write_text(text)
        result['artifacts']['program'] = str(program_path)
        result['rule_options'] = dict(RULE_OPTIONS)
        result['plan_completed'] = True
    with _phase(args, recorder, result, 'parse_typecheck'):
        program = parse(text)
        checked = check(program, {name: value.shape for name, value in info.graph.weights.items()}, graph=info.graph)
        save('typecheck', dict(ok=checked.ok, diags=[asdict(diag) for diag in checked.diags]))
        result['typecheck_passed'] = checked.ok
        if not checked.ok:
            raise WorkerFailure('The rule program failed typechecking', status='typecheck_failed')
    with _phase(args, recorder, result, 'signature_check'):
        signature = check_program_signature(info, program)
        save('signature', signature)
        result['signature_passed'] = signature['passed']
        if not signature['passed']:
            raise WorkerFailure('The rule program changed the original signature', status='signature_failed')
    with _phase(args, recorder, result, 'count_lower'):
        result['count_attempts'] = 1
        write_json(args.work_dir / 'worker_result.json', result)
        counter = CountBackend(program.header.slots)
        count_values, layouts = lower(program, info.graph.weights, counter, act_mode='program')
        count = counter.stats()
        result['count_stats'] = count
        save('count', dict(stats=count, output_layouts=[layout_record(item) for item in layouts]))
        result['count_completed'] = True
        del count_values, counter
        gc.collect()
    emit_dir = args.work_dir / 'emit'
    with _phase(args, recorder, result, 'emit_lower'):
        result['emit_attempts'] = 1
        write_json(args.work_dir / 'worker_result.json', result)
        emitter = EmitBackend(program.header.slots, frontend=args.frontend,
                              constants=emit_dir / 'constants.npz', save_dir=emit_dir)
        lower(program, info.graph.weights, emitter, act_mode='program')
        emitted = emitter.stats()
        result['emit_stats'] = emitted
        differences = {key: dict(count=count[key], emit=emitted[key])
                       for key in ('counts', 'ops_total', 'distinct_consts') if count[key] != emitted[key]}
        result['stats_equal'] = not differences
        save('emit_stats', dict(stats=emitted, differences=differences, matched=not differences))
        if differences:
            raise WorkerFailure('Count and Emit operation statistics differ', status='count_emit_mismatch')
    with _phase(args, recorder, result, 'emit_write'):
        emitted_text, constants = emitter.render()
        budget = _emit_budget(emitted_text, constants)
        result['emit_budget'] = budget
        save('emit_budget', budget)
        if not budget['within_budget']:
            raise WorkerFailure('The Emit artifact estimate exceeds 3 GiB', status='artifact_budget_exceeded')
        del emitted_text
        emitter.write(emit_dir)
        result['artifacts'].update(trace_source=str(emit_dir / 'candidate_trace.py'),
                                   constants=str(emit_dir / 'constants.npz'))
    with _phase(args, recorder, result, 'metadata_readback'):
        metadata = _readback_emit(emit_dir, constants)
        save('emit_metadata', metadata)
        result['emit_completed'] = True
        result['lowering_ready'] = True
        del constants, emitter
        gc.collect()


def _memory_error(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, MemoryError):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-info', 'r13-result', 'r13-probes', 'r13-probe-source', 'inputs',
                 'selections', 'frontend', 'work-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.work_dir / 'worker_result.json'
    if result_path.exists():
        raise FileExistsError('Existing R14 worker results are preserved')
    result = dict(round='R14', model=None, status='error', error=None, failed_phase=None,
                  pid=os.getpid(), ppid=os.getppid(), artifacts={}, case_budget_observations=[],
                  case_artifact_bytes=None, count_stats=None, emit_stats=None,
                  compiled_bootstraps=None, predicted_latency=None, CKKS_latency=None,
                  accuracy_acceptance=None, llm_calls=0, real_ckks_attempts=0,
                  real_ckks_completed=0, trace_runs=0, compiler_runs=0, keygen_runs=0,
                  model_imports=0, downloads=0, installations=0,
                  calibration_attempts=0, plan_attempts=0, count_attempts=0, emit_attempts=0,
                  **dict.fromkeys(REQUIRED_STAGES))
    recorder = StageRecorder(args.work_dir / 'events.jsonl')
    old_handler = None
    try:
        def timeout(signum, frame):
            raise WorkerFailure('The R14 worker exceeded 1800 seconds', status='timeout')
        old_handler = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(INTERNAL_SECONDS)
        with recorder.span('worker'):
            run(args, recorder, result)
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = ('resource_limit' if _memory_error(exc) else
                            'dependency_missing' if isinstance(exc, (ModuleNotFoundError, FileNotFoundError)) else
                            getattr(exc, 'status', 'error'))
        result['error'] = dict(type=type(exc).__name__, message=str(exc)[:2000])
        errors = [event for event in read_events(args.work_dir / 'events.jsonl')
                  if event.get('type') == 'end' and event.get('status') == 'error']
        result['failed_phase'] = errors[0]['stage_id'] if errors else 'worker'
    finally:
        if old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        _observe_budget(args, result, 'worker', 'terminal', enforce=False)
        if result['case_artifact_bytes'] > CASE_BYTES:
            result.update(status='artifact_budget_exceeded',
                          error=dict(type='WorkerFailure', message='The R14 case exceeded 4 GiB'))
        result['resource_ru_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(result_path, result)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
