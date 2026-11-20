"""R17: one fixed margin candidate from saved calibration, then heldout checks."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import resource
import signal

from . import r14_common_worker as original
from . import r15_rule_validation_worker as validation
from .r10_paper_metrics import StageRecorder, read_events, write_json


INTERNAL_SECONDS = 1800
CASE_BYTES = 4 * 1024**3
SAMPLE_COUNT = 1000
PROBE_COUNT = 3
DOMAIN_LIMIT = 0.90
GRAPH_CRITERION = '1e-12+1e-12*abs(reference)'
RULE_OPTIONS = dict(deg=63, margin=1.50, slots=16384, placement='depth-dp',
                    refresh='compiler', act_method='cheb')
ORIGINAL_OPTIONS = dict(deg=63, margin=1.25, slots=16384, placement='depth-dp',
                        refresh='compiler', act_method='cheb')
MODEL_RECORDS = {
    'pt_agent.experiments.r13_common_models:resnet20_silu': dict(
        r14_run='r14-common-resnet20-silu-20261115', act_count=19, calibration_rows=74),
    'pt_agent.experiments.r13_common_models:vgg16_silu_avgpool': dict(
        r14_run='r14-common-vgg16-silu-avgpool-20261115', act_count=15, calibration_rows=50),
    'pt_agent.experiments.r13_common_models:alexnet_silu_avgpool': dict(
        r14_run='r14-common-alexnet-silu-avgpool-20261115', act_count=7, calibration_rows=24),
}
STAGE_FIELDS = ('source_checked', 'calibration_reused', 'plan_completed', 'typecheck_passed',
                'signature_passed', 'candidate_checked', 'count_completed',
                'samples_checked', 'math_probes_passed')
COUNT_FIELDS = ('plan_attempts', 'plan_completions', 'count_attempts', 'count_completions',
                'model_restore_attempts', 'model_restore_completions')
ZERO_FIELDS = ('llm_calls', 'real_ckks_attempts', 'real_ckks_completed', 'trace_runs',
               'compiler_runs', 'keygen_runs', 'model_imports', 'downloads', 'installations',
               'calibration_attempts', 'emit_attempts', 'training_samples', 'confirm_samples',
               'report_samples', 'freeze_attempts', 'agent_runs', 'tuner_runs', 'cost_estimates')
UNMEASURED_FIELDS = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance')
ENGINE_NAMES = validation.ENGINE_NAMES
CACHE_NAMES = ('r14_result', 'r13_result', 'calibration', 'calibration_source', 'original_program')
RESULT_FIELDS = ('evaluation_completed', 'completed_samples', 'heldout_samples', 'outputs_partial',
                 'task_passed', 'domain_passed', 'heldout_acceptance', 'all_finite',
                 'candidate_frozen', 'max_u', 'max_logit_absolute_difference', 'validation_metrics',
                 'source_count_stats', 'count_stats', 'domain_rows', 'engine_counts',
                 'cached_source_reads')
PRIOR_STAGES = ('source_checked', 'reference_passed', 'calibration_completed', 'plan_completed',
                'typecheck_passed', 'signature_passed', 'count_completed', 'stats_equal')
WorkerFailure = original.WorkerFailure
check_math = validation.check_math
_read_heldout = validation._read_heldout
_packed_logits = validation._packed_logits
_domain_rows = validation._domain_rows
_tensor_observer = validation._tensor_observer
_require_logits = validation._require_logits
_torch_initialized = False


def _source_error(message):
    return WorkerFailure(message, status='source_failed')


def validate_source(r14_result, r13_result, info, calibration, calibration_source) -> dict:
    errors = []
    if not all(isinstance(record, dict) for record in (r14_result, r13_result, calibration, calibration_source)):
        return dict(passed=False, errors=['Saved source records must be objects'], model=None)
    model = r14_result.get('model')
    expected = MODEL_RECORDS.get(model)
    worker = r14_result.get('worker') or {}
    if not isinstance(worker, dict):
        return dict(passed=False, errors=['The saved R14 worker must be an object'], model=model)
    if (expected is None or r14_result.get('round') != 'R14' or worker.get('round') != 'R14'
            or r14_result.get('mode') != 'common-model-rule-preparation'
            or worker.get('model') != model or r13_result.get('model') != model
            or any(worker.get(key) is not True for key in PRIOR_STAGES)):
        errors.append('The original R14 model or completed preparation stages differ')
    if expected is not None:
        resnet = expected['r14_run'] == 'r14-common-resnet20-silu-20261115'
        wanted = 'ok' if resnet else 'artifact_budget_exceeded'
        if r14_result.get('status') != wanted or worker.get('status') != wanted:
            errors.append('The original R14 retained status differs')
        if not resnet and worker.get('failed_phase') != 'emit_write':
            errors.append('The original VGG/Alex stop must be emit_write')
        if len(calibration) != expected['calibration_rows']:
            errors.append('The saved calibration row count differs')
    if worker.get('rule_options') != ORIGINAL_OPTIONS or original.RULE_OPTIONS != ORIGINAL_OPTIONS:
        errors.append('The original six fixed rule options differ')
    checked = original.check_saved_source(info, r13_result)
    if not checked['passed']:
        errors.extend(checked['errors'])
    if info.task != 'classification' or info.metric != 'classification_top1' or info.threshold != {'acc_drop_pp': 0.5}:
        errors.append('The original classification policy differs')
    try:
        original._check_calibration(info.graph, calibration)
    except (WorkerFailure, TypeError, ValueError, KeyError, OverflowError) as exc:
        errors.append('Invalid saved calibration: ' + str(exc))
    if calibration_source != worker.get('calibration_source'):
        errors.append('Calibration provenance differs from the independent R14 observation')
    entries = calibration_source.get('entries')
    if (calibration_source.get('split') != 'train' or calibration_source.get('count') != 1000
            or type(calibration_source.get('count')) is not int
            or calibration_source.get('batch') != 1 or type(calibration_source.get('batch')) is not int
            or not isinstance(entries, list) or len(entries) != 1000):
        errors.append('Expected the saved 1000 training selections and batch one')
        entries = []
    indices, sample_ids = [], []
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append('A saved training selection is not an object')
            continue
        index, label = entry.get('index'), entry.get('label')
        if (type(index) is not int or not 0 <= index < 50000 or type(label) is not int
                or not 0 <= label < 10):
            errors.append('Saved calibration index or label differs')
            continue
        if entry.get('shard') != f'train-{index // 1000 * 1000:05d}.records' or entry.get('row') != index % 1000:
            errors.append('Saved calibration shard or row differs')
        indices.append(index)
        sample_ids.append('train:' + str(index))
    if len(set(indices)) != 1000 or calibration_source.get('indices') != indices or calibration_source.get('sample_ids') != sample_ids:
        errors.append('Saved calibration indices or ordered sample IDs differ')
    if calibration_source.get('normalization') != info.source.get('normalization'):
        errors.append('Saved calibration normalization differs')
    dataset = dict(format='poseidon-cifar10-binary-records-v1', record_bytes=3073,
                   row_label_byte=0, preprocessing_applied=False, train_count=50000,
                   test_count=10000, pixel_layout='uint8 planar RGB 3x32x32, C order')
    if calibration_source.get('dataset_metadata') != dataset:
        errors.append('Saved CIFAR dataset metadata differs')
    summary = dict(count=1000, batch=1,
                   p999='maximum of each batch 99.9th absolute percentile; batch=1',
                   fields=['min', 'max', 'absmax', 'p999', 'count'])
    if worker.get('calibration_summary') != summary:
        errors.append('The saved calibration summary differs')
    stats = worker.get('count_stats')
    if not isinstance(stats, dict) or any(key not in stats for key in ('counts', 'ops_total', 'distinct_consts')):
        errors.append('The saved original Count statistics are missing')
    return dict(passed=not errors, model=model, errors=errors, saved_source=checked,
                source_status=r14_result.get('status'),
                prior_stages={key: worker.get(key) for key in PRIOR_STAGES},
                calibration_rows=len(calibration), calibration_summary=worker.get('calibration_summary'),
                policy=dict(task=info.task, metric=info.metric, threshold=info.threshold))


def _without_positions(record):
    return {key: value for key, value in asdict(record).items() if key not in ('lineno', 'call_lines')}


def check_candidate(info, original_program, candidate, calibration) -> dict:
    errors, mappings = [], []
    for label, program in (('original', original_program), ('candidate', candidate)):
        strategy = validation._check_program(info, program)
        if not strategy['passed']:
            errors.extend(label + ': ' + message for message in strategy['errors'])
    if (original_program.name != candidate.name
            or _without_positions(original_program.header) != _without_positions(candidate.header)
            or [_without_positions(item) for item in original_program.params] != [_without_positions(item) for item in candidate.params]
            or original_program.outputs != candidate.outputs
            or {key: _without_positions(value) for key, value in original_program.regions.items()}
            != {key: _without_positions(value) for key, value in candidate.regions.items()}):
        errors.append('Candidate program signature, header or region semantics changed')
    if len(original_program.ops) != len(candidate.ops):
        errors.append('Candidate operation count changed')
    for before, after in zip(original_program.ops, candidate.ops):
        left, right = _without_positions(before), _without_positions(after)
        if before.kind == 'act' and after.kind == 'act':
            left['attrs'].pop('domain', None)
            right['attrs'].pop('domain', None)
        if left != right:
            errors.append('Candidate changed operation semantics at op ' + str(before.id))
    points = [node for node in info.graph.nodes if node.op == 'act']
    old_acts = [op for op in original_program.ops if op.kind == 'act']
    new_acts = [op for op in candidate.ops if op.kind == 'act']
    expected = MODEL_RECORDS.get(info.source.get('factory'))
    if expected is None or not len(points) == len(old_acts) == len(new_acts) == expected['act_count']:
        errors.append('Original graph and program activation counts differ')
    for node, before, after in zip(points, old_acts, new_acts):
        name = node.inputs[0]
        row = calibration.get(name)
        if not isinstance(row, dict):
            errors.append('Missing calibration graph tensor: ' + name)
            continue
        try:
            # Preserve precisely the planner's endpoint-multiplication order.
            old_domain = (float(row['min']) * 1.25, float(row['max']) * 1.25)
            new_domain = (float(row['min']) * 1.50, float(row['max']) * 1.50)
        except (KeyError, TypeError, ValueError, OverflowError):
            errors.append('Invalid activation calibration: ' + name)
            continue
        if (node.attrs.get('fn') != 'silu' or before.attrs.get('domain') != old_domain
                or after.attrs.get('domain') != new_domain):
            errors.append('Candidate or original domain differs from the fixed cached formula: ' + name)
        mappings.append(dict(graph_node_id=node.id, graph_tensor=name, original_op_id=before.id,
                             candidate_op_id=after.id, old_domain=list(old_domain), new_domain=list(new_domain),
                             original_lineno=before.lineno, candidate_lineno=after.lineno,
                             original_call_lines=list(before.call_lines), candidate_call_lines=list(after.call_lines)))
    return dict(passed=not errors, errors=errors, act_count=len(new_acts), mappings=mappings,
                original_header_lineno=original_program.header.lineno,
                candidate_header_lineno=candidate.header.lineno)


def _source_paths(args, r14, r13):
    rule_dir, r13_dir = args.r14_result.resolve().parent, args.r13_result.resolve().parent
    worker = r14.get('worker') or {}
    artifacts = worker.get('artifacts') or {}

    def require(value, actual, directory, label):
        if (not isinstance(value, str) or Path(value).resolve() != actual.resolve()
                or not actual.resolve().is_relative_to(directory)):
            raise _source_error('Saved source path differs: ' + label)

    for key in ('calibration', 'calibration_source', 'original_program'):
        require(artifacts.get('program' if key == 'original_program' else key),
                getattr(args, key), rule_dir, key)
    source_r13 = r14.get('source_r13') or {}
    require(source_r13.get('result'), args.r13_result, r13_dir, 'source_r13.result')
    require(worker.get('r13_source_result'), args.r13_result, r13_dir, 'worker.r13_source_result')
    require((source_r13.get('artifacts') or {}).get('model_info'), args.model_info, r13_dir, 'source_r13.model_info')
    require((((r13.get('worker') or {}).get('artifacts')) or {}).get('model_info'),
            args.model_info, r13_dir, 'r13.artifacts.model_info')
    if source_r13.get('run') != str(r13_dir):
        raise _source_error('The saved R13 run path differs')
    for key in CACHE_NAMES:
        if not getattr(args, key).is_file():
            raise FileNotFoundError('Missing saved source: ' + str(getattr(args, key)))
    if not args.model_info.is_dir():
        raise FileNotFoundError('The original model_info directory is missing')
    return dict(**{key: str(getattr(args, key).resolve()) for key in CACHE_NAMES},
                model_info=str(args.model_info.resolve()),
                r13_timing=r13.get('timing'), r14_timing=r14.get('timing'),
                included_in_current_turnaround=False)


def _read_json_source(args, result, key):
    counts = result['cached_source_reads'][key]
    counts['attempted'] += 1
    write_json(args.work_dir / 'worker_result.json', result)
    value = json.loads(getattr(args, key).read_text())
    if not isinstance(value, dict):
        raise _source_error('Saved JSON must be an object: ' + key)
    counts['completed'] += 1
    return value


def _read_program_source(args, result):
    counts = result['cached_source_reads']['original_program']
    counts['attempted'] += 1
    write_json(args.work_dir / 'worker_result.json', result)
    text = args.original_program.read_text()
    counts['completed'] += 1
    return text


def _plan_candidate(info, calibration):
    from ..planners.rule import plan

    return plan(info.graph, calibration, **RULE_OPTIONS)


def _count_candidate(program, weights):
    from ..backends import CountBackend
    from ..kernels.layout import layout_record
    from ..lower import lower

    backend = CountBackend(slots=program.header.slots)
    values = layouts = None
    try:
        values, layouts = lower(program, weights, backend, act_mode='program')
        return dict(stats=backend.stats(), output_layouts=[layout_record(item) for item in layouts])
    finally:
        del values, layouts, backend
        gc.collect()


def _initialize_torch():
    global _torch_initialized
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    if not _torch_initialized:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _torch_initialized = True


def _observe_budget(args, result, stage, boundary, *, enforce=True):
    size = original._case_size(args.work_dir.parent)
    result['case_artifact_bytes'] = size
    result['case_budget_observations'].append(dict(stage=stage, boundary=boundary, bytes=size,
                                                  limit_bytes=CASE_BYTES, within_budget=size <= CASE_BYTES))
    if enforce and size > CASE_BYTES:
        raise WorkerFailure('The R17 case exceeded 4 GiB of artifacts', status='artifact_budget_exceeded')


@contextmanager
def _phase(args, recorder, result, name, **metadata):
    _observe_budget(args, result, name, 'before')
    write_json(args.work_dir / 'worker_result.json', result)
    try:
        with recorder.span(name, **metadata):
            yield
    except BaseException:
        _observe_budget(args, result, name, 'after', enforce=False)
        write_json(args.work_dir / 'worker_result.json', result)
        raise
    else:
        _observe_budget(args, result, name, 'after')
        write_json(args.work_dir / 'worker_result.json', result)


def _call_engine(args, recorder, result, engine, sample_id, function):
    return validation._call_engine(args, recorder, result, engine, sample_id, function)


def run_validation(args, recorder, result) -> None:
    # Set environment and Torch thread limits before importing numerical modules.
    with _phase(args, recorder, result, 'initialize_threads'):
        _initialize_torch()
    import numpy as np
    from ..analyzer import reference_forward
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from ..metrics import compare, passes
    from ..models.info import ModelInfo
    from ..tensor_sim import run_tensor

    def save(name, record):
        path = args.work_dir / (name + '.json')
        write_json(path, record)
        result['artifacts'][name] = str(path)

    with _phase(args, recorder, result, 'load_saved_source'):
        r14 = _read_json_source(args, result, 'r14_result')
        r13 = _read_json_source(args, result, 'r13_result')
        result['model'] = r14.get('model')
        source_paths = _source_paths(args, r14, r13)
        result['model_restore_attempts'] += 1
        write_json(args.work_dir / 'worker_result.json', result)
        info = ModelInfo.load(args.model_info)
        result['model_restore_completions'] += 1
        calibration = _read_json_source(args, result, 'calibration')
        calibration_source = _read_json_source(args, result, 'calibration_source')
        original_text = _read_program_source(args, result)
        source = validate_source(r14, r13, info, calibration, calibration_source)
        save('source_check', source)
        result['source_checked'] = source['passed']
        if not source['passed']:
            raise _source_error('Saved calibration source failed: ' + '; '.join(source['errors']))
        result['calibration_reused'] = True
        result['calibration_summary'] = r14['worker']['calibration_summary']
        result['source_count_stats'] = r14['worker']['count_stats']
        result['source_preparation'] = source_paths
        result['source_status'] = source['source_status']
        for key in ('source', 'inputs', 'split', 'assumptions', 'task', 'metric', 'threshold'):
            result[{'source': 'model_source', 'inputs': 'input_metadata'}.get(key, key)] = getattr(info, key)
    with _phase(args, recorder, result, 'parse_original'):
        original_program = parse(original_text)
        del original_text
        strategy = validation._check_program(info, original_program)
        if not strategy['passed']:
            raise _source_error('Original rule strategy differs: ' + '; '.join(strategy['errors']))
    with _phase(args, recorder, result, 'plan_candidate'):
        result['plan_attempts'] += 1
        write_json(args.work_dir / 'worker_result.json', result)
        text = _plan_candidate(info, calibration)
        result['plan_completions'] += 1
        program_path = args.work_dir / 'program.ptdsl'
        program_path.write_text(text)
        result['artifacts']['program'] = str(program_path)
        result['rule_options'] = dict(RULE_OPTIONS)
        result['plan_completed'] = True
    with _phase(args, recorder, result, 'parse_typecheck'):
        program = parse(text)
        del text
        checked = check(program, {name: value.shape for name, value in info.graph.weights.items()}, graph=info.graph)
        save('typecheck', dict(ok=checked.ok, diags=[asdict(item) for item in checked.diags]))
        result['typecheck_passed'] = checked.ok
        if not checked.ok:
            raise WorkerFailure('Candidate typechecking failed', status='typecheck_failed')
    with _phase(args, recorder, result, 'signature_and_candidate_check'):
        signature = original.check_program_signature(info, program)
        save('signature', signature)
        result['signature_passed'] = signature['passed']
        if not signature['passed']:
            raise WorkerFailure('Candidate changed the original signature', status='signature_failed')
        candidate = check_candidate(info, original_program, program, calibration)
        save('candidate_check', candidate)
        result['candidate_checked'] = candidate['passed']
        if not candidate['passed']:
            raise WorkerFailure('Candidate has an unexpected structural change: ' + '; '.join(candidate['errors']),
                                status='candidate_failed')
        del original_program, calibration
    with _phase(args, recorder, result, 'count_candidate'):
        result['count_attempts'] += 1
        write_json(args.work_dir / 'worker_result.json', result)
        count = _count_candidate(program, info.graph.weights)
        result['count_completions'] += 1
        result['count_stats'] = count['stats']
        save('count', count)
        before = result['source_count_stats']
        after = result['count_stats']
        differences = {key: dict(source=before.get(key), candidate=after.get(key))
                       for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)}
        save('count_differences', dict(differences=differences, diagnostic_only=True))
        result['count_completed'] = True
        del count
        gc.collect()
    with _phase(args, recorder, result, 'read_heldout'):
        arrays, labels, sample_source = _read_heldout(args, info)
        save('sample_source', sample_source)
        result['sample_source'] = sample_source
        result['samples_checked'] = True
    exact = np.empty((1000, 10), dtype=np.float64)
    approximate = np.empty_like(exact)
    rows = _domain_rows(program)
    result['domain_rows'] = rows
    probe_path, samples_path = args.work_dir / 'math_probes.jsonl', args.work_dir / 'samples.jsonl'
    probe_path.touch(exist_ok=False)
    samples_path.touch(exist_ok=False)
    result['artifacts'].update(math_probes=str(probe_path), samples=str(samples_path))
    name, output_name = info.graph.inputs[0], info.graph.outputs[0]

    def pair(reference, observed, sample_id, pair_name):
        with recorder.span('math_check', sample_id=sample_id, pair=pair_name):
            checked = check_math(reference, observed)
            record = dict(sample_id=sample_id, pair=pair_name, reference=reference.tolist(),
                          observed=observed.tolist(), **checked)
            result['math_probes'].append(record)
            validation._append_record(probe_path, record)
            if not checked['passed']:
                result['math_probes_passed'] = False
                raise WorkerFailure('Strict mathematical probe failed: ' + pair_name + ' at ' + sample_id,
                                    status='math_failed')

    for index in range(1000):
        sample_id = 'test:' + str(9000 + index)
        image = arrays[name][index:index + 1]
        with _phase(args, recorder, result, 'sample', sample_id=sample_id, sample_index=index):
            reference = _call_engine(args, recorder, result, 'reference_forward', sample_id,
                lambda: reference_forward(info.graph, {name: image}, keep=False)[output_name])
            if index < 3:
                tensor_exact = _call_engine(args, recorder, result, 'tensor_exact', sample_id,
                    lambda: run_tensor(program, info.graph.weights, image, cache=False, act_mode='exact'))
                pair(reference, tensor_exact, sample_id, 'tensor_exact_vs_reference')
                del tensor_exact
                mock_exact = _call_engine(args, recorder, result, 'mock_exact', sample_id,
                    lambda: _packed_logits(program, info.graph.weights, image, act_mode='exact'))
                pair(reference, mock_exact, sample_id, 'mock_exact_vs_reference')
                del mock_exact
            observer, finish = _tensor_observer(program, sample_id, rows)
            actual = _call_engine(args, recorder, result, 'tensor_program', sample_id,
                lambda: run_tensor(program, info.graph.weights, image, cache=False,
                                   act_mode='program', observer=observer))
            with recorder.span('domain_check', sample_id=sample_id):
                finish()
            exact[index], approximate[index] = reference[0], actual[0]
            result['completed_samples'] += 1
            result['heldout_samples'] += 1
            validation._append_record(samples_path,
                dict(sample_id=sample_id, index=9000 + index, label=int(labels[index]),
                     reference=reference.tolist(), approximate=actual.tolist(), partial=True))
            if index < 3:
                mock_program = _call_engine(args, recorder, result, 'mock_program', sample_id,
                    lambda: _packed_logits(program, info.graph.weights, image, act_mode='program'))
                pair(mock_program, actual, sample_id, 'tensor_program_vs_mock_program')
                del mock_program
                if index == 2:
                    result['math_probes_passed'] = True
            del reference, actual, observer, finish
    with _phase(args, recorder, result, 'summarize_heldout'):
        measured = compare(dict(values=exact, task=info.task), approximate, labels)
        maximum = max((row['max_u'] for row in rows), default=0.0)
        covered = len(rows) == MODEL_RECORDS[result['model']]['act_count'] and all(row['count'] == 1000 for row in rows)
        with np.errstate(over='ignore', invalid='ignore'):
            difference = np.abs(approximate - exact)
        if not np.isfinite(difference).all():
            raise WorkerFailure('The logit-difference diagnostic is nonfinite', status='nonfinite')
        result.update(validation_metrics=measured, task_passed=bool(passes(measured, info.threshold)),
                      domain_passed=bool(covered and maximum <= DOMAIN_LIMIT), max_u=maximum,
                      all_finite=bool(np.isfinite(exact).all() and np.isfinite(approximate).all()),
                      max_logit_absolute_difference=float(difference.max()),
                      evaluation_completed=True, outputs_partial=False)
        result['heldout_acceptance'] = bool(result['math_probes_passed'] and result['task_passed']
            and result['domain_passed'] and result['all_finite'] and result['completed_samples'] == 1000)
        result['acc_drop_pp_unit'] = 'percentage points'
        save('heldout_metrics', dict(metrics=measured, task_passed=result['task_passed'],
            domain_passed=result['domain_passed'], max_u=maximum, heldout_acceptance=result['heldout_acceptance'],
            completed_samples=1000, acc_drop_pp_unit='percentage points',
            max_logit_absolute_difference=result['max_logit_absolute_difference']))
        save('domain_rows', dict(rows=rows, limit=DOMAIN_LIMIT, domain_passed=result['domain_passed'], partial=False))
        outputs = args.work_dir / 'heldout_outputs.npz'
        np.savez(outputs, reference=exact, approximate=approximate, labels=labels,
                 indices=np.arange(9000, 10000, dtype=np.int64))
        result['artifacts']['heldout_outputs'] = str(outputs)


def _initial_result():
    return dict(round='R17', model=None, status='error', error=None, failed_phase=None,
                pid=os.getpid(), ppid=os.getppid(), artifacts={}, case_budget_observations=[],
                case_artifact_bytes=None, completed_samples=0, heldout_samples=0,
                evaluation_completed=False, outputs_partial=True, task_passed=None,
                domain_passed=None, heldout_acceptance=None, all_finite=None,
                validation_metrics=None, max_logit_absolute_difference=None, max_u=None,
                domain_rows=None, source_count_stats=None, count_stats=None, math_probes=[],
                engine_counts={key: dict(attempted=0, completed=0) for key in ENGINE_NAMES},
                cached_source_reads={key: dict(attempted=0, completed=0) for key in CACHE_NAMES},
                candidate_frozen=False,
                evidence_scope='Planned: one fixed margin candidate, Count, 1000 heldout samples and three packed probes; actual counts separately recorded; CKKS 0',
                **dict.fromkeys(STAGE_FIELDS), **dict.fromkeys(COUNT_FIELDS, 0),
                **dict.fromkeys(ZERO_FIELDS, 0), **dict.fromkeys(UNMEASURED_FIELDS))


def _write_terminal(args, result):
    path = args.work_dir / 'worker_result.json'
    others = original._case_size(args.work_dir.parent) - (path.stat().st_size if path.exists() else 0)
    observation = dict(stage='worker', boundary='terminal', bytes=None, limit_bytes=CASE_BYTES, within_budget=None)
    result['case_budget_observations'].append(observation)
    for _ in range(16):
        size = others + len((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
        if size > CASE_BYTES:
            result.update(status='artifact_budget_exceeded',
                          error=dict(type='WorkerFailure', message='The R17 case exceeded 4 GiB'))
        before = result['case_artifact_bytes'], observation['bytes'], observation['within_budget']
        result['case_artifact_bytes'] = observation['bytes'] = size
        observation['within_budget'] = size <= CASE_BYTES
        if before == (size, size, size <= CASE_BYTES):
            write_json(path, result)
            return
    raise RuntimeError('R17 worker terminal artifact accounting did not settle')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('r14-result', 'r13-result', 'model-info', 'calibration', 'calibration-source',
                 'original-program', 'records', 'work-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if (args.work_dir / 'worker_result.json').exists():
        raise FileExistsError('Existing R17 worker results are preserved')
    result = _initial_result()
    recorder = StageRecorder(args.work_dir / 'events.jsonl')
    old_handler = None
    try:
        def timeout(signum, frame):
            raise WorkerFailure('The R17 worker exceeded 1800 seconds', status='timeout')
        old_handler = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(INTERNAL_SECONDS)
        with recorder.span('worker'):
            run_validation(args, recorder, result)
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = ('resource_limit' if original._memory_error(exc) else
                            'dependency_missing' if isinstance(exc, (ModuleNotFoundError, FileNotFoundError))
                            else getattr(exc, 'status', 'error'))
        result['error'] = dict(type=type(exc).__name__, message=str(exc)[:2000])
        if result['status'] == 'nonfinite':
            result['all_finite'] = False
        errors = [item for item in read_events(args.work_dir / 'events.jsonl')
                  if item.get('type') == 'end' and item.get('status') == 'error']
        result['failed_phase'] = errors[0]['stage_id'] if errors else 'worker'
    finally:
        if old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        if not result['evaluation_completed']:
            for key in ('task_passed', 'domain_passed', 'heldout_acceptance', 'validation_metrics',
                        'max_logit_absolute_difference', 'max_u'):
                result[key] = None
        result['resource_ru_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _write_terminal(args, result)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
