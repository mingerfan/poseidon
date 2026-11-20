"""R15: fixed-rule heldout validation with three packed arithmetic probes."""

from __future__ import annotations

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

from . import r14_common_worker as previous
from .r10_paper_metrics import StageRecorder, read_events, write_json


INTERNAL_SECONDS = 1800
CASE_BYTES = 4 * 1024**3
SAMPLE_COUNT = 1000
PROBE_COUNT = 3
GRAPH_CRITERION = '1e-12+1e-12*abs(reference)'
DOMAIN_LIMIT = 0.90
MODEL_RECORDS = {
    'pt_agent.experiments.r13_common_models:resnet20_silu': dict(
        export_run='r14-common-resnet20-silu-20261115', round='R14',
        mode='common-model-rule-preparation', act_count=19),
    'pt_agent.experiments.r13_common_models:vgg16_silu_avgpool': dict(
        export_run='r14a-vgg16-emit-resume-20261115', round='R14a',
        mode='vgg-emit-resume', act_count=15),
    'pt_agent.experiments.r13_common_models:alexnet_silu_avgpool': dict(
        export_run='r14b-alexnet-emit-resume-20261115', round='R14b',
        mode='alex-emit-resume', act_count=7),
}
ENGINE_NAMES = ('reference_forward', 'tensor_exact', 'tensor_program', 'mock_exact', 'mock_program')
STAGE_FIELDS = ('source_checked', 'typecheck_passed', 'signature_passed', 'samples_checked', 'math_probes_passed')
ZERO_FIELDS = ('llm_calls', 'real_ckks_attempts', 'real_ckks_completed', 'trace_runs', 'compiler_runs',
               'keygen_runs', 'model_imports', 'downloads', 'installations', 'calibration_attempts',
               'plan_attempts', 'count_attempts', 'emit_attempts', 'confirm_samples', 'report_samples',
               'freeze_attempts', 'agent_runs', 'tuner_runs', 'cost_estimates')
UNMEASURED_FIELDS = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance')
SOURCE_STAGES = ('typecheck_passed', 'signature_passed', 'stats_equal', 'emit_completed', 'lowering_ready')
PRIOR_STAGES = ('reference_passed', 'calibration_completed', 'plan_completed',
                'typecheck_passed', 'signature_passed', 'count_completed', 'stats_equal')
STAT_FIELDS = ('counts', 'ops_total', 'distinct_consts')
WorkerFailure = previous.WorkerFailure
_torch_initialized = False


def validate_export_source(export_result, rule_result, r13_result, info) -> dict:
    """Check saved source observations and preserve the original ModelInfo."""
    errors = []
    if not all(isinstance(record, dict) for record in (export_result, rule_result, r13_result)):
        return dict(passed=False, model=None, errors=['Source records must be JSON objects'])
    model = export_result.get('model')
    expected = MODEL_RECORDS.get(model)
    exported = export_result.get('worker') or {}
    rule = rule_result.get('worker') or {}
    if not isinstance(exported, dict) or not isinstance(rule, dict):
        return dict(passed=False, model=model, errors=['Source workers must be JSON objects'])
    if (expected is None or export_result.get('status') != 'ok' or exported.get('status') != 'ok'
            or exported.get('model') != model or any(exported.get(key) is not True for key in SOURCE_STAGES)):
        errors.append('Expected one of the three successful rule exports')
    if expected is not None:
        if (export_result.get('round') != expected['round'] or exported.get('round') != expected['round']
                or export_result.get('mode') != expected['mode']):
            errors.append('Export round or mode differs from the selected model')
        required_status = 'ok' if expected['round'] == 'R14' else 'artifact_budget_exceeded'
        if (rule_result.get('round') != 'R14' or rule.get('round') != 'R14'
                or rule_result.get('mode') != 'common-model-rule-preparation'
                or rule_result.get('model') != model or rule.get('model') != model
                or rule_result.get('status') != required_status or rule.get('status') != required_status
                or any(rule.get(key) is not True for key in PRIOR_STAGES)):
            errors.append('The original R14 reference, calibration, rule and counts must have passed')
        if expected['round'] != 'R14' and (rule.get('failed_phase') != 'emit_write'
                or rule.get('emit_completed') is not None or rule.get('lowering_ready') is not None):
            errors.append('Expected the retained original R14 prewrite budget stop')
    if rule.get('rule_options') != previous.RULE_OPTIONS or exported.get('rule_options') != previous.RULE_OPTIONS:
        errors.append('The original fixed rule options differ')
    if r13_result.get('model') != model:
        errors.append('The R13 model differs from the exported model')
    checked = previous.check_saved_source(info, r13_result)
    if not checked['passed']:
        errors.extend(checked['errors'])
    if (info.task != 'classification' or info.metric != 'classification_top1'
            or info.threshold != {'acc_drop_pp': 0.5}):
        errors.append('The saved original classification policy differs')
    for name in STAT_FIELDS:
        if ((rule.get('count_stats') or {}).get(name) != (exported.get('count_stats') or {}).get(name)
                or (exported.get('count_stats') or {}).get(name) != (exported.get('emit_stats') or {}).get(name)):
            errors.append('Original Count and completed Emit statistics differ: ' + name)
    return dict(passed=not errors, model=model, errors=errors, saved_source=checked,
                prior_stages={key: rule.get(key) for key in PRIOR_STAGES},
                policy=dict(task=info.task, metric=info.metric, threshold=info.threshold))


def check_math(reference, observed) -> dict:
    """Check the original dtype, shape and strict elementwise mathematical rule."""
    import numpy as np

    reference_shape = list(reference.shape) if hasattr(reference, 'shape') else None
    observed_shape = list(observed.shape) if hasattr(observed, 'shape') else None
    shape_ok = reference_shape == observed_shape == [1, 10]
    dtype_ok = (isinstance(reference, np.ndarray) and isinstance(observed, np.ndarray)
                and reference.dtype == np.dtype('float64') and observed.dtype == np.dtype('float64'))
    finite = bool(dtype_ok and np.isfinite(reference).all() and np.isfinite(observed).all())
    result = dict(passed=False, shape_ok=shape_ok, dtype_ok=dtype_ok, finite=finite,
                  reference_shape=reference_shape, observed_shape=observed_shape,
                  reference_dtype=str(getattr(reference, 'dtype', None)),
                  observed_dtype=str(getattr(observed, 'dtype', None)),
                  max_absolute_error=None, max_error_ratio=None, criterion=GRAPH_CRITERION)
    if shape_ok and dtype_ok and finite:
        with np.errstate(over='ignore', invalid='ignore'):
            difference = np.abs(observed-reference)
            allowance = 1e-12 + 1e-12*np.abs(reference)
            ratio = difference/allowance
        finite_errors = bool(np.isfinite(difference).all() and np.isfinite(ratio).all())
        result.update(passed=bool(finite_errors and np.all(difference <= allowance)),
                      max_absolute_error=float(difference.max()) if finite_errors else None,
                      max_error_ratio=float(ratio.max()) if finite_errors else None)
    return result


def _source_paths(args, exported, rule, r13):
    """Require explicit argv to identify exactly the saved source artifacts."""
    export_dir, rule_dir, r13_dir = (path.resolve().parent for path in
                                   (args.export_result, args.rule_result, args.r13_result))
    ew, rw = exported.get('worker') or {}, rule.get('worker') or {}
    artifact = rw.get('artifacts') or {}

    def require(value, actual, directory, name):
        if (not isinstance(value, str) or Path(value).resolve() != actual.resolve()
                or not actual.resolve().is_relative_to(directory)):
            raise WorkerFailure('Source artifact location differs: ' + name, status='source_failed')

    if exported.get('round') == 'R14':
        if args.rule_result.resolve() != args.export_result.resolve():
            raise WorkerFailure('The ResNet rule and export result must be the same record', status='source_failed')
    else:
        resume = ew.get('source_resume') or {}
        outer_source = exported.get('source_r14') or {}
        outer_artifacts = outer_source.get('artifacts') or {}
        require(outer_source.get('result'), args.rule_result, rule_dir, 'source_r14.result')
        if outer_source.get('run') != str(rule_dir):
            raise WorkerFailure('The original rule run differs', status='source_failed')
        for name, actual, directory in (
                ('source_result', args.rule_result, rule_dir), ('program', args.program, rule_dir),
                ('count', args.count, rule_dir), ('r13_result', args.r13_result, r13_dir),
                ('model_info', args.model_info, r13_dir)):
            require(resume.get(name), actual, directory, 'source_resume.' + name)
            require(outer_artifacts.get(name), actual, directory, 'source_r14.artifacts.' + name)
    require(artifact.get('program'), args.program, rule_dir, 'program')
    require(artifact.get('count'), args.count, rule_dir, 'count')
    source_r13 = rule.get('source_r13') or {}
    require(source_r13.get('result'), args.r13_result, r13_dir, 'source_r13.result')
    require(rw.get('r13_source_result'), args.r13_result, r13_dir, 'worker.r13_source_result')
    require((source_r13.get('artifacts') or {}).get('model_info'), args.model_info, r13_dir, 'source_r13.model_info')
    require(((r13.get('worker') or {}).get('artifacts') or {}).get('model_info'),
            args.model_info, r13_dir, 'r13.artifacts.model_info')
    if source_r13.get('run') != str(r13_dir):
        raise WorkerFailure('The original R13 directory differs', status='source_failed')
    for path in (args.export_result, args.rule_result, args.program, args.count, args.r13_result):
        if not path.is_file():
            raise FileNotFoundError('Missing source prerequisite: ' + str(path))
    metadata = {}
    for key in ('trace_source', 'constants'):
        value = (ew.get('artifacts') or {}).get(key)
        if not isinstance(value, str) or not Path(value).resolve().is_relative_to(export_dir):
            raise WorkerFailure('Exported artifact location differs: ' + key, status='source_failed')
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError('Missing exported artifact: ' + str(path))
        metadata[key] = dict(path=str(path.resolve()), bytes=path.stat().st_size)
    return dict(export_result=str(args.export_result.resolve()), rule_result=str(args.rule_result.resolve()),
                program=str(args.program.resolve()), count=str(args.count.resolve()),
                r13_result=str(args.r13_result.resolve()), model_info=str(args.model_info.resolve()),
                exported_artifacts=metadata, included_in_current_turnaround=False)


def _check_program(info, program):
    header = program.header
    errors = []
    if (header.slots != 16384 or header.refresh != 'compiler' or header.placement != 'depth-dp'
            or header.manage != 'compiler' or header.l_max is not None or header.boot_out_level is not None):
        errors.append('Original rule header differs')
    acts = [op for op in program.ops if op.kind == 'act']
    expected = MODEL_RECORDS.get(info.source.get('factory'))
    if expected is None or len(acts) != expected['act_count']:
        errors.append('Original activation count differs')
    for op in acts:
        a = op.attrs
        domain = a.get('domain')
        if (a.get('fn') != 'silu' or a.get('method') != 'cheb' or a.get('deg') != 63
                or not isinstance(domain, tuple) or len(domain) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in domain)
                or domain[0] >= domain[1]):
            errors.append('Original activation strategy or domain differs at op ' + str(op.id))
    return dict(passed=not errors, errors=errors, act_count=len(acts), header=asdict(header))


def _read_heldout(args, info):
    import numpy as np
    from ..data import cifar
    from ..data.samples import LegacyCifar

    if args.records.resolve() != Path(cifar.RECORDS).resolve():
        raise WorkerFailure('Heldout records differ from the original CIFAR source', status='source_failed')
    metadata = json.loads((args.records / 'dataset.json').read_text())
    expected = dict(format='poseidon-cifar10-binary-records-v1', record_bytes=3073,
                    row_label_byte=0, preprocessing_applied=False, train_count=50000,
                    test_count=10000, pixel_layout='uint8 planar RGB 3x32x32, C order')
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise WorkerFailure('CIFAR record metadata differs', status='source_failed')
    normalization = info.source['normalization']
    arrays, labels = LegacyCifar(info.graph, normalization=(normalization['mean'], normalization['std'])).get('heldout', 1000)
    name = info.graph.inputs[0]
    if (set(arrays) != {name} or not isinstance(arrays[name], np.ndarray)
            or arrays[name].shape != (1000, 3, 32, 32) or arrays[name].dtype != np.dtype('float64')
            or not isinstance(labels, np.ndarray) or labels.shape != (1000,)
            or labels.dtype.kind not in ('i', 'u') or np.any(labels < 0) or np.any(labels > 9)):
        raise WorkerFailure('Heldout input shape, dtype or labels differ', status='shape_failed')
    if not np.isfinite(arrays[name]).all():
        raise WorkerFailure('Heldout inputs contain nonfinite values', status='nonfinite')
    return arrays, labels, dict(records=str(args.records.resolve()), split='heldout', source_split='test',
                               indices=list(range(9000, 10000)), sample_ids=['test:' + str(i) for i in range(9000, 10000)],
                               labels=labels.tolist(), count=1000, batch=1,
                               normalization=normalization, shard='test-09000.records',
                               confirm_samples=0, report_samples=0)


def _domain_rows(program):
    return [dict(op_id=op.id, program_line=op.lineno, input_name=op.inputs[0], fn=op.attrs['fn'],
                 method=op.attrs['method'], deg=op.attrs['deg'], domain=list(op.attrs['domain']),
                 count=0, obs_min=None, obs_max=None, max_u=0.0,
                 min_sample_id=None, max_sample_id=None, u_sample_id=None)
            for op in program.ops if op.kind == 'act']


def _tensor_observer(program, sample_id, rows):
    import numpy as np
    from ..domain_criteria import maximum_u

    acts = [op for op in program.ops if op.kind == 'act']
    observed = 0

    def observer(name, data, cfg):
        nonlocal observed
        if observed >= len(acts) or len(rows) != len(acts):
            raise WorkerFailure('Too many or mismatched activation observations', status='observer_failed')
        op, row = acts[observed], rows[observed]
        wanted = ('cheb', 63, *op.attrs['domain'])
        if name != op.inputs[0] or cfg != wanted or row['op_id'] != op.id:
            raise WorkerFailure('Activation observer order, name or configuration differs at ' + str(op.id),
                                status='observer_failed')
        if not isinstance(data, np.ndarray) or data.size == 0:
            raise WorkerFailure('Activation observer requires a nonempty ndarray', status='observer_failed')
        if not np.isfinite(data).all():
            raise WorkerFailure('Nonfinite activation input at ' + str(op.id), status='nonfinite')
        minimum, maximum = float(data.min()), float(data.max())
        u = maximum_u(data, op.attrs['domain'], fn='silu', domain_source='calibrated')
        if not math.isfinite(u):
            raise WorkerFailure('Nonfinite activation coverage at ' + str(op.id), status='nonfinite')
        if row['obs_min'] is None or minimum < row['obs_min']:
            row.update(obs_min=minimum, min_sample_id=sample_id)
        if row['obs_max'] is None or maximum > row['obs_max']:
            row.update(obs_max=maximum, max_sample_id=sample_id)
        if row['u_sample_id'] is None or u > row['max_u']:
            row.update(max_u=u, u_sample_id=sample_id)
        row['count'] += 1
        observed += 1

    def finish():
        if observed != len(acts):
            raise WorkerFailure('Missing activation observer calls for ' + str(sample_id), status='observer_failed')
    return observer, finish


def _packed_logits(program, weights, image, *, act_mode):
    from ..backends import MockBackend
    from ..lower import decode_outputs, lower, pack_inputs

    backend = MockBackend(slots=16384, batch=1, stream_constants=True)
    values = layouts = packed = None
    try:
        packed = pack_inputs(program, [image])
        backend.bind(packed)
        values, layouts = lower(program, weights, backend, act_mode=act_mode)
        outputs = decode_outputs(backend.outputs, layouts)
        if len(outputs) != 1:
            raise WorkerFailure('Packed execution must have one output', status='shape_failed')
        result = outputs[0].copy()
        del outputs
        return result
    finally:
        del values, layouts, packed
        backend.release_batch()
        del backend
        gc.collect()


def _initialize_torch():
    global _torch_initialized
    if not _torch_initialized:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _torch_initialized = True


def _observe_budget(args, result, stage, boundary, *, enforce=True):
    size = previous._case_size(args.work_dir.parent)
    result['case_artifact_bytes'] = size
    result['case_budget_observations'].append(dict(stage=stage, boundary=boundary, bytes=size,
                                                  limit_bytes=CASE_BYTES, within_budget=size <= CASE_BYTES))
    if enforce and size > CASE_BYTES:
        raise WorkerFailure('The R15 case exceeded 4 GiB of artifacts', status='artifact_budget_exceeded')


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


def _require_logits(value):
    import numpy as np
    if not isinstance(value, np.ndarray) or value.shape != (1, 10) or value.dtype != np.dtype('float64'):
        raise WorkerFailure('Engine logits must be original float64 [1,10]', status='shape_failed')
    if not np.isfinite(value).all():
        raise WorkerFailure('Engine logits contain nonfinite values', status='nonfinite')
    return value


def _call_engine(args, recorder, result, engine, sample_id, function):
    counter = result['engine_counts'][engine]
    counter['attempted'] += 1
    write_json(args.work_dir / 'worker_result.json', result)
    with recorder.span(engine, sample_id=sample_id):
        value = function()
        counter['completed'] += 1
        write_json(args.work_dir / 'worker_result.json', result)
        return _require_logits(value)


def _append_record(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def run_validation(args, recorder, result) -> None:
    import numpy as np
    from ..analyzer import reference_forward
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from ..metrics import compare, passes
    from ..models.info import ModelInfo
    from ..tensor_sim import run_tensor

    def save(name, value):
        path = args.work_dir / (name + '.json')
        write_json(path, value)
        result['artifacts'][name] = str(path)

    with _phase(args, recorder, result, 'initialize_threads'):
        _initialize_torch()
    with _phase(args, recorder, result, 'load_saved_source'):
        exported = json.loads(args.export_result.read_text())
        rule = json.loads(args.rule_result.read_text())
        r13 = json.loads(args.r13_result.read_text())
        result['model'] = exported.get('model')
        source_paths = _source_paths(args, exported, rule, r13)
        info = ModelInfo.load(args.model_info)
        source = validate_export_source(exported, rule, r13, info)
        save('source_check', source)
        result['source_checked'] = source['passed']
        if not source['passed']:
            raise WorkerFailure('Saved export source checks failed: ' + '; '.join(source['errors']), status='source_failed')
        count_record = json.loads(args.count.read_text())
        count = count_record.get('stats')
        if not isinstance(count, dict) or any(key not in count for key in STAT_FIELDS):
            raise WorkerFailure('Saved Count statistics are missing', status='source_failed')
        if any(count[key] != (rule['worker'].get('count_stats') or {}).get(key)
               or count[key] != (exported['worker'].get('count_stats') or {}).get(key) for key in STAT_FIELDS):
            raise WorkerFailure('Saved Count artifact differs from original observations', status='source_failed')
        result['source_count_stats'] = count
        result['source_preparation'] = dict(**source_paths, export_timing=exported.get('timing'),
                                            rule_timing=rule.get('timing'), r13_timing=r13.get('timing'))
        for key in ('source', 'inputs', 'split', 'assumptions', 'task', 'metric', 'threshold'):
            result[{'source': 'model_source', 'inputs': 'input_metadata'}.get(key, key)] = getattr(info, key)
    with _phase(args, recorder, result, 'parse_typecheck'):
        program = parse(args.program.read_text())
        checked = check(program, {name: weight.shape for name, weight in info.graph.weights.items()}, graph=info.graph)
        save('typecheck', dict(ok=checked.ok, diags=[asdict(diag) for diag in checked.diags]))
        result['typecheck_passed'] = checked.ok
        if not checked.ok:
            raise WorkerFailure('Saved rule typechecking failed', status='typecheck_failed')
        strategy = _check_program(info, program)
        save('rule_strategy', strategy)
        if not strategy['passed']:
            raise WorkerFailure('Saved rule strategy checks failed: ' + '; '.join(strategy['errors']), status='source_failed')
    with _phase(args, recorder, result, 'signature_check'):
        signature = previous.check_program_signature(info, program)
        save('signature', signature)
        result['signature_passed'] = signature['passed']
        if not signature['passed']:
            raise WorkerFailure('Saved rule input signature differs', status='signature_failed')
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
            observation = check_math(reference, observed)
            record = dict(sample_id=sample_id, pair=pair_name, reference=reference.tolist(),
                          observed=observed.tolist(), **observation)
            result['math_probes'].append(record)
            _append_record(probe_path, record)
            if not observation['passed']:
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
            finish()
            exact[index], approximate[index] = reference[0], actual[0]
            result['completed_samples'] += 1
            _append_record(samples_path, dict(sample_id=sample_id, index=9000 + index, label=int(labels[index]),
                                              reference=reference.tolist(), approximate=actual.tolist(),
                                              partial=True))
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
            difference = np.abs(approximate-exact)
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
                                    domain_passed=result['domain_passed'], max_u=maximum,
                                    heldout_acceptance=result['heldout_acceptance'],
                                    completed_samples=1000, acc_drop_pp_unit='percentage points',
                                    max_logit_absolute_difference=result['max_logit_absolute_difference']))
        save('domain_rows', dict(rows=rows, limit=DOMAIN_LIMIT, domain_passed=result['domain_passed'], partial=False))
        outputs = args.work_dir / 'heldout_outputs.npz'
        np.savez(outputs, reference=exact, approximate=approximate, labels=labels,
                 indices=np.arange(9000, 10000, dtype=np.int64))
        result['artifacts']['heldout_outputs'] = str(outputs)


def _initial_result():
    return dict(round='R15', model=None, status='error', error=None, failed_phase=None,
                pid=os.getpid(), ppid=os.getppid(), artifacts={}, case_budget_observations=[],
                case_artifact_bytes=None, evaluation_completed=False, completed_samples=0,
                task_passed=None, domain_passed=None, heldout_acceptance=None, all_finite=None,
                validation_metrics=None, max_logit_absolute_difference=None, max_u=None,
                domain_rows=None, source_count_stats=None, math_probes=[], outputs_partial=True,
                engine_counts={name: dict(attempted=0, completed=0) for name in ENGINE_NAMES},
                candidate_frozen=False,
                evidence_scope='Planned: 1000 Tensor samples and 3 packed Mock probes; actual counts recorded separately; CKKS 0',
                **dict.fromkeys(STAGE_FIELDS), **dict.fromkeys(UNMEASURED_FIELDS),
                **dict.fromkeys(ZERO_FIELDS, 0))


def _write_terminal(args, result):
    path = args.work_dir / 'worker_result.json'
    others = previous._case_size(args.work_dir.parent) - (path.stat().st_size if path.exists() else 0)
    observation = dict(stage='worker', boundary='terminal', bytes=None, limit_bytes=CASE_BYTES, within_budget=None)
    result['case_budget_observations'].append(observation)
    while True:
        size = others + len((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
        if size > CASE_BYTES:
            result.update(status='artifact_budget_exceeded',
                          error=dict(type='WorkerFailure', message='The R15 case exceeded 4 GiB'))
        same = result['case_artifact_bytes'] == size and observation['bytes'] == size
        result['case_artifact_bytes'] = observation['bytes'] = size
        observation['within_budget'] = size <= CASE_BYTES
        if same:
            break
    write_json(path, result)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('export-result', 'rule-result', 'program', 'count', 'r13-result',
                 'model-info', 'records', 'work-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.work_dir / 'worker_result.json'
    if result_path.exists():
        raise FileExistsError('Existing R15 worker results are preserved')
    result = _initial_result()
    recorder = StageRecorder(args.work_dir / 'events.jsonl')
    old_handler = None
    try:
        def timeout(signum, frame):
            raise WorkerFailure('The R15 worker exceeded 1800 seconds', status='timeout')
        old_handler = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(INTERNAL_SECONDS)
        with recorder.span('worker'):
            run_validation(args, recorder, result)
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = ('resource_limit' if previous._memory_error(exc) else
                            'dependency_missing' if isinstance(exc, (ModuleNotFoundError, FileNotFoundError)) else
                            getattr(exc, 'status', 'error'))
        result['error'] = dict(type=type(exc).__name__, message=str(exc)[:2000])
        if result['status'] == 'nonfinite':
            result['all_finite'] = False
        errors = [event for event in read_events(args.work_dir / 'events.jsonl')
                  if event.get('type') == 'end' and event.get('status') == 'error']
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
