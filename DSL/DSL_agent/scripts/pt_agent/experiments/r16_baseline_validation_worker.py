"""R16 worker: one faithful baseline model and three saved training probes."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import signal
import sys

from .r10_paper_metrics import StageRecorder, read_events, write_json
from .r13_common_models import FACTORIES
from .r14_common_worker import WorkerFailure, _case_size, _memory_error


INTERNAL_SECONDS = 300
CASE_BYTES = 4 * 1024**3
PROBE_COUNT = 3
STAGE_FIELDS = ('source_checked', 'checkpoint_checked', 'model_built', 'strict_load_passed',
                'structure_passed', 'probes_checked', 'math_probes_passed')
COUNT_FIELDS = ('checkpoint_load_attempts', 'checkpoint_load_completed',
                'probe_load_attempts', 'probe_load_completed',
                'model_build_attempts', 'model_build_completed',
                'forward_attempts', 'forward_completed')
ZERO_FIELDS = ('llm_calls', 'trace_runs', 'compiler_runs', 'solver_runs', 'keygen_runs',
               'real_ckks_attempts', 'real_ckks_completed', 'init_scheme_calls',
               'backend_initializations', 'crypto_contexts', 'fit_calls', 'downloads',
               'installations', 'calibration_attempts', 'plan_attempts', 'count_attempts',
               'emit_attempts', 'confirm_samples', 'report_samples', 'heldout_samples',
               'freeze_attempts', 'agent_runs', 'tuner_runs', 'cost_estimates', 'model_imports')
UNMEASURED_FIELDS = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance')
MODEL_RECORDS = {
    'pt_agent.experiments.r13_common_models:resnet20_silu': dict(
        r13_run='r13-common-resnet20-silu-20261115', source_name='ResNetSiLU.py'),
    'pt_agent.experiments.r13_common_models:vgg16_silu_avgpool': dict(
        r13_run='r13-common-vgg16-silu-avgpool-20261115', source_name='VGG16SiLU.py'),
    'pt_agent.experiments.r13_common_models:alexnet_silu_avgpool': dict(
        r13_run='r13-common-alexnet-silu-avgpool-20261115', source_name='AlexNetSiLU.py'),
}
_NORMALIZATION = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_torch_initialized = False


def _initialize_environment():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'


def _same_path(left, right):
    return (isinstance(left, (str, Path)) and isinstance(right, (str, Path))
            and Path(left).resolve() == Path(right).resolve())


def validate_source(r13_result, model_info_record, graph_record) -> dict:
    """Validate saved JSON provenance without loading the Graph weight archive."""
    errors = []
    if not all(isinstance(item, dict) for item in (r13_result, model_info_record, graph_record)):
        return dict(passed=False, errors=['Saved source records must be JSON objects'], model=None)
    model = r13_result.get('model')
    expected = FACTORIES.get(model)
    worker = r13_result.get('worker') or {}
    if not isinstance(worker, dict):
        return dict(passed=False, errors=['Saved R13 worker must be a JSON object'], model=model)
    if (expected is None or model not in MODEL_RECORDS or r13_result.get('round') != 'R13'
            or r13_result.get('mode') != 'common-model-intake-preparation'
            or r13_result.get('status') != 'ok' or worker.get('round') != 'R13'
            or worker.get('model') != model or worker.get('status') != 'ok'):
        errors.append('Expected one of the three successful original R13 preparations')
    stages = ('import_completed', 'real_probes_passed', 'graph_math_passed',
              'roundtrip_passed', 'static_coverage')
    if any(worker.get(key) is not True for key in stages):
        errors.append('Original R13 import, references, readback or coverage did not pass')
    source = model_info_record.get('source') or {}
    if not isinstance(source, dict):
        errors.append('ModelInfo source must be a JSON object')
        source = {}
    if (source.get('factory') != model or source.get('path') != model
            or source.get('type') != 'torch' or source.get('spec_revision') != 'R13'):
        errors.append('Saved ModelInfo factory or original source kind differs')
    if expected is not None:
        for key, original in (('original_source', 'source'), ('checkpoint', 'checkpoint'),
                              ('inference_preprocessing_source', 'inference_preprocessing_source')):
            if not _same_path(source.get(key), expected[original]):
                errors.append('Original source location differs: ' + key)
    if (source != worker.get('model_source') or source.get('normalization') != _NORMALIZATION
            or source.get('training_preprocessing_verified') is not False
            or source.get('preprocessing_provenance') != 'official inference test'):
        errors.append('Saved original preprocessing provenance differs')
    split = dict(kind='legacy_cifar', heldout=list(range(9000, 10000)),
                 confirm=list(range(8000, 9000)), report=list(range(8000)),
                 calibration='frozen training selection')
    if model_info_record.get('split') != split or worker.get('split') != split:
        errors.append('The original fixed CIFAR split differs')
    if (model_info_record.get('task') != 'classification'
            or model_info_record.get('metric') != 'classification_top1'
            or model_info_record.get('threshold') != {'acc_drop_pp': 0.5}
            or (model_info_record.get('input_handling') or {}).get('internal_regression_source') != 'legacy_cifar'
            or model_info_record.get('graph') != 'graph'):
        errors.append('The original task, metric, policy or saved Graph location differs')
    inputs, outputs = graph_record.get('inputs'), graph_record.get('outputs')
    tensors = {}
    try:
        tensors = {name: list(shape) for name, shape, kind in graph_record.get('tensors', [])}
    except (TypeError, ValueError):
        errors.append('Saved Graph tensor metadata differs')
    if (not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(outputs, list)
            or len(outputs) != 1 or tensors.get(inputs[0]) != [1, 3, 32, 32]
            or tensors.get(outputs[0]) != [1, 10]):
        errors.append('Original Graph input or output shape differs')
    described = model_info_record.get('inputs')
    if (not isinstance(described, list) or len(described) != 1 or not isinstance(described[0], dict)
            or described[0].get('name') != (inputs[0] if isinstance(inputs, list) and len(inputs) == 1 else None)
            or described[0].get('kind') != 'features' or described[0].get('dtype') != 'float64'
            or described[0].get('shape') != [1, 3, 32, 32]):
        errors.append('Original ModelInfo features interface differs')
    if (worker.get('graph_math') or {}).get('passed') is not True:
        errors.append('The saved original Graph comparison did not pass')
    return dict(passed=not errors, errors=errors, model=model, graph_inputs=inputs,
                graph_outputs=outputs, input_metadata=described,
                normalization=source.get('normalization'), training_preprocessing_verified=False)


def check_math(reference, observed) -> dict:
    _initialize_environment()
    from .r15_rule_validation_worker import check_math as original_check
    return original_check(reference, observed)


def _source_paths(args, r13, info):
    model = args.model
    if model not in MODEL_RECORDS or r13.get('model') != model:
        raise WorkerFailure('The selected original model differs', status='source_failed')
    run = args.r13_result.resolve().parent
    if args.r13_result.name != 'result.json' or run.name != MODEL_RECORDS[model]['r13_run']:
        raise WorkerFailure('The original R13 directory differs', status='source_failed')
    artifacts = (r13.get('worker') or {}).get('artifacts') or {}
    for key, actual in (('model_info', args.model_info.resolve().parent),
                        ('training_probes', args.probes), ('training_probe_source', args.probe_source)):
        if not _same_path(artifacts.get(key), actual) or not actual.resolve().is_relative_to(run):
            raise WorkerFailure('Original R13 artifact differs: ' + key, status='source_failed')
    if (args.model_info.name != 'model_info.json'
            or args.graph_record.resolve() != args.model_info.resolve().parent / 'graph/graph.json'
            or not _same_path(args.checkpoint, (info.get('source') or {}).get('checkpoint'))):
        raise WorkerFailure('Original model metadata or checkpoint location differs', status='source_failed')
    source = (Path(args.baseline_root) / 'frontend/dacapo/python/poly/poly/models'
              / MODEL_RECORDS[model]['source_name'] if args.baseline == 'orbit'
              else Path((info.get('source') or {}).get('original_source', '')))
    if not _same_path(args.model_source, source):
        raise WorkerFailure('The selected baseline model source differs', status='source_failed')
    for path in (args.r13_result, args.model_info, args.graph_record, args.probes,
                 args.probe_source, args.checkpoint, args.model_source):
        if not path.is_file():
            raise FileNotFoundError('Missing source prerequisite: ' + str(path))
    return dict(r13_result=str(args.r13_result.resolve()), model_info=str(args.model_info.resolve()),
                graph_record=str(args.graph_record.resolve()), probes=str(args.probes.resolve()),
                probe_source=str(args.probe_source.resolve()), checkpoint=str(args.checkpoint.resolve()),
                model_source=str(args.model_source.resolve()), baseline_root=str(args.baseline_root.resolve()),
                r13_timing=r13.get('timing'), included_in_current_turnaround=False)


def _initialize_torch():
    global _torch_initialized
    if not _torch_initialized:
        _initialize_environment()
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _torch_initialized = True


def _build_model(args, state):
    if args.baseline == 'orbit':
        from . import r16_orbit_models as factory
        prior = factory._RECORDER
        factory._RECORDER = getattr(args, '_recorder', None)
        try:
            return factory.build_model(args.model, args.model_source, state)
        finally:
            factory._RECORDER = prior
    if args.baseline != 'orion':
        raise WorkerFailure('Unknown baseline', status='source_failed')
    root = args.baseline_root.resolve()
    sys.path.insert(0, str(root))
    try:
        import orion
        from orion.core import scheme
        from . import r16_orion_models as factory
        if not Path(orion.__file__).resolve().is_relative_to(root):
            raise WorkerFailure('Imported Orion is outside the selected baseline root', status='source_failed')
        if scheme.backend is not None or scheme.traced is not None:
            raise WorkerFailure('Orion must remain uninitialized', status='structure_failed')
        prior = factory._RECORDER
        factory._RECORDER = getattr(args, '_recorder', None)
        try:
            module, report = factory.build_model(args.model, state)
        finally:
            factory._RECORDER = prior
        report['orion_version'] = getattr(orion, '__version__', None)
        report['orion_source'] = str(Path(orion.__file__).resolve())
        return module, report
    finally:
        sys.path.remove(str(root))


def _observe_budget(args, result, stage, boundary, *, enforce=True):
    size = _case_size(args.work_dir.parent)
    result['case_artifact_bytes'] = size
    result['case_budget_observations'].append(dict(stage=stage, boundary=boundary, bytes=size,
                                                  limit_bytes=CASE_BYTES, within_budget=size <= CASE_BYTES))
    if enforce and size > CASE_BYTES:
        raise WorkerFailure('The R16 case exceeds 4 GiB', status='artifact_budget_exceeded')


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


def _read_probes(args, r13, info, graph, result):
    import numpy as np

    source = json.loads(args.probe_source.read_text())
    saved = (info.get('source') or {}).get('calibration_source')
    if (source != saved or source != r13['worker'].get('source') or source.get('split') != 'train'
            or source.get('normalization') != _NORMALIZATION):
        raise WorkerFailure('Original training probe provenance differs', status='source_failed')
    indices, labels = source.get('indices'), source.get('labels')
    if (not isinstance(indices, list) or len(indices) != 3
            or any(type(value) is not int or not 0 <= value < 50000 for value in indices)
            or not isinstance(labels, list) or len(labels) != 3
            or any(type(value) is not int or not 0 <= value < 10 for value in labels)
            or source.get('sample_ids') != ['train:' + str(value) for value in indices]):
        raise WorkerFailure('Original training probe indices or labels differ', status='source_failed')
    result['probe_load_attempts'] += 1
    write_json(args.work_dir / 'worker_result.json', result)
    with np.load(args.probes, allow_pickle=False) as archive:
        if set(archive.files) != {'input', 'labels', 'indices'}:
            raise WorkerFailure('Original probe archive fields differ', status='source_failed')
        images, stored_labels, stored_indices = (np.array(archive[name]) for name in ('input', 'labels', 'indices'))
    result['probe_load_completed'] += 1
    if (images.shape != (3, 3, 32, 32) or images.dtype != np.dtype('float64')
            or stored_labels.shape != (3,) or stored_indices.shape != (3,)
            or stored_labels.dtype.kind not in ('i', 'u') or stored_indices.dtype.kind not in ('i', 'u')
            or stored_labels.tolist() != labels or stored_indices.tolist() != indices):
        raise WorkerFailure('Saved training tensors, labels or indices differ', status='shape_failed')
    if not np.isfinite(images).all():
        raise WorkerFailure('Saved training probes contain nonfinite values', status='nonfinite')
    samples = (r13['worker'].get('graph_math') or {}).get('samples')
    if not isinstance(samples, list) or len(samples) != 3:
        raise WorkerFailure('Expected the three saved original reference outputs', status='source_failed')
    references = []
    for index, sample in enumerate(samples):
        if (not isinstance(sample, dict) or sample.get('probe_index') != index
                or sample.get('passed') is not True or sample.get('finite') is not True
                or sample.get('shape_ok') is not True or sample.get('input_name') != graph['inputs'][0]
                or sample.get('output_name') != graph['outputs'][0]
                or sample.get('reference_shape') != [1, 10]):
            raise WorkerFailure('Saved reference identity, shape or finite status differs', status='source_failed')
        reference = np.asarray(sample.get('reference'), dtype=np.float64)
        if reference.shape != (1, 10) or not np.isfinite(reference).all():
            raise WorkerFailure('Saved reference logits are not finite float64 [1,10]', status='source_failed')
        references.append(reference)
    return images, stored_labels, references, source


def _checkpoint(args, result):
    import torch

    result['checkpoint_load_attempts'] += 1
    write_json(args.work_dir / 'worker_result.json', result)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    result['checkpoint_load_completed'] += 1
    state = checkpoint.get('state_dict') if args.model.endswith(':resnet20_silu') and isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict) or not state or any(not isinstance(name, str) for name in state):
        raise WorkerFailure('Original checkpoint state is not a nonempty named tensor dictionary', status='state_failed')
    rows = []
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu':
            raise WorkerFailure('Original checkpoint contains a non-CPU tensor: ' + name, status='state_failed')
        finite = bool(torch.isfinite(tensor).all())
        rows.append(dict(name=name, shape=list(tensor.shape), original_dtype=str(tensor.dtype), finite=finite))
        if not finite:
            result['checkpoint_state'] = rows
            raise WorkerFailure('Original checkpoint contains nonfinite values: ' + name, status='nonfinite')
    result['checkpoint_state'] = rows
    result['checkpoint_envelope'] = dict(kind='state_dict' if args.model.endswith(':resnet20_silu') else 'bare_state_dict',
                                          keys=list(checkpoint) if args.model.endswith(':resnet20_silu') else None,
                                          original_dictionary_modified=False)
    return state


def run_validation(args, recorder, result) -> None:
    _initialize_environment()
    import numpy as np
    import torch

    args._recorder = recorder

    def save(name, record):
        path = args.work_dir / (name + '.json')
        write_json(path, record)
        result['artifacts'][name] = str(path)

    with _phase(args, recorder, result, 'initialize_threads'):
        _initialize_torch()
        result['environment'] = dict(python=sys.version, executable=sys.executable,
                                      torch=str(torch.__version__), numpy=np.__version__,
                                      baseline_root=str(args.baseline_root.resolve()), cpu=True,
                                      intraop_threads=torch.get_num_threads(),
                                      interop_threads=torch.get_num_interop_threads(),
                                      thread_environment={name: os.environ[name] for name in
                                          ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                           'NUMEXPR_NUM_THREADS')})
    with _phase(args, recorder, result, 'source_check'):
        r13 = json.loads(args.r13_result.read_text())
        info = json.loads(args.model_info.read_text())
        graph = json.loads(args.graph_record.read_text())
        source_paths = _source_paths(args, r13, info)
        checked = validate_source(r13, info, graph)
        result['source_check'] = checked
        result['source_checked'] = checked['passed']
        save('source_check', checked)
        if not checked['passed']:
            raise WorkerFailure('Saved original source checks failed: ' + '; '.join(checked['errors']), status='source_failed')
        result['source_summary'] = source_paths
        for field in ('source', 'inputs', 'split', 'assumptions', 'task', 'metric', 'threshold'):
            result[{'source': 'model_source', 'inputs': 'input_metadata'}.get(field, field)] = info.get(field)
    with _phase(args, recorder, result, 'checkpoint_load'):
        state = _checkpoint(args, result)
        result['checkpoint_checked'] = True
        save('checkpoint_state', result['checkpoint_state'])
    with _phase(args, recorder, result, 'model_build_and_strict_load'):
        result['model_build_attempts'] += 1
        write_json(args.work_dir / 'worker_result.json', result)
        module, report = _build_model(args, state)
        result['model_build_completed'] += 1
        result['model_built'] = True
        result['state_report'] = report
        result['strict_load_passed'] = report.get('strict_load') is True
        result['structure_passed'] = (report.get('structure') or {}).get('passed') is True
        save('state_report', report)
        if (report.get('passed') is not True or not result['strict_load_passed']
                or not result['structure_passed'] or len(report.get('original_state', [])) != len(state)
                or any(row.get('finite') is not True or row.get('numerically_equal') is not True
                       for row in report.get('original_state', []))):
            raise WorkerFailure('Strict original state or model structure checks failed', status='state_failed')
        if args.baseline == 'orion':
            if (report.get('lightweight_scheme_present') is not True or report.get('backend_is_none') is not True
                    or report.get('all_he_mode_false') is not True
                    or any(report.get(key) != 0 for key in ('init_scheme_calls', 'backend_initializations',
                                                          'keygen_runs', 'fit_calls', 'compile_runs'))):
                raise WorkerFailure('Orion must remain in uninitialized plaintext mode', status='structure_failed')
            result['environment'].update(lightweight_scheme_present=True, backend_is_none=True,
                                          orion_version=report.get('orion_version'),
                                          orion_source=report.get('orion_source'))
        del state
    with _phase(args, recorder, result, 'read_training_probes'):
        images, labels, references, source = _read_probes(args, r13, info, graph, result)
        result.update(probes_checked=True, probe_source=source)
        save('probe_source', source)
    for index in range(3):
        sample_id = source['sample_ids'][index]
        with _phase(args, recorder, result, 'probe', probe_index=index, sample_id=sample_id):
            with recorder.span('model_forward', probe_index=index, sample_id=sample_id):
                result['forward_attempts'] += 1
                write_json(args.work_dir / 'worker_result.json', result)
                with torch.no_grad():
                    output = module(torch.from_numpy(images[index:index + 1]))
                result['forward_completed'] += 1
                write_json(args.work_dir / 'worker_result.json', result)
                if not isinstance(output, torch.Tensor) or output.dtype != torch.float64 or tuple(output.shape) != (1, 10):
                    raise WorkerFailure('Baseline output must be original float64 [1,10]', status='shape_failed')
                if output.device.type != 'cpu' or not bool(torch.isfinite(output).all()):
                    raise WorkerFailure('Baseline output must be finite and on CPU', status='nonfinite')
                observed = output.detach().cpu().numpy()
            with recorder.span('math_check', probe_index=index, sample_id=sample_id):
                checked = check_math(references[index], observed)
                record = dict(probe_index=index, sample_id=sample_id, input_name=graph['inputs'][0],
                              output_name=graph['outputs'][0], reference=references[index].tolist(),
                              observed=observed.tolist(), **checked)
                result['math_probes'].append(record)
                save('math_probes', dict(samples=result['math_probes']))
                if not checked['passed']:
                    result['math_probes_passed'] = False
                    raise WorkerFailure('Strict mathematical probe failed at ' + sample_id, status='math_failed')
            del output, observed
    result.update(math_probes_passed=True, all_finite=True, compatibility_passed=True)


def _initial_result():
    return dict(round='R16', model=None, baseline=None, status='error', error=None, failed_phase=None,
                pid=os.getpid(), ppid=os.getppid(), artifacts={}, case_budget_observations=[],
                case_artifact_bytes=None, compatibility_passed=None, all_finite=None,
                candidate_frozen=False, source_check=None, state_report=None, probe_source=None,
                source_summary=None, environment=None, math_probes=[],
                evidence_scope='Original structure/state and at most three saved training probes; CKKS 0',
                **dict.fromkeys(STAGE_FIELDS), **dict.fromkeys(COUNT_FIELDS, 0),
                **dict.fromkeys(ZERO_FIELDS, 0), **dict.fromkeys(UNMEASURED_FIELDS))


def _write_terminal(args, result):
    path = args.work_dir / 'worker_result.json'
    base = _case_size(args.work_dir.parent) - (path.stat().st_size if path.exists() else 0)
    terminal = dict(stage='worker', boundary='terminal', bytes=None,
                    limit_bytes=CASE_BYTES, within_budget=None)
    result['case_budget_observations'].append(terminal)
    for _ in range(8):
        size = base + len((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
        if size > CASE_BYTES:
            result.update(status='artifact_budget_exceeded',
                          error=dict(type='WorkerFailure', message='The R16 case exceeds 4 GiB'))
        same = terminal['bytes'] == size and result['case_artifact_bytes'] == size
        terminal.update(bytes=size, within_budget=size <= CASE_BYTES)
        result['case_artifact_bytes'] = size
        if same:
            write_json(path, result)
            return
    raise RuntimeError('R16 terminal artifact accounting did not settle')


def main(argv=None) -> int:
    _initialize_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', choices=('orion', 'orbit'), required=True)
    parser.add_argument('--model', choices=tuple(MODEL_RECORDS), required=True)
    for name in ('r13-result', 'model-info', 'graph-record', 'probes', 'probe-source',
                 'checkpoint', 'model-source', 'baseline-root', 'work-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if (args.work_dir / 'worker_result.json').exists():
        raise FileExistsError('Existing R16 worker results are preserved')
    result = _initial_result()
    result.update(model=args.model, baseline=args.baseline)
    recorder = StageRecorder(args.work_dir / 'events.jsonl')
    old_handler = None
    try:
        def timeout(signum, frame):
            raise WorkerFailure('The R16 worker exceeded 300 seconds', status='timeout')
        old_handler = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(INTERNAL_SECONDS)
        with recorder.span('worker', baseline=args.baseline, model=args.model):
            run_validation(args, recorder, result)
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = ('resource_limit' if _memory_error(exc) else
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
        result['resource_ru_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _write_terminal(args, result)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
