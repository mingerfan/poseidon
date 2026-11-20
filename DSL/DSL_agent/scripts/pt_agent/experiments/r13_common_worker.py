"""R13 path-driven intake of one original DaCapo model and three training inputs."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import resource
import signal

from .r10_paper_metrics import StageRecorder, read_events, write_json


REAL_PROBE_CRITERION = '1e-9*(1+abs(reference))'
GRAPH_CRITERION = '1e-12+1e-12*abs(reference)'
INTERNAL_SECONDS = 1800


class WorkerFailure(RuntimeError):
    def __init__(self, message, *, status='error'):
        super().__init__(message)
        self.status = status


def read_calibration(records, selections):
    """Read only the first three frozen training selections, with their provenance."""
    import numpy as np
    from ..data import cifar

    records, selections = Path(records).resolve(), Path(selections).resolve()
    if records != cifar.RECORDS.resolve() or selections != cifar.SELECTIONS.resolve():
        raise ValueError('R13 requires the existing CIFAR records and frozen selections')
    metadata = json.loads((records / 'dataset.json').read_text())
    expected = dict(format='poseidon-cifar10-binary-records-v1', record_bytes=3073,
                    row_label_byte=0, preprocessing_applied=False,
                    train_count=50000, test_count=10000,
                    pixel_layout='uint8 planar RGB 3x32x32, C order')
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('CIFAR record metadata does not describe the original raw dataset')
    selected = json.loads(selections.read_text()).get('calibration')
    if not isinstance(selected, list) or len(selected) < 3:
        raise ValueError('R13 requires three existing calibration selections')
    entries = []
    for entry in selected[:3]:
        if not isinstance(entry, dict):
            raise ValueError('Invalid calibration selection record')
        index, label = entry.get('index'), entry.get('label')
        if type(index) is not int or not 0 <= index < 50000:
            raise ValueError('Calibration indices must identify original training rows')
        if type(label) is not int or not 0 <= label < 10:
            raise ValueError('Calibration labels must identify CIFAR classes')
        shard = f'train-{index // 1000 * 1000:05d}.records'
        if entry.get('shard') != shard or entry.get('row') != index % 1000:
            raise ValueError('Calibration shard/row disagrees with the original index')
        matches = [item for item in metadata.get('shards', [])
                   if item.get('file') == shard and item.get('split') == 'train'
                   and item.get('start') == index // 1000 * 1000
                   and item.get('count') == 1000]
        if len(matches) != 1 or not (records / shard).is_file():
            raise ValueError('Selected training shard metadata or file is missing')
        entries.append(dict(index=index, label=label, shard=shard, row=index % 1000))
    indices = [item['index'] for item in entries]
    pixels, labels = cifar.read_rows('train', indices)
    if any(item['label'] != int(label) for item, label in zip(entries, labels)):
        raise ValueError('Calibration selection label disagrees with the original record')
    images = cifar.normalize(pixels, normalization=(cifar.MEAN, cifar.STD))
    if images.shape != (3, 3, 32, 32) or not np.isfinite(images).all():
        raise ValueError('Expected three finite original CIFAR images')
    source = dict(records=str(records), selections=str(selections), split='train',
                  indices=indices, sample_ids=['train:' + str(i) for i in indices],
                  labels=[int(label) for label in labels], entries=entries,
                  normalization=dict(mean=list(cifar.MEAN), std=list(cifar.STD)),
                  dataset_metadata=expected)
    return images, labels, source


def finalize_info(imported, model, metadata, source_record):
    """Retain inferred intake metadata and replace only the temporary split."""
    from ..data import cifar
    from ..data.samples import legacy_cifar

    legacy_info, sample_source = legacy_cifar(
        imported.info.graph, normalization=(cifar.MEAN, cifar.STD))
    source = copy.deepcopy(imported.info.source)
    source.update(factory=str(model), original_source=str(Path(metadata['source']).resolve()),
                  checkpoint=str(Path(metadata['checkpoint']).resolve()),
                  inference_preprocessing_source=str(
                      Path(metadata['inference_preprocessing_source']).resolve()),
                  spec_revision=metadata['spec_revision'],
                  normalization=copy.deepcopy(source_record['normalization']),
                  calibration_source=copy.deepcopy(source_record),
                  preprocessing_provenance='official inference test',
                  training_preprocessing_verified=False)
    info = replace(imported.info, source=source,
                   input_handling=dict(imported.info.input_handling,
                                       internal_regression_source='legacy_cifar'),
                   split=copy.deepcopy(legacy_info.split))
    return info, sample_source


def _json_values(array):
    return [[float(value) if math.isfinite(float(value)) else None for value in row]
            for row in array]


def compare_graph(graph, images, references):
    """Compare named GraphExec outputs to the isolated original-model references."""
    import numpy as np
    from ..intake import graph_exec

    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise ValueError('R13 requires one inferred graph input and output')
    if len(images) != 3 or len(references) != 3:
        raise ValueError('R13 graph checks require the three original probes')
    observations = []
    for index in range(3):
        reference = np.asarray(references[index], dtype=np.float64)
        output = np.asarray(graph_exec.forward(
            graph, {graph.inputs[0]: images[index:index + 1]})[graph.outputs[0]],
            dtype=np.float64)
        shape_ok = reference.shape == output.shape == (1, 10)
        finite = bool(np.isfinite(reference).all() and np.isfinite(output).all())
        absolute = np.abs(output - reference) if shape_ok and finite else None
        passed = bool(absolute is not None and
                      np.all(absolute <= 1e-12 + 1e-12 * np.abs(reference)))
        observations.append(dict(probe_index=index, passed=passed, finite=finite,
                                 shape_ok=shape_ok, input_name=graph.inputs[0],
                                 output_name=graph.outputs[0],
                                 reference_shape=list(reference.shape), output_shape=list(output.shape),
                                 reference=_json_values(reference) if reference.ndim == 2 else None,
                                 output=_json_values(output) if output.ndim == 2 else None,
                                 max_absolute_error=float(absolute.max()) if absolute is not None else None))
        if not passed:
            break
    return dict(passed=len(observations) == 3 and all(item['passed'] for item in observations),
                criterion=GRAPH_CRITERION, samples=observations)


def _inventory(module):
    import torch

    def tensor_record(name, value):
        return dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                    finite=bool(torch.isfinite(value).all()))

    return dict(modules=[dict(name=name, type=type(child).__module__ + '.' + type(child).__qualname__)
                         for name, child in module.named_modules()],
                parameters=[tensor_record(name, value) for name, value in module.named_parameters()],
                buffers=[tensor_record(name, value) for name, value in module.named_buffers()])


def _shape_check(info):
    graph = info.graph
    inputs = {name: list(graph.tensors[name].shape) for name in graph.inputs}
    outputs = {name: list(graph.tensors[name].shape) for name in graph.outputs}
    return dict(ok=len(inputs) == len(outputs) == 1
                and next(iter(inputs.values()), None) == [1, 3, 32, 32]
                and next(iter(outputs.values()), None) == [1, 10],
                inputs=inputs, outputs=outputs)


def _probe_observation(events_path):
    events = [item for item in read_events(events_path)
              if item.get('stage_id') == 'intake.real_probe']
    started = [item for item in events if item.get('type') == 'begin']
    ended = [item for item in events if item.get('type') == 'end']
    passed = (None if not started else len(started) == 3 and len(ended) == 3
              and all(item.get('status') == 'ok' for item in ended))
    return dict(passed=passed, attempted=len(started),
                completed=sum(item.get('status') == 'ok' for item in ended),
                criterion=REAL_PROBE_CRITERION)


def _persist(result, work_dir):
    write_json(Path(work_dir) / 'worker_result.json', result)


def run(args, recorder, result):
    import numpy as np
    import torch
    from ..data.samples import Samples
    from ..intake import coverage
    from ..intake.probe import import_torch, load_callable
    from ..models.info import ModelInfo
    from .r13_common_models import FACTORIES

    torch.set_num_threads(1)
    if args.model not in FACTORIES:
        raise ValueError('R13 accepts only the three specified original-model factories')
    metadata = FACTORIES[args.model]
    for key in ('source', 'checkpoint', 'inference_preprocessing_source'):
        if not Path(metadata[key]).is_file():
            raise FileNotFoundError('Missing original model prerequisite: ' + str(metadata[key]))
    with recorder.span('read_samples', artifact=str(args.inputs)):
        images, labels, source_record = read_calibration(args.inputs, args.selections)
        samples = Samples(kind='tensor', items=[{'input': images[i:i + 1]} for i in range(3)],
                          ids=source_record['sample_ids'], labels=labels)
        probe_path = args.work_dir / 'training_probes.npz'
        np.savez(probe_path, input=images, labels=labels,
                 indices=np.asarray(source_record['indices'], dtype=np.int64))
        source_path = args.work_dir / 'training_probe_source.json'
        write_json(source_path, source_record)
        result.update(source=source_record)
        result['artifacts'].update(training_probes=str(probe_path), training_probe_source=str(source_path))
    with recorder.span('load_model', artifact=str(metadata['checkpoint'])):
        module, name = load_callable(args.model)
    with recorder.span('model_inventory'):
        inventory = _inventory(module)
        inventory_path = args.work_dir / 'model_inventory.json'
        write_json(inventory_path, inventory)
        result['artifacts']['model_inventory'] = str(inventory_path)
        result['model_inventory'] = inventory
        if not all(item['finite'] for item in inventory['parameters'] + inventory['buffers']):
            raise WorkerFailure('The original model contains non-finite parameters or buffers')
    temporary_split = dict(kind='legacy_cifar_intake_only', calibration=[0, 1, 2],
                           heldout=[], confirm=[], report=[], sample_ids=list(samples.ids))
    result['import_completed'] = False
    with recorder.span('import_model', artifact=args.model):
        imported = import_torch(module, samples, name=name, source=args.model,
                                split=temporary_split, observer=recorder)
    result['import_completed'] = True
    result['real_probes'] = _probe_observation(args.work_dir / 'events.jsonl')
    result['real_probes_passed'] = result['real_probes']['passed']
    if result['real_probes_passed'] is not True:
        raise WorkerFailure('The three real import probes did not complete successfully')
    with recorder.span('finalize_model_info'):
        info, sample_source = finalize_info(imported, args.model, metadata, source_record)
        # The three-item exchange source never interprets final legacy test indices.
        imported.info, imported.samples = info, sample_source
        del samples
        result['shape_check'] = _shape_check(info)
        result['intake_stats'] = copy.deepcopy(getattr(info.graph, 'intake_stats', {}))
        result['model_source'] = copy.deepcopy(info.source)
        result['split'] = copy.deepcopy(info.split)
        if result['shape_check']['ok'] is not True:
            raise WorkerFailure('The inferred model input/output shapes violate the R13 contract')
    with recorder.span('original_reference'):
        reference_model = copy.deepcopy(module).eval().double()
        references = []
        with torch.no_grad():
            for index in range(3):
                output = reference_model(torch.from_numpy(images[index:index + 1]))
                if not isinstance(output, torch.Tensor):
                    raise WorkerFailure('The original model must return one tensor')
                reference = output.detach().cpu().numpy().copy()
                if reference.shape != (1, 10) or not np.isfinite(reference).all():
                    raise WorkerFailure('The original model returned an invalid or non-finite reference')
                references.append(reference)
        del reference_model
    with recorder.span('graph_semantics'):
        result['graph_math'] = compare_graph(info.graph, images, references)
        result['graph_math_passed'] = result['graph_math']['passed']
        write_json(args.work_dir / 'graph_math.json', result['graph_math'])
        result['artifacts']['graph_math'] = str(args.work_dir / 'graph_math.json')
        if result['graph_math_passed'] is not True:
            raise WorkerFailure('PT-Graph differs from the original float64 model', status='math_failed')
    with recorder.span('static_coverage'):
        result['coverage_report'] = coverage.report(info.graph)
        result['coverage_check'] = coverage.check(info.graph, slots=16384)
        result['static_coverage'] = bool(result['coverage_check']['ok'])
        diagnostics = args.work_dir / 'coverage.json'
        write_json(diagnostics, dict(report=result['coverage_report'], check=result['coverage_check']))
        result['artifacts']['coverage'] = str(diagnostics)
        if result['static_coverage'] is not True:
            raise WorkerFailure('PT-Graph static coverage reported diagnostics', status='coverage_failed')
    saved = args.work_dir / 'model_info'
    with recorder.span('save_model_info', artifact=str(saved)):
        info.save(saved)
        result['artifacts']['model_info'] = str(saved)
    with recorder.span('load_model_info', artifact=str(saved)):
        restored = ModelInfo.load(saved)
        retained = dict(source=restored.source == info.source, split=restored.split == info.split,
                        inputs=restored.inputs == info.inputs,
                        input_handling=restored.input_handling == info.input_handling,
                        sequence=restored.sequence == info.sequence, task=restored.task == info.task,
                        metric=restored.metric == info.metric, threshold=restored.threshold == info.threshold,
                        assumptions=restored.assumptions == info.assumptions,
                        graph_inputs=restored.graph.inputs == info.graph.inputs,
                        graph_outputs=restored.graph.outputs == info.graph.outputs,
                        graph_shapes=_shape_check(restored) == result['shape_check'],
                        intake_stats=getattr(restored.graph, 'intake_stats', {}) ==
                                     getattr(info.graph, 'intake_stats', {}))
        result['roundtrip_metadata'] = retained
        if not all(retained.values()):
            result['roundtrip_passed'] = False
            raise WorkerFailure('Saved ModelInfo metadata changed on readback', status='roundtrip_failed')
    with recorder.span('roundtrip_semantics'):
        result['roundtrip_math'] = compare_graph(restored.graph, images, references)
        result['roundtrip_passed'] = result['roundtrip_math']['passed']
        readback_path = args.work_dir / 'roundtrip_math.json'
        write_json(readback_path, dict(metadata=result['roundtrip_metadata'], math=result['roundtrip_math']))
        result['artifacts']['roundtrip_math'] = str(readback_path)
        if result['roundtrip_passed'] is not True:
            raise WorkerFailure('Readback graph differs from the original model', status='roundtrip_failed')


def _memory_error(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, MemoryError):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--selections', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if (args.work_dir / 'worker_result.json').exists():
        raise FileExistsError('Existing R13 worker results are preserved')
    result = dict(round='R13', model=args.model, status='error', error=None, failed_phase=None,
                  pid=os.getpid(), ppid=os.getppid(), import_completed=None,
                  real_probes_passed=None, real_probe_criterion=REAL_PROBE_CRITERION,
                  graph_math_passed=None, roundtrip_passed=None, static_coverage=None,
                  lowering_ready=None, compiled_bootstraps=None, predicted_latency=None,
                  CKKS_latency=None, llm_calls=0, real_ckks_attempts=0, artifacts={})
    recorder = StageRecorder(args.work_dir / 'events.jsonl')
    old_handler = None
    try:
        def timeout(signum, frame):
            raise WorkerFailure('R13 worker exceeded 1800 seconds', status='timeout')
        old_handler = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(INTERNAL_SECONDS)
        with recorder.span('worker', model=args.model):
            run(args, recorder, result)
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = ('resource_limit' if _memory_error(exc) else
                            'dependency_missing' if isinstance(exc, (ModuleNotFoundError, FileNotFoundError)) else
                            getattr(exc, 'status', 'error'))
        result['error'] = dict(type=type(exc).__name__, message=str(exc)[:2000])
        if hasattr(exc, 'entries'):
            result['error']['diags'] = exc.entries
        errors = [event for event in read_events(args.work_dir / 'events.jsonl')
                  if event.get('type') == 'end' and event.get('status') == 'error']
        result['failed_phase'] = errors[0]['stage_id'] if errors else 'worker'
    finally:
        if old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        result['real_probes'] = _probe_observation(args.work_dir / 'events.jsonl')
        result['real_probes_passed'] = result['real_probes']['passed']
        result['resource_ru_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _persist(result, args.work_dir)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
