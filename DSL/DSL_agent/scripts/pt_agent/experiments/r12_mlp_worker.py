"""Path-driven inner workers for the approved R12 official MLP experiment."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import struct
import sys
import time
from types import SimpleNamespace


SAMPLES_FORMAT = 'pt-agent-r12-mlp-samples-v1'
PROFILE_FORMAT = 'pt-agent-r12-per-op-profile-v1'
CONTEXT_ID = 'poseidon-agent-n15-q13-p1-tc128-v1'
COMPILE_MODES = ('dacapo-optimized', 'eva')
SHAPES = {'linear1.weight': (100, 784), 'linear1.bias': (100,),
          'linear2.weight': (10, 100), 'linear2.bias': (10,)}
UNUSED_BN_SHAPES = {
    f'{layer}.{field}': (() if field == 'num_batches_tracked' else (width,))
    for layer, width in (('bn1', 100), ('bn2', 10))
    for field in ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked')
}
PHASE_ARGUMENTS = {
    'prepare': ('model', 'mnist', 'source'),
    'trace': ('model', 'source'),
    'compile': ('input', 'hecate_opt', 'compiler_profile', 'operator_spec'),
    'build': ('repo_root', 'runtime_root', 'client_source_dir',
              'generated_include_dir', 'native_library', 'build_dir'),
    'verify-client': ('client', 'plan', 'spec', 'inputs', 'native_library'),
    'execute': ('client', 'plan', 'spec', 'inputs', 'native_library', 'verification'),
}


class WorkerFailure(RuntimeError):
    def __init__(self, message, *, status='error'):
        super().__init__(message)
        self.status = status


def _read_json(path):
    def reject(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def _weights_checked(weights):
    import numpy as np

    if not isinstance(weights, dict):
        raise ValueError('MLP weights must be a mapping')
    checked = {}
    for key, shape in SHAPES.items():
        if key not in weights:
            raise ValueError('Missing checkpoint tensor: ' + key)
        value = np.asarray(weights[key], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid shape or non-finite checkpoint tensor: ' + key)
        checked[key] = value
    return checked


def _load_checkpoint(path):
    """Check the known 14 tensors once; return four used arrays and BN metadata."""
    import torch

    original = torch.load(str(Path(path)), map_location='cpu', weights_only=True)
    expected = set(SHAPES) | set(UNUSED_BN_SHAPES)
    if not isinstance(original, dict) or set(original) != expected:
        raise ValueError('The original checkpoint must contain exactly the 14 known MLP tensors')
    weights = {}
    for key, shape in SHAPES.items():
        value = original.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError('Invalid or missing checkpoint tensor: ' + key)
        weights[key] = value.detach().to(dtype=torch.float64, device='cpu').numpy()
    weights = _weights_checked(weights)
    unused = []
    for key, shape in UNUSED_BN_SHAPES.items():
        value = original[key]
        dtype = torch.int64 if key.endswith('.num_batches_tracked') else torch.float32
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid shape, dtype or non-finite unused BN tensor: ' + key)
        unused.append(dict(name=key, shape=list(shape),
                           dtype=str(value.dtype).removeprefix('torch.'),
                           unused_by_artifact=True))
    return weights, unused


def load_checkpoint(path) -> dict:
    """Return only the four linear arrays used by the original FHE artifact."""
    return _load_checkpoint(path)[0]


def read_mnist_images(path):
    """Return original training rows 0, 1 and 2 as three float64 784-vectors."""
    import numpy as np

    source = Path(path).resolve()
    opener = gzip.open if source.suffix == '.gz' else open
    with opener(source, 'rb') as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError('Incomplete MNIST image header')
        magic, count, rows, cols = struct.unpack('>IIII', header)
        if magic != 2051 or (rows, cols) != (28, 28) or count < 3:
            raise ValueError('MNIST requires magic 2051, 28 by 28, and at least three rows')
        data = stream.read(3 * 784)
        if len(data) != 3 * 784:
            raise ValueError('Incomplete first three MNIST images')
    images = np.frombuffer(data, dtype=np.uint8).reshape(3, 784).astype(np.float64) / 255
    return images, dict(source=str(source), magic=magic, rows=rows, cols=cols,
                        indices=[0, 1, 2])


def repeat_slots(values, slots=16384):
    """Repeat a non-empty original vector and truncate it at the slot count."""
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError('Slot source must be a non-empty finite vector')
    if type(slots) is not int or slots <= 0:
        raise ValueError('Slot count must be a positive integer')
    return np.resize(array, slots)


def _input_checked(input784):
    import numpy as np

    array = np.asarray(input784, dtype=np.float64)
    if array.shape != (784,) or not np.isfinite(array).all():
        raise ValueError('Expected one finite 784-element image')
    if (array < 0).any() or (array > 1).any():
        raise ValueError('Image values must be in [0, 1]')
    return array


def pack_input(input784):
    import numpy as np

    return repeat_slots(np.concatenate((_input_checked(input784), np.zeros(16))))


def matrix_reference(input784, weights):
    image, values = _input_checked(input784), _weights_checked(weights)
    hidden = image @ values['linear1.weight'].T + values['linear1.bias']
    output = (hidden * hidden) @ values['linear2.weight'].T + values['linear2.bias']
    if not all(_finite_number(float(value)) for value in output):
        raise ValueError('The matrix reference is non-finite')
    return output


def packed_reference(source, input784, weights):
    """Execute only the artifact's two arithmetic helper function definitions."""
    import numpy as np

    class Slots:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float64)

        def rotate(self, step):
            return Slots(np.roll(self.values, -step))

        def __add__(self, other):
            return Slots(self.values + (other.values if isinstance(other, Slots) else other))

        __radd__ = __add__

        def __mul__(self, other):
            return Slots(self.values * (other.values if isinstance(other, Slots) else other))

        __rmul__ = __mul__

    source = Path(source)
    tree = ast.parse(source.read_text(), filename=str(source))
    names = ('input_to_layer_MNIST', 'layer_to_output_MNIST')
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(functions) != 2 or sorted(node.name for node in functions) != sorted(names):
        raise ValueError('Expected the two unique original MLP arithmetic helpers')
    for node in functions:
        if (node.decorator_list or node.args.defaults
                or any(value is not None for value in node.args.kw_defaults)
                or node.returns is not None
                or any(arg.annotation is not None for arg in
                       node.args.posonlyargs + node.args.args + node.args.kwonlyargs)):
            raise ValueError('Arithmetic helper definitions must have no executable definition metadata')
    selected = ast.Module(body=functions, type_ignores=[])
    namespace = {'__builtins__': {'range': range},
                 'hc': SimpleNamespace(Plain=lambda values: Slots(repeat_slots(values)))}
    exec(compile(selected, str(source), 'exec'), namespace)
    values = _weights_checked(weights)
    image = Slots(pack_input(input784))
    hidden = (namespace[names[0]](image, values['linear1.weight'])
              + Slots(repeat_slots(values['linear1.bias'])))
    output = (namespace[names[1]](hidden * hidden, values['linear2.weight'])
              + Slots(repeat_slots(values['linear2.bias'])))
    result = output.values[:10]
    if result.shape != (10,) or not np.isfinite(result).all():
        raise ValueError('The original packed arithmetic returned invalid output')
    return result


def validate_samples_record(record) -> dict:
    """Validate the fixed internal input protocol without importing model code."""
    if not isinstance(record, dict) or record.get('format') != SAMPLES_FORMAT:
        raise ValueError('Invalid R12 samples format')
    source = record.get('source')
    if not isinstance(source, dict) or source.get('split') != 'train' or source.get('normalization') != 'float64(uint8)/255':
        raise ValueError('Invalid R12 sample source or normalization')
    for key in ('model', 'mnist'):
        value = source.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError('Sample source paths must be absolute')
    samples = record.get('samples')
    if not isinstance(samples, list) or len(samples) != 3:
        raise ValueError('R12 requires exactly three samples')
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or type(sample.get('index')) is not int or sample['index'] != index:
            raise ValueError('R12 sample indices must be 0, 1, 2 in order')
        image, reference = sample.get('input784'), sample.get('reference10')
        if (not isinstance(image, list) or len(image) != 784
                or not all(_finite_number(value) and 0 <= value <= 1 for value in image)):
            raise ValueError('Invalid R12 input784')
        if (not isinstance(reference, list) or len(reference) != 10
                or not all(_finite_number(value) for value in reference)):
            raise ValueError('Invalid R12 reference10')
    return record


def check_plan_contract(record) -> dict:
    """Reject unsupported JSON structure before launching any crypto client."""
    if not isinstance(record, dict) or type(record.get('format_version')) is not int or record['format_version'] != 1:
        raise ValueError('R12 requires RuntimePlan format 1')
    if 'plaintext_bundle' in record:
        raise ValueError('R12 requires inline constants without a plaintext bundle')
    target = record.get('target', {})
    if (target.get('target_id') != 'poseidon-ckks-cpu'
            or target.get('world_size') != 1 or target.get('device_counts') != [0]):
        raise ValueError('R12 requires one CPU host with no devices')
    for name in ('external_inputs', 'final_outputs'):
        if not isinstance(record.get(name), list) or len(record[name]) != 1:
            raise ValueError('R12 requires one external input and one final output')
    values = record.get('values')
    if not isinstance(values, list) or not values:
        raise ValueError('RuntimePlan values are required')
    by_id = {}
    for value in values:
        if not isinstance(value, dict) or value.get('id') in by_id:
            raise ValueError('RuntimePlan values must have unique IDs')
        if value.get('place') != {'kind': 'host', 'rank': 0}:
            raise ValueError('All RuntimePlan values must reside on host rank 0')
        if (value.get('context') != CONTEXT_ID or not _nonnegative_int(value.get('level'))
                or value['level'] > 12 or not _nonnegative_int(value.get('scale_log2'))
                or value.get('ntt') is not True):
            raise ValueError('Unsupported R12 context, level, scale, or NTT descriptor')
        by_id[value.get('id')] = value
    for name in ('external_inputs', 'final_outputs'):
        if record[name][0] not in by_id or by_id[record[name][0]].get('kind') != 'ciphertext':
            raise ValueError('R12 external input and final output must be ciphertext values')
    rotations = set()
    for section in ('initialization', 'execution', 'finalization'):
        steps = record.get(section)
        if not isinstance(steps, list):
            raise ValueError('RuntimePlan instruction sections must be arrays')
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError('Invalid RuntimePlan instruction')
            kind = step.get('kind')
            if kind == 'encode':
                payload = step.get('payload', {})
                entries = payload.get('values')
                if (section != 'initialization' or payload.get('kind') != 'inline'
                        or not isinstance(entries, list) or not entries
                        or not all(_finite_number(entry) for entry in entries)):
                    raise ValueError('Every Encode must contain a non-empty finite inline payload')
            elif kind == 'compute':
                if step.get('place') != {'kind': 'host', 'rank': 0}:
                    raise ValueError('Compute must execute on host rank 0')
                if step.get('op') == 'boot':
                    raise ValueError('R12 does not permit a bootstrap requirement')
                if step.get('op') == 'rotate':
                    rotation = step.get('attrs', {}).get('steps')
                    if type(rotation) is not int or rotation == 0:
                        raise ValueError('Invalid rotation step')
                    rotations.add(rotation)
            else:
                raise ValueError('R12 requires only host Encode and Compute instructions')
    return dict(ok=True, boots=0, rotation_steps=sorted(rotations), context_id=CONTEXT_ID,
                external_inputs=1, final_outputs=1)


def _key_string(key):
    fields = {'op', 'mode', 'input_levels', 'input_scale_log2', 'output_level',
              'output_scale_log2', 'rotation_step'}
    if not isinstance(key, dict) or set(key) != fields:
        raise ValueError('Invalid per-operation timing key')
    if key['mode'] not in ('compute', 'compute_reuse') or not isinstance(key['op'], str):
        raise ValueError('Invalid operation or call mode')
    for field in ('input_levels', 'input_scale_log2'):
        if not isinstance(key[field], list) or not all(_nonnegative_int(v) for v in key[field]):
            raise ValueError('Invalid input descriptors in timing key')
    if len(key['input_levels']) != len(key['input_scale_log2']):
        raise ValueError('Timing input level and scale counts differ')
    if not all(_nonnegative_int(key[field]) for field in ('output_level', 'output_scale_log2')):
        raise ValueError('Invalid output descriptor in timing key')
    step = key['rotation_step']
    is_rotation = key['op'].lower() == 'rotate'
    if (is_rotation and (type(step) is not int or step == 0)) or (not is_rotation and step is not None):
        raise ValueError('Invalid rotation step in timing key')
    return json.dumps(key, sort_keys=True, separators=(',', ':'))


def validate_observation(record, calls, profile) -> dict:
    """Read back the frozen native timing table and its reported predictions."""
    if not isinstance(record, dict) or not isinstance(calls, list):
        raise ValueError('Native observation and calls must be JSON records')
    grouped, sample_calls, next_index = {}, {0: [], 1: [], 2: []}, {0: 0, 1: 0, 2: 0}
    for call in calls:
        sample = call.get('sample_index')
        if type(sample) is not int or sample not in sample_calls or call.get('call_index') != next_index[sample]:
            raise ValueError('Per-operation calls must retain each sample call order')
        next_index[sample] += 1
        if not _nonnegative_int(call.get('elapsed_ns')) or call.get('status') not in ('ok', 'error'):
            raise ValueError('Invalid per-operation elapsed time or status')
        key = _key_string(call.get('key'))
        if call['status'] == 'ok':
            sample_calls[sample].append((key, call['elapsed_ns']))
            if sample == 0:
                entry = grouped.setdefault(key, {'call_count': 0, 'total_ns': 0})
                entry['call_count'] += 1
                entry['total_ns'] += call['elapsed_ns']
    samples = record.get('samples')
    if not isinstance(samples, list) or len(samples) > 3:
        raise ValueError('Invalid native sample observations')
    if any(sample.get('index') != index for index, sample in enumerate(samples)):
        raise ValueError('Native samples must retain original order')
    calibration_ok = bool(samples and samples[0].get('passed') is True
                          and samples[0].get('encrypted_execution_completed') is True)
    predictions = record.get('predictions')
    if not isinstance(predictions, list):
        raise ValueError('Native predictions must be an array')
    if not calibration_ok:
        if profile is not None or predictions or any(sample.get('encrypted_execution_attempted') is True for sample in samples[1:]):
            raise ValueError('Failed calibration must not freeze a profile or execute later samples')
        return dict(ok=True, profile_checked=False, predictions_checked=0)
    if (not isinstance(profile, dict) or profile.get('format') != PROFILE_FORMAT
            or profile.get('time_unit') != 'ns' or profile.get('context_id') != CONTEXT_ID
            or profile.get('calibration_sample_index') != 0 or profile.get('frozen') is not True
            or profile.get('bootstrap_latency_ns') is not None):
        raise ValueError('Invalid frozen R12 timing profile')
    entries, means = profile.get('entries'), {}
    if not isinstance(entries, list):
        raise ValueError('Frozen timing entries must be an array')
    for entry in entries:
        key = _key_string(entry.get('key'))
        if key in means or key not in grouped:
            raise ValueError('Frozen timing table contains duplicate or uncalibrated operations')
        expected = grouped[key]
        mean = expected['total_ns'] / expected['call_count']
        if (entry.get('call_count') != expected['call_count']
                or entry.get('total_ns') != expected['total_ns']
                or not _finite_number(entry.get('mean_ns')) or entry['mean_ns'] != mean):
            raise ValueError('Frozen timing table differs from calibration calls')
        means[key] = mean
    if set(means) != set(grouped):
        raise ValueError('Frozen timing table is missing calibration operations')
    predicted_indices = set()
    for prediction in predictions:
        index = prediction.get('sample_index')
        if type(index) is not int or index not in (1, 2) or index in predicted_indices:
            raise ValueError('Invalid or duplicate prediction sample')
        predicted_indices.add(index)
        if index >= len(samples) or samples[index].get('encrypted_execution_completed') is not True:
            raise ValueError('Prediction requires an actually completed sample')
        own_calls = sample_calls[index]
        missing = {key for key, _ in own_calls if key not in means}
        reported_missing = prediction.get('missing_keys')
        if not isinstance(reported_missing, list) or {_key_string(key) for key in reported_missing} != missing or len(reported_missing) != len(missing):
            raise ValueError('Prediction missing_keys differ from observed operations')
        measured = sum(elapsed for _, elapsed in own_calls)
        if prediction.get('measured_compute_ns') != measured:
            raise ValueError('Measured prediction time differs from successful calls')
        runtime = samples[index].get('runtime_run_ns')
        if prediction.get('runtime_run_ns') != runtime:
            raise ValueError('Prediction runtime interval differs from its sample')
        if missing:
            if (prediction.get('prediction_complete') is not False
                    or prediction.get('predicted_compute_ns') is not None
                    or prediction.get('relative_abs_error') is not None):
                raise ValueError('An incomplete prediction must retain null estimates')
        else:
            predicted = 0.0
            for key, _ in own_calls:
                predicted += means[key]
            relative = abs(predicted - measured) / measured if measured else None
            if (prediction.get('prediction_complete') is not True
                    or prediction.get('predicted_compute_ns') != predicted
                    or prediction.get('relative_abs_error') != relative):
                raise ValueError('Prediction does not use the frozen calibration table')
    expected_indices = {sample['index'] for sample in samples[1:]
                        if sample.get('encrypted_execution_completed') is True}
    if predicted_indices != expected_indices:
        raise ValueError('Completed observation samples must retain their predictions')
    return dict(ok=True, profile_checked=True, predictions_checked=len(predictions))


def _bind_source(model, source):
    model, source = Path(model).resolve(), Path(source).resolve()
    expected = source.parent.parent / 'data/mlp.model'
    if source.name != 'MLP.py' or source.parent.name != 'benchmarks' or model != expected.resolve():
        raise WorkerFailure('R12 requires the original source-relative official MLP checkpoint',
                            status='unsupported_contract')
    if not model.is_file() or not source.is_file():
        raise FileNotFoundError('Missing original model or source script')
    return model, source


def _prepare(args, recorder, result):
    from .r10_paper_metrics import write_json

    model, source = _bind_source(args.model, args.source)
    result['model'] = dict(source=str(model), source_script=str(source), shapes=None,
                           finite=None, checkpoint_keys=None, unused_checkpoint_tensors=None)
    with recorder.span('prepare.read_checkpoint', artifact=str(model)):
        weights, unused = _load_checkpoint(model)
        result['model'].update(shapes={key: list(value.shape) for key, value in weights.items()},
                                finite=True, checkpoint_keys=list(weights) + [entry['name'] for entry in unused],
                                unused_checkpoint_tensors=unused)
    with recorder.span('prepare.read_mnist', artifact=str(args.mnist)):
        images, metadata = read_mnist_images(args.mnist)
        result['mnist'] = metadata
    math_result = dict(atol=1e-12, rtol=1e-12, passed=None, samples=[])
    result['math'] = math_result
    samples = []
    with recorder.span('prepare.math'):
        for index, image in enumerate(images):
            with recorder.span('prepare.math.sample', sample_index=index):
                reference = matrix_reference(image, weights)
                packed = packed_reference(source, image, weights)
                difference = abs(packed - reference)
                passed = bool((difference <= 1e-12 + 1e-12 * abs(reference)).all())
                math_result['samples'].append(dict(index=index, packed_output10=packed.tolist(),
                                                    abs_error10=difference.tolist(), passed=passed))
                samples.append(dict(index=index, input784=image.tolist(), reference10=reference.tolist()))
        math_result['passed'] = all(sample['passed'] for sample in math_result['samples'])
        if not math_result['passed']:
            raise WorkerFailure('Original packed arithmetic differs from the matrix reference', status='math_failed')
    record = dict(format=SAMPLES_FORMAT,
                  source=dict(model=str(model), mnist=str(args.mnist.resolve()), split='train',
                              normalization='float64(uint8)/255'), samples=samples)
    validate_samples_record(record)
    samples_path = args.work_dir / 'samples.json'
    with recorder.span('prepare.write_samples', artifact=str(samples_path)):
        write_json(samples_path, record)
    result['artifacts']['samples'] = str(samples_path)


def _trace(args, recorder, result):
    from .. import toolchain
    from ..manage.streams import resolve

    model, source = _bind_source(args.model, args.source)
    traced = args.work_dir / 'traced'
    traced.mkdir()
    command = [sys.executable, '-B', str(source)]
    result.update(command=command, model=str(model), source=str(source))
    with recorder.span('trace.official_source', wall_seconds=3600, artifact=str(traced)):
        observation = toolchain.trusted_run(args.work_dir, command, args.work_dir / 'trace.log',
            env_extra={**toolchain.package_trace_env(), 'PYTHONDONTWRITEBYTECODE': '1'}, wall_seconds=3600)
        result['process'] = observation
        if observation['exit'] != 0:
            raise WorkerFailure('Trusted original MLP trace failed',
                                status='timeout' if observation.get('timeout') else 'error')
    with recorder.span('trace.discover_mlir', artifact=str(traced)):
        candidates = [path for path in (traced / 'MLP.mlir', traced / 'MLP.mlir.gz') if path.is_file()]
        if len(candidates) != 1:
            raise WorkerFailure('Expected exactly one fresh traced/MLP.mlir or MLP.mlir.gz',
                                status='unsupported_contract')
        mlir = resolve(traced / 'MLP.mlir')
        if mlir.stat().st_size == 0:
            raise WorkerFailure('The original trace is empty')
        result['artifacts']['mlir'] = str(mlir.resolve())


def _compiler_prediction(plan, profile_path):
    """Keep the original compiler latency table separate from measured times."""
    table = _read_json(profile_path)['latencyTable']
    levels = {value['id']: value['level'] for value in plan['values']}
    names = dict(add_cc='add_double', sub_cc='add_double', add_cp='add_single',
                 sub_cp='add_single', negate='negate_single', rotate='rotate_single',
                 mul_cc='mul_double', mul_cp='mul_single', rescale='rescale_single',
                 mod_switch='modswitch_single', boot='bootstrap_single')
    by_op, untimed = {}, Counter()
    for step in plan['execution']:
        op = step.get('op')
        timings = table.get('earth.' + names[op]) if op in names else None
        inputs = step.get('inputs') or []
        level = levels.get(inputs[0]) if inputs else None
        if (not isinstance(timings, list) or type(level) is not int or not 0 <= level < len(timings)
                or not _finite_number(timings[level])):
            untimed[op or step.get('kind', 'unknown')] += 1
        else:
            by_op[op] = by_op.get(op, 0) + timings[level]
    return dict(source=str(Path(profile_path).resolve()), profile='original-compiler-profile',
                time_unit='us', prediction_complete=not untimed,
                latency_us=sum(by_op.values()) if not untimed else None,
                covered_subtotal_us=sum(by_op.values()), by_op=by_op, untimed=dict(untimed),
                bootstrap_latency_us=None)


def _compile(args, recorder, result):
    from .. import toolchain
    from ..manage.streams import resolve

    mode = getattr(args, 'compile_mode', 'dacapo-optimized')
    result.update(compile_mode=mode, spec_revision='R12b')
    if mode not in COMPILE_MODES:
        raise WorkerFailure('Unknown R12 compile mode: ' + str(mode), status='unsupported_contract')
    expected = toolchain.host_paths()
    profile, spec = args.compiler_profile.resolve(), args.operator_spec.resolve()
    if profile != Path(expected['profile']).resolve() or spec != Path(expected['spec']).resolve():
        raise WorkerFailure('R12 requires the existing original CPU compiler profile and operator spec',
                            status='unsupported_contract')
    source = resolve(args.input.resolve())
    if not source.is_file():
        raise FileNotFoundError(source)
    output = args.work_dir / 'lowered.mlir'
    command = toolchain.hecate_command(dacapo=(mode == 'dacapo-optimized'))
    replacements = {'/hecate-opt': str(args.hecate_opt.resolve()),
                    '/out/candidate_trace.mlir': str(source),
                    '--ckks-config=/profile.json': '--ckks-config=' + str(profile),
                    '--runtime-plan-inline-payload-max-bytes=4096':
                        '--runtime-plan-inline-payload-max-bytes=131072',
                    '/out/lowered.mlir': str(output)}
    for old, new in replacements.items():
        if command.count(old) != 1:
            raise WorkerFailure('Original compiler command changed: ' + old, status='unsupported_contract')
        command[command.index(old)] = new
    result.update(command=command, compiler_profile=str(profile), operator_spec=str(spec),
                  inline_payload_max_bytes=131072)
    plan_path = args.work_dir / 'lowered._hecate_MLP.runtime-plan.json'
    stage = 'compile.official_dacapo' if mode == 'dacapo-optimized' else 'compile.official_eva'
    with recorder.span(stage, wall_seconds=7200, artifact=str(plan_path)):
        observation = toolchain.trusted_run(args.work_dir, command, args.work_dir / 'compile.log',
            env_extra={'LD_LIBRARY_PATH': os.environ['HECATE_PYTHON_LIBRARY_PATH']}, wall_seconds=7200)
        result['process'] = observation
        if plan_path.is_file():
            result['artifacts'].update(plan=str(plan_path), spec=str(spec))
        if observation['exit'] != 0 or not plan_path.is_file():
            raise WorkerFailure('Original ' + mode + ' compilation failed or did not emit the MLP RuntimePlan',
                                status='timeout' if observation.get('timeout') else 'error')
    with recorder.span('compile.read_runtime_plan', artifact=str(plan_path)):
        plan = _read_json(plan_path)
        result['plan_stats'] = toolchain.plan_stats(args.work_dir, function='MLP')
        result['compiled_bootstraps'] = result['plan_stats']['boots']
        try:
            result['plan_contract'] = check_plan_contract(plan)
        except ValueError as exc:
            raise WorkerFailure(str(exc), status='unsupported_contract') from exc
        result['rotation_requirements'] = result['plan_contract']['rotation_steps']
        result['compiler_profile_prediction'] = _compiler_prediction(plan, profile)
        lowered = resolve(output)
        if lowered.is_file():
            result['artifacts']['lowered_mlir'] = str(lowered)


def _run_child(args, command, label, timeout, *, environment=None):
    from .r10_paper_metrics import run_process

    return run_process(command, cwd=args.work_dir, log=args.work_dir / (label + '.log'),
                       resources=args.work_dir / 'resources.jsonl', timeout=timeout, mem_gib=4,
                       fsize_gib=8, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                                        **(environment or {})})


def _require_child(observation, label):
    if (observation['exit_code'] != 0 or observation['timed_out']
            or observation['signal'] or observation['remaining_pids']):
        raise WorkerFailure(label + ' failed', status='timeout' if observation['timed_out'] else 'error')


def _build(args, recorder, result):
    build = args.build_dir.resolve()
    if not build.is_relative_to(args.work_dir):
        raise WorkerFailure('The isolated build directory must be inside this phase work directory',
                            status='unsupported_contract')
    if build.exists():
        raise FileExistsError('Existing build directory is preserved: ' + str(build))
    cache = args.generated_include_dir.resolve().parent / 'CMakeCache.txt'
    dependency_types = {'GMP_INCLUDE_DIR': 'PATH', 'GMPXX_INCLUDE_DIR': 'PATH',
                        'GMP_LIBRARY': 'FILEPATH', 'GMPXX_LIBRARY': 'FILEPATH'}
    dependencies = {}
    result.update(gmp_cache_source=str(cache), gmp_cache_values=None)
    with recorder.span('build.read_dependencies', artifact=str(cache)):
        if not cache.is_file():
            raise WorkerFailure('The existing generated-header build cache is missing',
                                status='dependency_missing')
        for line in cache.read_text().splitlines():
            if not line or line.startswith(('#', '//')):
                continue
            declaration, separator, value = line.partition('=')
            name, typed, kind = declaration.partition(':')
            if name not in dependency_types:
                continue
            if name in dependencies:
                raise WorkerFailure('Duplicate existing GMP cache entry: ' + name,
                                    status='dependency_missing')
            if not separator or not typed or kind != dependency_types[name] or not value:
                raise WorkerFailure('Invalid existing GMP cache type or value: ' + name,
                                    status='dependency_missing')
            path = Path(value)
            exists = path.is_dir() if kind == 'PATH' else path.is_file()
            if not path.is_absolute() or not exists:
                raise WorkerFailure('Existing GMP cache path is unavailable: ' + name,
                                    status='dependency_missing')
            dependencies[name] = value
        missing = set(dependency_types) - set(dependencies)
        if missing:
            raise WorkerFailure('Missing existing GMP cache entries: ' + ', '.join(sorted(missing)),
                                status='dependency_missing')
        result['gmp_cache_values'] = dependencies
    command = ['cmake', '-S', str(args.client_source_dir.resolve()), '-B', str(build),
               '-DPOSEIDON_ROOT=' + str(args.repo_root.resolve()),
               '-DCKKS_RUNTIME_ROOT=' + str(args.runtime_root.resolve()),
               '-DPOSEIDON_GENERATED_INCLUDE_DIR=' + str(args.generated_include_dir.resolve()),
               '-DPOSEIDON_NATIVE_LIBRARY=' + str(args.native_library.resolve()),
               *['-D' + name + '=' + dependencies[name] for name in dependency_types]]
    result['configure_command'] = command
    with recorder.span('build.configure', wall_seconds=1800, artifact=str(build)):
        result['configure'] = _run_child(args, command, 'configure', 1800)
        _require_child(result['configure'], 'CMake configure')
    command = ['cmake', '--build', str(build), '--parallel', '1', '--target', 'mlp_client']
    result['build_command'] = command
    with recorder.span('build.compile_client', wall_seconds=1800, artifact=str(build)):
        result['build'] = _run_child(args, command, 'build', 1800)
        _require_child(result['build'], 'CMake build')
    client = build / 'mlp_client'
    if not client.is_file() or not os.access(client, os.X_OK):
        raise WorkerFailure('The isolated build did not create an executable mlp_client')
    result['artifacts']['client'] = str(client)


def _client_environment(args):
    parent = str(args.native_library.resolve().parent)
    existing = os.environ.get('LD_LIBRARY_PATH', '')
    return {'LD_LIBRARY_PATH': parent + (os.pathsep + existing if existing else '')}


_GATE_DRIVER = '''import json, pathlib, sys, unittest
from pt_agent.tests._r12_mlp_cases import ClientGateCases
suite = unittest.defaultTestLoader.loadTestsFromTestCase(ClientGateCases)
expected = suite.countTestCases()
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = dict(expected=expected, tests_run=result.testsRun,
              failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
pathlib.Path(sys.argv[1]).write_text(json.dumps(report) + "\\n")
raise SystemExit(0 if result.wasSuccessful() and expected > 0 and result.testsRun == expected and not result.skipped else 1)
'''


def _path_record(args):
    return {name: str(getattr(args, name).resolve())
            for name in ('client', 'plan', 'spec', 'inputs', 'native_library')}


def _verify_client(args, recorder, result):
    result.update(_path_record(args), tests_run=None, failures=None, errors=None, skipped=None)
    report = args.work_dir / 'client_gate_report.json'
    fixtures = args.work_dir / 'client_gate_fixtures'
    fixtures.mkdir()
    environment = _client_environment(args)
    scripts = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = os.environ.get('PYTHONPATH', '')
    environment['PYTHONPATH'] = scripts + (os.pathsep + existing_pythonpath if existing_pythonpath else '')
    environment.update(PT_R12_CLIENT=result['client'], PT_R12_PLAN=result['plan'],
                       PT_R12_SPEC=result['spec'], PT_R12_INPUTS=result['inputs'],
                       PT_R12_TEST_TMP=str(fixtures))
    command = [sys.executable, '-B', '-c', _GATE_DRIVER, str(report)]
    result['command'] = command
    with recorder.span('verify_client.actual_binary', wall_seconds=3600, artifact=str(report)):
        result['process'] = _run_child(args, command, 'verify_client', 3600, environment=environment)
        if report.is_file():
            counts = _read_json(report)
            for name in ('tests_run', 'failures', 'errors', 'skipped'):
                result[name] = counts.get(name)
            result['expected_tests'] = counts.get('expected')
            result['artifacts']['test_report'] = str(report)
        _require_child(result['process'], 'Actual binary ClientGateCases')
        if (not _nonnegative_int(result['tests_run']) or result['tests_run'] <= 0
                or result['tests_run'] != result.get('expected_tests')
                or any(type(result[name]) is not int or result[name] != 0 for name in ('failures', 'errors', 'skipped'))):
            raise WorkerFailure('ClientGateCases must execute every case with no errors or skips')
    result['artifacts']['verification'] = str(args.work_dir / 'verify_client_result.json')


def _validate_verification(args):
    record = _read_json(args.verification)
    if (record.get('status') != 'ok' or not _nonnegative_int(record.get('tests_run'))
            or record['tests_run'] <= 0
            or any(type(record.get(name)) is not int or record[name] != 0 for name in ('failures', 'errors', 'skipped'))):
        raise WorkerFailure('The actual client gate did not complete successfully', status='unsupported_contract')
    for name, path in _path_record(args).items():
        if record.get(name) != path:
            raise WorkerFailure('Client gate path differs from this execution: ' + name,
                                status='unsupported_contract')
    return record


def _execute(args, recorder, result):
    result.update(_path_record(args), client_result=None)
    with recorder.span('execute.validate_verification', artifact=str(args.verification)):
        _validate_verification(args)
    output = args.work_dir / 'client_output.json'
    result['artifacts']['client_result'] = str(output)
    command = [str(args.client.resolve()), '--plan', str(args.plan.resolve()),
               '--spec', str(args.spec.resolve()), '--inputs', str(args.inputs.resolve()),
               '--output', str(output)]
    result['command'] = command
    with recorder.span('execute.native_client', wall_seconds=3600, artifact=str(output)):
        result['process'] = _run_child(args, command, 'execute', 3600,
                                       environment=_client_environment(args))
        if output.is_file():
            result['client_result'] = _read_json(output)
            calls_path = args.work_dir / 'per_op_calls.jsonl'
            profile_path = args.work_dir / 'per_op_profile.json'
            if calls_path.is_file():
                result['artifacts']['per_op_calls'] = str(calls_path)
            if profile_path.is_file():
                result['artifacts']['per_op_profile'] = str(profile_path)
        _require_child(result['process'], 'Real CKKS client')
        if result['client_result'] is None or result['client_result'].get('status') != 'ok':
            raise WorkerFailure('The real CKKS client did not report a successful result')
    with recorder.span('execute.validate_observation'):
        calls_path = args.work_dir / 'per_op_calls.jsonl'
        profile_path = args.work_dir / 'per_op_profile.json'
        if not calls_path.is_file() or not profile_path.is_file():
            raise WorkerFailure('A successful CKKS run must retain calls and the frozen operation profile')
        calls = [json.loads(line) for line in calls_path.read_text().splitlines() if line.strip()]
        result['observation_check'] = validate_observation(result['client_result'], calls,
                                                          _read_json(profile_path))


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
    parser.add_argument('--phase', choices=tuple(PHASE_ARGUMENTS), required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--compile-mode', choices=COMPILE_MODES, default='dacapo-optimized')
    for name in sorted({name for values in PHASE_ARGUMENTS.values() for name in values}):
        parser.add_argument('--' + name.replace('_', '-'), type=Path)
    args = parser.parse_args(argv)
    for name in PHASE_ARGUMENTS[args.phase]:
        if getattr(args, name) is None:
            parser.error('--' + name.replace('_', '-') + ' is required for ' + args.phase)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.work_dir / (args.phase.replace('-', '_') + '_result.json')
    if result_path.exists():
        raise FileExistsError('Existing phase result is preserved: ' + str(result_path))
    result = dict(phase=args.phase, status='error', error=None, artifacts={},
                  pid=os.getpid(), ppid=os.getppid(), llm_calls=0, spec_revision='R12b')
    if args.phase == 'prepare':
        result.update(model=None, mnist=None, math=None, artifacts={'samples': None})
    if args.phase == 'compile':
        result.update(plan_stats=None, plan_contract=None, compiled_bootstraps=None,
                      compiler_profile_prediction=None, rotation_requirements=None)
    if args.phase == 'execute':
        result['client_result'] = None
    started_ns = time.perf_counter_ns()
    old_alarm = None
    try:
        from .r10_paper_metrics import StageRecorder
        recorder = StageRecorder(args.work_dir / 'events.jsonl')
        if args.phase == 'prepare':
            def timeout(signum, frame):
                raise WorkerFailure('Preparation exceeded 3600 seconds', status='timeout')
            old_alarm = signal.signal(signal.SIGALRM, timeout)
            signal.alarm(3600)
        with recorder.span('worker', phase=args.phase):
            if shutil.disk_usage(args.work_dir).free < 10 * 1024**3:
                raise WorkerFailure('At least 10 GiB free disk space is required', status='resource_limit')
            {'prepare': _prepare, 'trace': _trace, 'compile': _compile, 'build': _build,
             'verify-client': _verify_client, 'execute': _execute}[args.phase](args, recorder, result)
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = ('resource_limit' if _memory_error(exc) else
                            'dependency_missing' if isinstance(exc, ModuleNotFoundError) else
                            getattr(exc, 'status', 'error'))
        result['error'] = dict(type=type(exc).__name__, message=str(exc))
        if args.phase == 'prepare':
            result['artifacts']['samples'] = None
    finally:
        if old_alarm is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_alarm)
        result['elapsed_ns'] = time.perf_counter_ns() - started_ns
        result['resource_ru_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        from .r10_paper_metrics import write_json
        write_json(result_path, result)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
