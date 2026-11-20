"""Official Orion fit/compile worker for the three original DaCapo CNNs.

The native centered margin1.5 is recorded separately from Rule63's endpoint
expansion. This worker performs no encrypted inference. Official initialization
does create the key material required by the unmodified Orion compile path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import importlib
import io
import json
import math
from pathlib import Path
import re
import resource
import sys
import time
import traceback
from types import SimpleNamespace

from .r10_paper_metrics import StageRecorder, read_events, summarize_events, write_json


FACTORY_NAMES = dict(resnet20='resnet20_silu', vgg16='vgg16_silu_avgpool',
                     alexnet='alexnet_silu_avgpool')
FACTORY_PREFIX = 'pt_agent.experiments.r13_common_models:'
MODE = 'orion-native-rule63-fit-compile'
MARGIN = 1.5
MAX_LEVEL = 5


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Expected JSON metadata object: ' + str(path))
    return value


def _persist(args, result):
    result['peak_rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    write_json(args.work_dir / 'worker_result.json', result)
    write_json(args.work_dir / 'phase_result.json', result)


@contextmanager
def _stage(args, recorder, result, name):
    result['active_stage'] = name
    started = time.perf_counter_ns()
    result.setdefault('stage_clocks_ns', {})[name] = dict(
        started_ns=started, ended_ns=None, elapsed_ns=None, elapsed_s=None)
    _update_clocks(result)
    _persist(args, result)
    try:
        with recorder.span(name):
            yield
    finally:
        ended = time.perf_counter_ns()
        result.setdefault('stage_clocks_ns', {})[name] = dict(
            started_ns=started, ended_ns=ended, elapsed_ns=ended-started,
            elapsed_s=(ended-started)/1e9)
        _update_clocks(result)
        _persist(args, result)


def _update_clocks(result):
    clocks = result.get('stage_clocks_ns', {})
    result['init_scheme_seconds'] = (clocks.get('init_scheme') or {}).get('elapsed_s')
    phases = [clocks.get(name) for name in ('fit', 'compile')]
    measured = [item['elapsed_ns'] for item in phases if item and item.get('elapsed_ns') is not None]
    known = sum(measured)/1e9 if measured else None
    open_phase = any(item and item.get('elapsed_ns') is None for item in phases)
    both_closed = all(item and item.get('elapsed_ns') is not None for item in phases)
    result['fit_compile_known_completed_seconds'] = known
    result['fit_compile_clock_incomplete'] = bool(open_phase or not both_closed)
    result['fit_compile_seconds'] = (
        None if open_phase or (result.get('status') == 'running' and not both_closed) else known)
    result['fit_compile_clock_scope'] = (
        'Observed official fit and compile calls, including elapsed time to failure; '
        'initialization/key setup and model/adapter preparation are separate')
    result['fit_compile_both_completed'] = bool(
        result.get('fit_completed') == 1 and result.get('compile_completed') == 1)


def source_contract(args):
    """Check saved metadata and original paths without restoring graph weights."""
    from ..data import cifar
    from .r13_common_models import FACTORIES

    factory = FACTORY_PREFIX + FACTORY_NAMES[args.model]
    canonical = FACTORIES[factory]
    for argument, field in (('checkpoint', 'checkpoint'), ('model_source', 'source')):
        supplied = Path(getattr(args, argument)).resolve()
        if supplied != Path(canonical[field]).resolve() or not supplied.is_file():
            raise ValueError('Original canonical model path differs: ' + argument)
    metadata_path = Path(args.model_info)
    if metadata_path.is_dir():
        metadata_path /= 'model_info.json'
    metadata = _read(metadata_path)
    source, entries, split = metadata.get('source') or {}, metadata.get('inputs'), metadata.get('split')
    expected_norm = dict(mean=list(cifar.MEAN), std=list(cifar.STD))
    if (source.get('factory') != factory or source.get('path') != factory
            or source.get('type') != 'torch' or source.get('spec_revision') != 'R13'
            or source.get('normalization') != expected_norm
            or source.get('preprocessing_provenance') != 'official inference test'
            or source.get('training_preprocessing_verified') is not False):
        raise ValueError('Original model/preprocessing metadata differs')
    for field, key in (('checkpoint', 'checkpoint'), ('original_source', 'source'),
                       ('inference_preprocessing_source', 'inference_preprocessing_source')):
        if Path(source.get(field, '')).resolve() != Path(canonical[key]).resolve():
            raise ValueError('Saved original model path differs: ' + field)
    if (not isinstance(entries, list) or len(entries) != 1
            or entries[0].get('shape') != [1, 3, 32, 32]
            or entries[0].get('dtype') != 'float64' or entries[0].get('kind') != 'features'
            or not isinstance(entries[0].get('name'), str) or not entries[0]['name']):
        raise ValueError('Original inferred input metadata differs')
    expected_split = dict(kind='legacy_cifar', heldout=list(cifar.HELDOUT),
                          confirm=list(cifar.CONFIRM), report=list(cifar.REPORT),
                          calibration='frozen training selection')
    if split != expected_split:
        raise ValueError('Original fixed CIFAR split differs')
    name = entries[0]['name']
    # LegacyCifar needs only the input name/shape. No Graph.load/weights.npz read.
    light = SimpleNamespace(source=source, split=split,
        graph=SimpleNamespace(inputs=[name], tensors={name: SimpleNamespace(shape=(1, 3, 32, 32))}))
    return factory, light, dict(model_info=str(metadata_path.resolve()), factory=factory,
        checkpoint=str(Path(args.checkpoint).resolve()), model_source=str(Path(args.model_source).resolve()),
        normalization=expected_norm, split=split, model_info_graph_loaded=False)


def config_contract(args):
    """Retain the native resnet recipe except LogQ length and centered margin."""
    import yaml

    base_path = Path(args.orion_root).resolve() / 'configs/resnet.yml'
    base = yaml.safe_load(base_path.read_text())
    expected = json.loads(json.dumps(base))
    expected['ckks_params']['LogQ'] = [55, 40, 40, 40, 40, 40]
    expected['orion']['margin'] = MARGIN
    supplied = _read(args.config)
    if supplied != expected:
        raise ValueError('Orion config must retain resnet.yml except LogQ6 and native margin1.5')
    if supplied['orion']['backend'] != 'lattigo' or supplied['orion']['io_mode'] != 'none':
        raise ValueError('Expected the existing Lattigo backend with native io_mode none')
    return supplied, dict(official_config=str(base_path), prepared_config=str(Path(args.config).resolve()),
        LogN=16, LogQ=expected['ckks_params']['LogQ'], LogP=expected['ckks_params']['LogP'],
        LogScale=40, H=192, RingType='standard', native_slots=32768, l_eff=MAX_LEVEL,
        native_input_level_max=MAX_LEVEL, native_input_level_selected=None,
        qwen48_input_level=8, qwen48_bootstrap_output_level=5, qwen48_scale_bits=43,
        qwen48_active_slots=16384, native_margin=MARGIN, same_full_parameters=False,
        domain_equivalent=False, comparison_scope='Native Orion placement availability at degree63 and l_eff5; no same-coefficient accuracy or latency comparison')


def _scalar(value):
    if hasattr(value, 'item'):
        value = value.item()
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Expected a finite Orion calibration scalar')
    return value


def activation_inventory(net):
    rows = []
    for name, module in net.named_modules():
        if not hasattr(module, 'degree'):
            continue
        degree = int(module.degree)
        depth = getattr(module, 'depth', None)
        depth = int(depth) if depth is not None else None
        row = dict(name=name, kind=type(module).__name__, degree=degree, depth=depth,
                   depth_exceeds_l_eff=(depth > MAX_LEVEL) if depth is not None else None)
        if all(hasattr(module, key) for key in ('input_min', 'input_max', 'low', 'high')):
            low, high = _scalar(module.input_min), _scalar(module.input_max)
            center, half = (low+high)/2, (high-low)/2
            row.update(input_min=low, input_max=high, native_domain=[_scalar(module.low), _scalar(module.high)],
                rule63_endpoint_domain=[MARGIN*low, MARGIN*high],
                symmetric_maxabs_domain=[-MARGIN*max(abs(low), abs(high)), MARGIN*max(abs(low), abs(high))],
                domain_equivalent=False, native_margin=MARGIN, native_margin_semantics='center +/- margin*half_range',
                symmetric_domain_covering_margin=((MARGIN*max(abs(low),abs(high))+abs(center))/half if half > 0 else None),
                prescale=_scalar(module.prescale), constant=_scalar(module.constant))
        rows.append(row)
    return rows


def _save_inventory(args, result, net, *, label='fitted'):
    rows = activation_inventory(net)
    name = 'activation-domains.json' if label == 'fitted' else 'post-compile-activation-domains.json'
    write_json(args.work_dir / name, rows)
    if label == 'fitted':
        result['activation_layers'] = rows
        result['first_depth_exceeding_l_eff'] = next((row for row in rows if row['depth_exceeds_l_eff']), None)
        result['activation_domain_source'] = str(args.work_dir / name)
    else:
        result['post_compile_activation_layers'] = rows


def _post_compile_inventory(args, result, net, primary_error):
    """Keep a native compiler exception authoritative if diagnostics also fail."""
    try:
        _save_inventory(args, result, net, label='post_compile')
    except Exception as exc:
        result.setdefault('secondary_errors', []).append(dict(
            stage='post_compile_inventory', type=type(exc).__name__, message=str(exc)))
        _persist(args, result)
        if primary_error is None:
            raise


class CompileOutput(io.TextIOBase):
    """Tee native output and retain only its explicit weighted-bootstrap count."""
    def __init__(self, stream, log):
        self.stream, self.log, self.buffer, self.weighted_boots = stream, log, '', None

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.buffer = (self.buffer + text)[-16384:]
        matches = re.findall(r'Network requires ([0-9]+) bootstrap operation', self.buffer)
        if matches:
            self.weighted_boots = int(matches[-1])
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def classify_exception(exc, active_stage, orion_root):
    frames = traceback.extract_tb(exc.__traceback__)
    official = Path(orion_root).resolve() / 'orion/core/auto_bootstrap.py'
    placement_origin = any(Path(item.filename).resolve() == official
                           and item.name == 'finally_solve_full_level_dag' for item in frames)
    if (active_stage == 'compile' and isinstance(exc, ValueError) and placement_origin
            and str(exc).startswith('Automatic bootstrap placement failed.')):
        return 'no_placement'
    return 'driver_error'


def run(args, recorder, result):
    import torch
    import orion
    from .r14_common_worker import _read_calibration
    from .r15_rule_validation_worker import check_math
    from .s1_orion_layer_translation import build_from_torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Official pool/compiler temporaries use torch.ones without a dtype.
    torch.set_default_dtype(torch.float64)
    result['torch_default_dtype'] = 'float64'
    net = None
    with _stage(args, recorder, result, 'source_contract'):
        if Path(orion.__file__).resolve().parent != Path(args.orion_root).resolve() / 'orion':
            raise ValueError('Imported Orion is outside the explicitly selected source tree')
        factory, info, proof = source_contract(args)
        config, mapping = config_contract(args)
        result.update(source=proof, native_parameter_mapping=mapping, same_full_parameters=False,
                      domain_equivalent=False, orion_version=getattr(orion, '__version__', None))
    with _stage(args, recorder, result, 'read_calibration'):
        arrays, labels, source = _read_calibration(
            SimpleNamespace(inputs=args.records, selections=args.selections), info)
        values = arrays[info.graph.inputs[0]]
        if values.shape != (1000, 3, 32, 32) or source.get('batch') != 1 or source.get('split') != 'train':
            raise ValueError('Expected exactly the original1000 training calibration samples, batch1')
        result.update(calibration_source=source, calibration_samples=1000, calibration_batch=1,
                      heldout_samples=0, confirm_samples=0, report_samples=0)
        tensors = torch.from_numpy(values)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensors),
            batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    with _stage(args, recorder, result, 'load_original'):
        module_name, function = factory.split(':', 1)
        original = getattr(importlib.import_module(module_name), function)().cpu().double()
        original.eval()
    with _stage(args, recorder, result, 'translate_layers'):
        net, report = build_from_torch(original, tensors[:1])
        net.double()
        net.eval()
        result['translation'] = report
    with _stage(args, recorder, result, 'strict_translation_math'):
        samples = []
        with torch.no_grad():
            for index in range(3):
                reference = original(tensors[index:index+1]).detach().cpu().numpy()
                observed = net(tensors[index:index+1]).detach().cpu().numpy()
                checked = check_math(reference, observed)
                samples.append(dict(index=index, sample_id=source['sample_ids'][index], **checked))
                result['translation_math'] = samples
                if checked['passed'] is not True:
                    raise ValueError('Translated model differs under the original strict float64 criterion')
        result['strict_translation_math_passed'] = True
    del original
    # Do not initialize fake keys or skip official preparation to reach placement.
    with _stage(args, recorder, result, 'init_scheme'):
        result['init_scheme_calls'] += 1
        _persist(args, result)
        orion.init_scheme(config)
        result['init_scheme_completed'] = 1
        result['official_initialization_key_setup_executed'] = True
    with _stage(args, recorder, result, 'fit'):
        result['fit_calls'] += 1
        _persist(args, result)
        with torch.no_grad():
            orion.fit(net, loader, batch_size=1)
        result['fit_completed'] = 1
        _save_inventory(args, result, net)
        if not result['activation_layers'] or any(row['degree'] != 63 for row in result['activation_layers']):
            raise ValueError('Translated activations must all retain degree63')
    # Depth>5 is diagnostic only: always call the actual official compiler.
    tee = None
    try:
        with _stage(args, recorder, result, 'compile'):
            result['compile_calls'] += 1
            _persist(args, result)
            with (args.work_dir / 'official-compile-stdout.log').open('w') as log:
                tee = CompileOutput(sys.stdout, log)
                with redirect_stdout(tee):
                    selected_level = orion.compile(net)
            result['compile_completed'] = 1
            result['native_parameter_mapping']['native_input_level_selected'] = int(selected_level)
            result['native_reported_weighted_boots'] = tee.weighted_boots
            result['compiled_execution_bootstraps'] = tee.weighted_boots
            result['compiled_bootstrap_count_scope'] = 'Official compiler-reported expanded bootstrap count; encrypted execution unobserved'
    finally:
        _post_compile_inventory(args, result, net, sys.exc_info()[1])
    result.update(status='ok', native_compile_accepted=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', required=True, choices=tuple(FACTORY_NAMES))
    for name in ('work-dir', 'checkpoint', 'model-source', 'model-info', 'records',
                 'selections', 'orion-root', 'config'):
        result.add_argument('--' + name, required=True, type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result = dict(status='running', model=args.model, mode=MODE, init_scheme_calls=0,
        init_scheme_completed=0, fit_calls=0, fit_completed=0, compile_calls=0,
        compile_completed=0, native_compile_accepted=False, stage_clocks_ns={},
        native_reported_weighted_boots=None, compiled_execution_bootstraps=None,
        actual_CKKS_executions=0, observed_encrypted_bootstrap_calls=None,
        ckks_inference_runs=0, init_may_generate_keys=True, official_initialization_key_setup_executed=False,
        pred_latency_s=None, full_inference_prediction_s=None, RuntimePlan=None,
        llm_calls=0, downloads=0, installations=0, error=None)
    recorder = StageRecorder(args.work_dir / 'events.jsonl')
    _persist(args, result)
    try:
        run(args, recorder, result)
    except BaseException as exc:
        status = classify_exception(exc, result.get('active_stage'), args.orion_root)
        if status == 'no_placement' and result.get('compile_calls') != 1:
            status = 'driver_error'
        error_detail = dict(type=type(exc).__name__, message=str(exc), stage=result.get('active_stage'))
        result.update(status=status, error=type(exc).__name__ + ': ' + str(exc), error_detail=error_detail)
        detail = traceback.format_exc()
        (args.work_dir / 'traceback.log').write_text(detail)
        write_json(args.work_dir / 'exception.json', dict(error_detail, traceback=detail))
        if status == 'no_placement':
            result['no_placement_proof'] = dict(compile_calls=result['compile_calls'],
                official_exception_origin=True, exception_type=type(exc).__name__,
                message=str(exc), source=str(Path(args.orion_root).resolve() / 'orion/core/auto_bootstrap.py'),
                function='finally_solve_full_level_dag',
                first_depth_exceeding_l_eff=result.get('first_depth_exceeding_l_eff'),
                traceback=str(args.work_dir / 'traceback.log'),
                classification_scope='Official no-placement exception; activation depth is a separate diagnostic and does not establish the cause',
                predicted_before_compile=False)
    finally:
        _update_clocks(result)
        timing = summarize_events(read_events(args.work_dir / 'events.jsonl'))
        write_json(args.work_dir / 'timing.json', timing)
        result['timing'] = str(args.work_dir / 'timing.json')
        result['worker_events'] = str(args.work_dir / 'events.jsonl')
        _persist(args, result)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
