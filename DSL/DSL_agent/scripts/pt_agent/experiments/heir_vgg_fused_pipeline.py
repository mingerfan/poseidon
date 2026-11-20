"""Explicit VGG pool/conv fusion: saved-source export, strict math, native compile.

The original50-node model and the46-node adapter retain separate contracts.
Only the host owns the registry. The pure numeric child receives every path.
No quality selection, Go generation or ciphertext execution belongs to this job.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import sys
import threading
import time

from .completion_metrics import artifact_bytes, persist_case_completion
from .heir_initializer_probe import (durable_json, physical_observation, serial_lock,
    interruption_status, receive_signal, observe_termination)
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample
from . import heir_vgg_pool_conv_fusion as fusion_helper

MODEL = fusion_helper.MODEL
VARIANT = fusion_helper.VARIANT
ROUND = 'HEIR-VGG-FUSED'
MODE = 'explicit-four-pool-conv-fusion-export-strict-math-native-compile'
MODULE = 'pt_agent.experiments.heir_vgg_fused_pipeline'
GIB = 1024**3
TOTAL_SECONDS = 9600
WORKER_SECONDS = 1800
NIX_SECONDS = 1860
WORKER_OUTER_SECONDS = 1920
NATIVE_SECONDS = 7200
CASE_BYTES = 32*GIB
COMPLETION_RESERVE = 1024**2
MEMORY_GIB_CHOICES = (12, 24)
SOURCE_FIELDS = ('intake_result', 'graph_dir', 'program', 'frontend_result', 'probes',
                 'probe_source', 'pipeline', 'native_parameters')
COUNTERS = ('graph_load', 'fusion', 'export', 'constant_check', 'probe_load',
            'reference_forward', 'adapter_forward')
ZERO_FIELDS = ('llm_calls', 'heldout_samples', 'confirm_samples', 'report_samples',
               'keygen_runs', 'encryptions', 'decryptions', 'real_ckks_attempts',
               'real_ckks_completed', 'downloads', 'installations', 'go_codegen_runs')
FORMAL = ('accuracy_acceptance', 'predicted_latency', 'CKKS_latency', 'security_acceptance')
ORIGINAL_OPS = dict(act=15, avgpool=4, batchnorm=14, conv2d=13, linear=3, mean=1)


def validate_memory_gib(value):
    """Keep the original allowance or use the explicit24GiB virtual-space retry."""
    if type(value) is not int or value not in MEMORY_GIB_CHOICES:
        raise ValueError('Address-space allowance must be12 or24GiB')
    return value


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Object source record required')
    return value


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def source_gate(source):
    from .heir_common_model_pipeline import source_gate as original_gate
    checked = original_gate(MODEL, source)
    original = read(source['intake_result'])
    artifacts = (original.get('worker') or {}).get('artifacts') or {}
    required = dict(graph_dir=Path(artifacts.get('model_info', ''))/'graph',
                    probes=Path(artifacts.get('training_probes', '')),
                    probe_source=Path(artifacts.get('training_probe_source', '')))
    if any(source[key].resolve() != value.resolve() for key, value in required.items()):
        raise ValueError('Graph and saved probes must belong to the accepted original intake')
    contract = checked['contract']
    if (contract.get('graph_nodes') != 50 or contract.get('operator_counts') != ORIGINAL_OPS
            or contract.get('activation_count') != 15
            or contract.get('original_tensor_constant_count') != 60
            or contract.get('input_shape') != [1, 3, 32, 32]
            or contract.get('output_shape') != [1, 10]):
        raise ValueError('Original50-node VGG architecture contract differs')
    for name, path in source.items():
        target = path/'graph.json' if name == 'graph_dir' else path
        if not target.is_file() or target.stat().st_size <= 0:
            raise ValueError('Required original source missing: '+name)
    checked.update(variant=VARIANT, original_model_contract=contract,
                   original_graph_contract_role='original-trained-model',
                   lowered_adapter_contract=None, source_payloads_read=False)
    return checked


def checked_fusion_contract(original, lowered, fusion, exported):
    """Check the complete structural delta without substituting the reference."""
    from .heir_common_models import checked_export_contract
    original, lowered, fusion, exported = map(normalized, (original, lowered, fusion, exported))
    if (original.get('graph_nodes') != 50 or original.get('operator_counts') != ORIGINAL_OPS
            or original.get('activation_count') != 15 or original.get('original_tensor_constant_count') != 60):
        raise ValueError('Original VGG contract differs')
    pairs = fusion.get('pairs') or []
    if (fusion.get('variant') != VARIANT or fusion.get('pair_count') != 4 or len(pairs) != 4
            or fusion.get('original_graph_modified') is not False
            or fusion.get('original_weights_modified') is not False
            or fusion.get('original_model_state_unchanged') is not True
            or fusion.get('derived_constant_count') != 4):
        raise ValueError('Exact fusion provenance or original-state contract differs')
    expected = copy.deepcopy(original)
    nodes = {row['id']: row for row in expected['nodes']}
    removed, replacements, derived_bytes = set(), {}, 0
    for row, (pool_id, conv_id, weight_name) in zip(pairs, fusion_helper.PAIRS):
        pool, conv = nodes[pool_id], nodes[conv_id]
        derived = 'derived-exact-pool-conv:'+pool_id+':'+conv_id
        if (row.get('pool_id') != pool_id or row.get('conv_id') != conv_id
                or row.get('source_weight') != weight_name
                or row.get('source_bias') != conv['attrs'].get('bias')
                or row.get('input') != pool['inputs'][0]
                or row.get('removed_tensor') != pool['outputs'][0]
                or row.get('output') != conv['outputs'][0]
                or row.get('derived_weight') != derived or row.get('bias_added_once') is not True
                or row.get('pool_attrs') != pool['attrs'] or row.get('conv_attrs') != conv['attrs']
                or row.get('input_shape') != pool['input_shapes'][0]
                or row.get('pooled_shape') != pool['output_shapes'][0]
                or row.get('output_shape') != conv['output_shapes'][0]):
            raise ValueError('Four original pool/conv source edges differ')
        constant = row.get('constant') or {}
        source_shape, shape = row.get('source_weight_shape'), row.get('derived_weight_shape')
        if (not isinstance(shape, list) or not isinstance(source_shape, list) or len(shape) != 4
                or source_shape != shape[:2]+[3, 3] or shape[2:] != [6, 6]
                or constant.get('source_shape') != source_shape or constant.get('derived_shape') != shape
                or constant.get('coefficient') != .25 or constant.get('source_weight_modified') is not False
                or constant.get('derived_readonly') is not True
                or constant.get('derived_bytes') != math.prod(shape)*8):
            raise ValueError('Derived6x6 constant source rule differs')
        derived_bytes += constant['derived_bytes']
        removed.add(pool_id);replacements[weight_name] = derived
        conv['inputs'] = list(pool['inputs']);conv['input_shapes'] = copy.deepcopy(pool['input_shapes'])
        conv['attrs'].update(weight=derived, stride=[2, 2], padding=[2, 2], dilation=[1, 1], groups=1)
    expected['nodes'] = [node for node in expected['nodes'] if node['id'] not in removed]
    expected['graph_nodes'] = 46;expected['operator_counts'].pop('avgpool')
    expected['original_tensor_constant_names'] = [replacements.get(name, name)
        for name in original['original_tensor_constant_names']]
    if lowered != expected or fusion.get('new_graph_additional_storage_bytes') != derived_bytes:
        raise ValueError('Only the declared50-to46 structural fusion is admitted')
    acts = [node['id'] for node in original['nodes'] if node['op'] == 'act']
    if (fusion.get('activation_ids_preserved') != acts
            or [row.get('graph_node') for row in exported.get('approximations', [])] != acts
            or exported.get('exact_spatial_mean_matvec') is not True):
        raise ValueError('Original15 activation identities or fixed final mean differ')
    checked_export_contract(exported, lowered)
    return dict(passed=True, original_nodes=50, lowered_nodes=46, activation_ids=acts,
                derived_constant_count=4, emitted_tensor_constants=61,
                original_weights_modified=False, binary64_math_passed=None)


def initial_worker():
    return dict(status='running', error=None, model=MODEL, variant=VARIANT,
        attempted={name: 0 for name in COUNTERS}, completed={name: 0 for name in COUNTERS},
        activation_pairs=0, logit_elements=0, probes=[], constants=[],
        original_model_contract=None, lowered_adapter_contract=None, fusion=None, export=None,
        strict_math_passed=None, quality_selection=False, real_number_equivalence=True,
        **{name: 0 for name in ZERO_FIELDS}, **{name: None for name in FORMAL})


def _runtime():
    # Nix pure does not forward arbitrary numeric-library environment values.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from .heir_cnn_export import load_graph, saved_activations, export_graph
    from .heir_cnn_math_check import check_constants, check_math, evaluate_graph
    from .heir_common_models import graph_contract
    from ..tensor_sim import run_tensor
    return dict(np=np, load_graph=load_graph, saved_activations=saved_activations,
        export_graph=export_graph, check_constants=check_constants, check_math=check_math,
        evaluate_graph=evaluate_graph, graph_contract=graph_contract, run_tensor=run_tensor)


def run_probes(api, graph, fused, program, configs, images, source, result, recorder, save):
    for index, sample_id in enumerate(source['sample_ids']):
        original_values = []
        result['attempted']['reference_forward'] += 1;save()
        with recorder.span('original-program-reference', sample_id=sample_id):
            reference = api['run_tensor'](program, graph.weights, images[index:index+1], cache=False,
                observer=lambda name, value, cfg: original_values.append(value.copy()))
        result['completed']['reference_forward'] += 1
        result['attempted']['adapter_forward'] += 1;save()
        with recorder.span('fused-adapter-forward', sample_id=sample_id):
            observed, activations = api['evaluate_graph'](fused, configs, images[index:index+1], common_model=MODEL)
        result['completed']['adapter_forward'] += 1
        if len(original_values) != 15 or len(activations) != 15:
            raise ValueError('Original15 activation observations required per training input')
        if any(getattr(value, 'dtype', None) != 'float64' for value in [reference, *original_values]):
            raise ValueError('Original program reference and activations must retain float64')
        pairs = [api['check_math'](want, got) for want, got in zip(original_values, activations)]
        check = api['check_math'](reference, observed)
        if reference.shape != (1, 10):
            raise ValueError('Original classifier reference must retain all10 logits')
        result['activation_pairs'] += len(pairs);result['logit_elements'] += int(reference.size)
        result['probes'].append(dict(sample_id=sample_id, reference=reference.tolist(), observed=observed.tolist(),
            activation_checks=pairs, logit_check=check))
        save()
        if not check['passed'] or not all(row['passed'] for row in pairs):
            raise ValueError('Original strict1e-12 mathematical criterion failed')


def worker(args):
    """Pure child: explicit inputs, no registry/config/results-root access."""
    out = args.out.resolve();out.mkdir(parents=True, exist_ok=False)
    result = initial_worker();recorder = StageRecorder(out/'events.jsonl')
    save = lambda: durable_json(out/'result.json', result)
    def stage(name, action):
        result['attempted'][name] += 1;save()
        with recorder.span(name):value = action()
        result['completed'][name] += 1;save()
        return value
    try:
        with recorder.span('fused-export-and-three-training-math'):
            source = {name: getattr(args, name).resolve() for name in SOURCE_FIELDS}
            result['source'] = source_gate(source)
            api = _runtime();np = api['np']
            graph = stage('graph_load', lambda: api['load_graph'](source['graph_dir']))
            original = api['graph_contract'](graph, MODEL)
            if original != result['source']['original_model_contract']:
                raise ValueError('Loaded original graph differs from admitted source metadata')
            fused, fusion = stage('fusion', lambda: fusion_helper.fuse_graph(graph))
            lowered = api['graph_contract'](fused, MODEL)
            result.update(original_model_contract=original, lowered_adapter_contract=lowered, fusion=fusion)
            text_source = source['program'].read_text()
            program, configs = api['saved_activations'](text_source)
            text, exported = stage('export', lambda: api['export_graph'](fused, text_source,
                common_model=MODEL, exact_spatial_mean_matvec=True))
            exported.update(sources={name: str(path) for name, path in source.items()},
                original_model_contract=original, lowered_adapter_contract=lowered, fusion=fusion,
                contract_role='explicit-lowered-adapter-with-separate-original-model-provenance')
            result['contract_check'] = checked_fusion_contract(original, lowered, fusion, exported)
            result['export'] = exported
            (out/'model.mlir').write_text(text)
            durable_json(out/'export.json', exported)
            result['constants'] = stage('constant_check', lambda: api['check_constants'](fused, text,
                common_model=MODEL, exact_spatial_mean_matvec=True))
            from .heir_common_models import expected_constant_names
            if ([row.get('name') for row in result['constants']] != expected_constant_names(lowered, exported)
                    or len(result['constants']) != 61
                    or not all(row.get('passed') is True and row.get('numerically_equal') is True
                               for row in result['constants'])):
                raise ValueError('All61 original/derived emitted tensor constants must retain their numeric values')
            def probes():
                with np.load(source['probes'], allow_pickle=False) as archive:
                    images, indices = archive['input'], archive['indices']
                if (images.dtype != np.float64 or images.shape != (3, 3, 32, 32)
                        or indices.tolist() != [0, 1, 2] or not np.isfinite(images).all()):
                    raise ValueError('Original saved three-probe shape/dtype/finite/indices differ')
                return images
            images = stage('probe_load', probes)
            run_probes(api, graph, fused, program, configs, images, read(source['probe_source']), result, recorder, save)
            result.update(status='ok', strict_math_passed=True)
            result['contract_check']['binary64_math_passed'] = True
            fusion.update(binary64_strict_math_passed=True, real_training_probes_run=3,
                          exporter_or_pipeline_integrated=True)
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__+': '+str(exc)[:4000])
    finally:
        if result.get('export') is not None:
            durable_json(out/'export.json', result['export'])
        result['resource_ru_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['timing'] = summarize_events(read_events(out/'events.jsonl'), time.perf_counter_ns())
        save()
    return result


def check_worker(record, source):
    if (record.get('status') != 'ok' or record.get('model') != MODEL or record.get('variant') != VARIANT
            or record.get('source') != source or record.get('strict_math_passed') is not True
            or record.get('quality_selection') is not False
            or record.get('activation_pairs') != 45 or record.get('logit_elements') != 30
            or any(record.get(name) != 0 for name in ZERO_FIELDS)
            or any(record.get(name) is not None for name in FORMAL)):
        raise ValueError('Completed source-bound export/math worker required')
    for name in COUNTERS:
        expected = 3 if name in ('reference_forward', 'adapter_forward') else 1
        if record.get('attempted', {}).get(name) != expected or record.get('completed', {}).get(name) != expected:
            raise ValueError('Single export/math operation counts differ: '+name)
    checked_fusion_contract(record['original_model_contract'], record['lowered_adapter_contract'],
                            record['fusion'], record['export'])
    from .heir_common_models import expected_constant_names
    if ([row.get('name') for row in record['constants']] != expected_constant_names(record['lowered_adapter_contract'], record['export'])
            or len(record['constants']) != 61
            or not all(row.get('passed') is True and row.get('numerically_equal') is True for row in record['constants'])):
        raise ValueError('Completed emitted-constant verification required')
    rows = record.get('probes') or []
    if [row.get('sample_id') for row in rows] != ['train:0', 'train:1', 'train:2']:
        raise ValueError('Exactly original train:0..2 mathematics required')
    for row in rows:
        if (len(row.get('activation_checks', [])) != 15 or row.get('logit_check', {}).get('passed') is not True
                or not all(check.get('passed') is True for check in row['activation_checks'])):
            raise ValueError('All45 activation and30 logit strict comparisons required')
        want, got = row['reference'], row['observed']
        if not all(isinstance(values, list) and len(values) == 1 and len(values[0]) == 10 for values in (want, got)):
            raise ValueError('Full original logit reference shape differs')
        for reference, observed in zip(want[0], got[0]):
            if (type(reference) not in (int, float) or type(observed) not in (int, float)
                    or not math.isfinite(reference) or not math.isfinite(observed)
                    or abs(observed-reference) > 1e-12+1e-12*abs(reference)):
                raise ValueError('Saved full logits fail the original strict criterion')
    return dict(passed=True, training_samples=3, activation_pairs=45, logit_elements=30,
                emitted_constants=61, reference='original saved program and original trained weights')


def worker_command(source, out, interpreter, memory_gib=12):
    memory_gib = validate_memory_gib(memory_gib)
    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(NIX_SECONDS),
               '--python', str(interpreter), '--', '-m', MODULE, '--worker', '--out', str(out),
               '--memory-gib', str(memory_gib)]
    for name in SOURCE_FIELDS:
        command += ['--'+name.replace('_', '-'), str(source[name])]
    return command


def native_command(tool, source, out, checked):
    from .heir_cnn_preflight import compiler_command
    out.mkdir(parents=True, exist_ok=False);resources = out/'resources';resources.mkdir()
    preserved = out/'native-parameters.json'
    shutil.copyfile(checked['paths']['native_parameters'], preserved)
    # The original gate admits exactly one existing parameter file identity.
    pipeline = checked['pipeline']
    for name in dict.fromkeys(re.findall(r'parameter-file=([^\s}]+)', pipeline)):
        if Path(name).resolve() != Path(checked['paths']['native_parameters']).resolve():
            raise ValueError('Native parameter source identity differs')
        pipeline = pipeline.replace('parameter-file='+name, 'parameter-file='+str(preserved))
    pipeline = pipeline.replace('${RESOURCES}', str(resources))
    if read(preserved) != checked['native_parameters']:
        raise ValueError('Preserved native parameter literal differs')
    (out/'explicit.pipeline').write_text(pipeline+'\n')
    command = compiler_command(tool, source, out/'model-lattigo.mlir', resources)
    command = [value for value in command if not value.startswith(('--annotate-module=', '--torch-linalg-to-ckks=', '--scheme-to-lattigo'))]
    command.insert(2, '--pass-pipeline='+pipeline)
    return command


def check_native(out, source):
    from .heir_cnn_preflight import native_artifact_metadata
    target = out/'model-lattigo.mlir'
    if not target.is_file() or target.stat().st_size <= 0:
        raise ValueError('Native full adapter compiler artifact missing')
    record = native_artifact_metadata(target);actual = record['parameters'];wanted = source['native_parameters']
    if (len(actual) != 1 or any(actual[0].get(key) != wanted[field] for key, field in
            (('logN', 'logN'), ('Q', 'Q'), ('P', 'P'), ('log_default_scale', 'logDefaultScale')))):
        raise ValueError('Every generated native parameter literal must match the original requested parameters')
    durable_json(out/'model-native-artifact.json', record)
    return record


def clean_process(observation):
    if (observation.get('exit_code') != 0 or observation.get('timed_out') is not False
            or observation.get('signal') is not None or observation.get('signals_sent') != []
            or observation.get('remaining_pids') != []):
        raise RuntimeError('Stage process did not exit cleanly; retained process/log metadata describes the failure')


def append_registry(out, result, registry):
    metrics = {key: result[key] for key in ('status', 'attempted', 'completed', 'strict_math_passed',
        'model_compiled', 'static_native_bootstraps', *FORMAL)}
    entry = dict(run_id=out.name, round=ROUND, model=MODEL, mode=MODE, git_rev=None,
                 metrics=metrics, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    with Path(registry).open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise ValueError('A result identity may be registered once')
        stream.seek(0, os.SEEK_END);stream.write(json.dumps(entry, allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def terminal_result(out, result):
    result['completion_metadata_reserved_bytes'] = COMPLETION_RESERVE
    for _ in range(20):
        durable_json(out/'result.json', result)
        size = artifact_bytes(out)
        if size > CASE_BYTES-COMPLETION_RESERVE:
            result.update(status='resource_stopped', error='Case artifact allowance including completion reserve exhausted',
                guard_trigger=dict(kind='artifact', actual_bytes=size, allowance_bytes=CASE_BYTES,
                                   completion_reserve_bytes=COMPLETION_RESERVE))
        if result.get('artifact_bytes') == size:
            durable_json(out/'result.json', result)
            if artifact_bytes(out) == size:return
        result['artifact_bytes'] = size
    raise RuntimeError('Terminal result size did not converge')


def _snapshot(out):
    from . import heir_common_model_pipeline, heir_common_models, heir_cnn_export, heir_cnn_math_check, heir_cnn_preflight
    modules = [sys.modules[__name__], fusion_helper, heir_common_model_pipeline,
               heir_common_models, heir_cnn_export, heir_cnn_math_check, heir_cnn_preflight]
    directory = out/'source';directory.mkdir()
    records = []
    for module in modules:
        source = Path(module.__file__).resolve();target = directory/source.name
        shutil.copyfile(source, target)
        records.append(dict(original=str(source), snapshot=str(target), bytes=target.stat().st_size))
    return records


def execute(args):
    from .. import paths
    from .heir_common_model_pipeline import sources
    memory_gib = validate_memory_gib(getattr(args, 'memory_gib', 12))
    root = paths.RESULTS_DIR.resolve();out = args.out.resolve()
    if out.exists() or out == root or not out.is_relative_to(root):
        raise ValueError('A new isolated case under the results root is required')
    with serial_lock(root):
        out.mkdir(parents=True);(out/'tmp').mkdir();(out/'phases').mkdir()
        recorder = StageRecorder(out/'events.jsonl');started = time.perf_counter_ns()
        deadline = time.monotonic()+TOTAL_SECONDS
        result = dict(round=ROUND, model=MODEL, mode=MODE, variant=VARIANT, status='running', error=None,
            attempted=dict(export_math=0, native_compile=0), completed=dict(export_math=0, native_compile=0),
            phases={}, strict_math_passed=None, model_compiled=False, static_native_bootstraps=None,
            original_model_changed=False, quality_selection=False, incoming_signal=None, timeout_signal=None,
            wall_timeout_observed=False, guard_trigger=None, guard_stop_signal=None,
            limits=dict(total_seconds=TOTAL_SECONDS, worker_seconds=WORKER_SECONDS, nix_seconds=NIX_SECONDS,
                worker_outer_seconds=WORKER_OUTER_SECONDS, native_seconds=NATIVE_SECONDS,
                AS_gib=memory_gib, subtree_RSS_gib=14, minimum_available_gib=1.25, case_gib=32,
                file_gib=16, minimum_free_disk_gib=10, concurrency=1),
            **{name: 0 for name in ZERO_FIELDS}, **{name: None for name in FORMAL})
        active = [None];failures = [];stop = threading.Event()
        old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        def check():
            if failures:raise RuntimeError(failures[0])
            if time.monotonic() >= deadline:
                result['wall_timeout_observed'] = True;raise TimeoutError('9600s full-case allowance exhausted')
            observed = physical_observation(_latest_rss_sample(active[0]) if active[0] else None,
                _available_ram_bytes(), shutil.disk_usage(out).free)
            if observed['stop']:
                result['guard_trigger'] = observed
                durable_json(out/'resource-guard.json', observed)
                raise RuntimeError('; '.join(observed['reasons']))
            size = artifact_bytes(out)
            if size > CASE_BYTES-COMPLETION_RESERVE:
                result['guard_trigger'] = dict(kind='artifact', actual_bytes=size, allowance_bytes=CASE_BYTES)
                raise RuntimeError('32GiB case allowance exhausted')
        def monitor():
            while not stop.wait(.2):
                try:check()
                except BaseException as exc:
                    failures.append(type(exc).__name__+': '+str(exc))
                    result['monitor_stop_reason'] = failures[-1]
                    try:durable_json(out/'monitor-stop.json', dict(reason=failures[-1], guard=result.get('guard_trigger')))
                    finally:
                        try:os.kill(os.getpid(), signal.SIGTERM)
                        except OSError as error:result['monitor_signal_error'] = str(error)
                    return
        watcher = threading.Thread(target=monitor, daemon=True)
        for sig in old:signal.signal(sig, lambda signum, frame: receive_signal(result, signum))
        signal.setitimer(signal.ITIMER_REAL, TOTAL_SECONDS)
        try:
            with recorder.span('complete-fused-adapter-job'):
                check();watcher.start()
                with recorder.span('source-admission'):
                    source = sources(MODEL)
                    source.update({name: getattr(args, name).resolve() for name in SOURCE_FIELDS if getattr(args, name, None) is not None})
                    checked = source_gate(source);result['source'] = checked
                    if not args.heir_opt.is_file() or not os.access(args.heir_opt, os.X_OK):
                        raise ValueError('Existing executable HEIR compiler required')
                    result['source_snapshots'] = _snapshot(out)
                    command = worker_command(source, out/'worker', args.python, memory_gib)
                    result['planned_worker_command'] = command
                    durable_json(out/'source.json', checked)
                result['status'] = 'prepared';durable_json(out/'result.json', result)
                if args.run:
                    env = {key: value for key, value in os.environ.items() if key not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')
                           and not key.startswith(('LD_', 'NIX_', '__NIX_'))}
                    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TMPDIR=str(out/'tmp'))
                    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):env[name] = '1'
                    def launch(stage, command, seconds):
                        check();result['attempted'][stage] = 1
                        directory = out/'phases'/stage;directory.mkdir()
                        row = dict(argv=command, status='running', observation=None)
                        result['phases'][stage] = row;durable_json(out/'result.json', result)
                        with recorder.span(stage, argv=command):
                            active[0] = directory/'resources.jsonl'
                            try:
                                observation = run_process(command, cwd=paths.REPO/'DSL/DSL_agent/scripts',
                                    log=directory/'worker.log', resources=active[0],
                                    timeout=min(seconds, max(.01, deadline-time.monotonic())), mem_gib=memory_gib, fsize_gib=16, env=env)
                                row['observation'] = observation;observe_termination(result, observation)
                            finally:active[0] = None
                            clean_process(observation);check()
                        return row
                    row = launch('export_math', command, WORKER_OUTER_SECONDS)
                    result['worker'] = read(out/'worker/result.json')
                    row['audit'] = check_worker(result['worker'], checked)
                    row['status'] = 'ok';result['completed']['export_math'] = 1;result['strict_math_passed'] = True
                    durable_json(out/'result.json', result)
                    command = native_command(args.heir_opt, out/'worker/model.mlir', out/'native', checked)
                    result['native_request'] = dict(pipeline=str(out/'native/explicit.pipeline'),
                        parameters=checked['native_parameters'], source_mlir=str(out/'worker/model.mlir'),
                        original_model_contract=checked['contract'], lowered_adapter_contract=result['worker']['lowered_adapter_contract'])
                    row = launch('native_compile', command, NATIVE_SECONDS)
                    artifact = check_native(out/'native', checked)
                    row.update(status='ok', artifact=artifact)
                    result['completed']['native_compile'] = 1
                    result.update(status='ok', model_compiled=True, static_native_bootstraps=artifact['static_bootstrap_operations'])
        except BaseException as exc:
            result.update(status=interruption_status(result), error=type(exc).__name__+': '+str(exc)[:4000])
            if (out/'worker/result.json').is_file():result['worker'] = read(out/'worker/result.json')
            for row in result['phases'].values():
                if row['status'] == 'running':row.update(status=result['status'], error=result['error'])
        finally:
            stop.set()
            if watcher.is_alive():watcher.join(timeout=3)
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in old.items():signal.signal(sig, handler)
            if failures:result.update(status=interruption_status(result), error=failures[0])
            result['timing'] = summarize_events(read_events(out/'events.jsonl'), time.perf_counter_ns())
            terminal_result(out, result)
            append_registry(out, result, args.registry or root/'registry.jsonl')
            marker = persist_case_completion(out, round_label=ROUND, started_ns=started, status=result['status'],
                timing=result['timing'], required_artifact_bytes=artifact_bytes(out))
            if (out/'completion.json').stat().st_size > COMPLETION_RESERVE or marker['case_artifact_bytes'] > CASE_BYTES:
                raise RuntimeError('Completion metadata exceeded its explicit reserved case allowance')
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--run', action='store_true')
    for name in SOURCE_FIELDS:parser.add_argument('--'+name.replace('_', '-'), type=Path)
    parser.add_argument('--python', type=Path)
    parser.add_argument('--heir-opt', type=Path)
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--memory-gib', type=int, choices=MEMORY_GIB_CHOICES, default=12)
    args = parser.parse_args(argv)
    if args.worker:
        if any(getattr(args, name) is None for name in SOURCE_FIELDS) or args.registry or args.run:
            parser.error('Pure worker requires every explicit source path and has no registry/run options')
        # These limits also bound a direct worker launch, without a host wrapper.
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_gib*GIB, args.memory_gib*GIB))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16*GIB, 16*GIB))
        signal.alarm(WORKER_SECONDS)
        try:result = worker(args)
        finally:signal.alarm(0)
    else:
        from .. import paths
        args.python = args.python or paths.INTAKE_PYTHON
        args.heir_opt = args.heir_opt or paths.WORKSPACE/'baselines/heir-2026.11.1-venv/bin/heir-opt'
        result = execute(args)
    print(json.dumps({name: result.get(name) for name in ('status', 'error', 'strict_math_passed', 'model_compiled')}))
    return 0 if result['status'] in ('ok', 'prepared') else 1


if __name__ == '__main__':raise SystemExit(main())
