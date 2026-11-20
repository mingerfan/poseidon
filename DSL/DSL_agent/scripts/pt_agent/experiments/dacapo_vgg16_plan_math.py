"""Three saved-training-probe mathematics for official DaCapo VGG arithmetic.

The official FHE source omits convolution bias in HE_ConvBN. A private copy of
the original factory model follows that source and its physical32 Poly units.
Original Torch outputs remain separate diagnostics, with no task acceptance.
The unchanged RuntimePlan is replayed through the existing lazy ideal oracle;
no key, ciphertext, compiler, API, archive restore or dataset reader is invoked.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import threading
import time
from types import FunctionType

from .. import paths
from . import dacapo_native_binding as binding
from . import dacapo_native_plan_math as replay
from . import dacapo_compile_continuation as supervision
from . import r24_orbit_polynomial_worker as polynomial
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

ROUND = 'DACAPO-VGG16-OFFICIAL-ARITHMETIC-MATH'
MODEL = 'dacapo-original-vgg16-silu-avgpool'
FACTORY = 'pt_agent.experiments.r13_common_models:vgg16_silu_avgpool'
MODE = 'official-FHE-arithmetic-physical32-three-saved-training-probes'
CONVS = tuple('conv_' + str(stage) + '_' + str(index)
    for stage, count in ((1, 2), (2, 2), (3, 3), (4, 3), (5, 3)) for index in range(1, count + 1))
EXPECTED = dict(execution_count=42860, value_count=52860, initialization_count=9999,
    inline_scalar_count=179, bundle_encode_count=9820, unique_blob_count=9716, bootstrap_count=20)
GIB = 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def _function(tree, name):
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise ValueError('One original source function required: ' + name)
    return ast.unparse(found[0])


def source_facts(upstream):
    """Verify the source distinctions before opening any model/probe payload."""
    upstream = Path(upstream)
    funcs = ast.parse((upstream / 'python/poly/poly/Func.py').read_text())
    conv = _function(funcs, 'HE_ConvBN')
    if 'conv.weight' not in conv or 'MPCB.abstractBN(bn)' not in conv or 'conv.bias' in conv:
        raise ValueError('Official HE_ConvBN convolution-bias omission changed')
    silu = _function(funcs, 'HE_SiLU')
    if 'calculation = Poly.GenPoly()' not in silu or 'x * (calculation(x) + 0.5)' not in silu:
        raise ValueError('Original source Poly SiLU expression changed')
    benchmark = ast.parse((upstream / 'examples/benchmarks/VGG16.py').read_text())
    body = next(n for n in benchmark.body if isinstance(n, ast.FunctionDef) and n.name == 'VGG16')
    linear = [n for n in ast.walk(body) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'HE_Linear']
    mpbn = [n for n in ast.walk(body) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'HE_MPBN']
    if len(linear) != 3 or len(mpbn) != 1 or any(
            not any(k.arg == 'scale' and isinstance(k.value, ast.Constant) and k.value.value == 32
                    for k in call.keywords) for call in linear + mpbn):
        raise ValueError('Official VGG affine physical32 source changed')
    runtime = upstream / 'lib/Runtime/HEAAN_HEVM.cpp'
    if runtime.read_text().count('src[i % src.size()]') != 2:
        raise ValueError('Original HEAAN online/preencoded periodic plaintext expansion changed')
    return dict(convolution_bias_used_by_official_FHE=False, convolution_bias_layers=13,
        original_Poly_preserved=True, physical_activation='32 * original_HE_SiLU(z / 32)',
        affine_scale=32, scalar_periodic_runtime_source=str(runtime), scalar_periodic_runtime_line=289,
        input_packing='C-order CHW/32; pad3072 to4096; repeat16 to65536',
        output_decoding='first10 slots times32, shape[1,10]',
        packing_sources=[str(upstream / 'examples/tests/VGG16.py'), str(upstream / 'python/poly/poly/MPCB.py')],
        pooling_geometry='Five2x2 pools:32->16->8->4->2->1; final channel order occupies first512 slots',
        task_accuracy_acceptance=None)


def source_gate(args, *, require_restored=True):
    case = Path(args.compile_case).resolve()
    compiled, closed = read(case / 'result.json'), read(case / 'completion.json')
    source = compiled.get('source') or {}
    original_case = Path(source['original_case'])
    original = read(original_case / 'result.json')
    if (compiled.get('round') != 'DACAPO-ORIGINAL-COMPILE-CONTINUATION' or compiled.get('model') != MODEL
            or compiled.get('status') != 'ok' or closed.get('status') != 'ok'
            or closed.get('required_artifacts_persisted') is not True or closed.get('registry_persisted') is not True
            or any(compiled.get(k) != 1 for k in ('compile_attempts', 'compile_completed', 'native_compile_attempts', 'native_compile_completed'))
            or compiled.get('trace_attempts') != 0 or compiled.get('original_trace_reused') is not True
            or compiled.get('bootstrap_count') != 20 or compiled.get('execution_instructions') != 42860
            or compiled.get('export_contract_compatible') is not True
            or not replay._clean((compiled.get('compile_worker') or {}).get('process') or {})
            or not replay._clean((compiled.get('processes') or [{}])[0].get('observation') or {})
            or original.get('trace_completed') != 1 or not replay._clean(original.get('trace_process') or {})):
        raise ValueError('A closed successful official VGG continuation with its clean original trace is required')
    r13_path = Path(source['original_sources']['saved_r13_result']['path'])
    r13 = read(r13_path)
    worker = r13.get('worker') or {}
    model = worker.get('model_source') or {}
    official = source['original_sources']['original_model']
    if (r13.get('round') != 'R13' or r13.get('status') != 'ok' or r13.get('model') != FACTORY
            or model.get('factory') != FACTORY or official.get('factory') != FACTORY
            or not replay._same(model.get('checkpoint'), official.get('checkpoint'))
            or not replay._same(model.get('original_source'), official.get('original_model_source'))
            or worker.get('real_probes', {}).get('passed') is not True):
        raise ValueError('Same original VGG factory/checkpoint and saved three-probe intake are required')
    probe_source = Path(worker['artifacts']['training_probe_source'])
    probes = Path(worker['artifacts']['training_probes'])
    saved = read(probe_source)
    if saved != worker.get('source') or saved.get('split') != 'train' or saved.get('indices') != [0, 1, 2] or saved.get('sample_ids') != ['train:0', 'train:1', 'train:2']:
        raise ValueError('Only the original saved three training inputs may be read')
    upstream = Path(source['original_sources']['script']['path']).parents[2]
    facts = source_facts(upstream)
    plan = Path(compiled['plan']['path'])
    bundle = plan.with_name(plan.name.removesuffix('.runtime-plan.json') + '.bundle')
    archive = read(Path(args.archive_case) / 'manifest.json')
    archived = read(Path(args.archive_case) / 'result.json')
    selected = [v for v in archive.get('files', []) if v.get('kind') == 'runtime_bundle']
    if (archived.get('status') != 'ok' or len(selected) != 1
            or not replay._same(archive.get('case_result'), case / 'result.json')
            or not replay._same(selected[0].get('path'), bundle)):
        raise ValueError('Closed same-plan archive inventory is required')
    if require_restored and not (bundle / 'manifest.json').is_file():
        raise FileNotFoundError('Restore only the original runtime bundle before the scheduled math worker')
    return dict(compile_case=str(case), original_case=str(original_case), r13_result=str(r13_path),
        plan=str(plan), bundle=str(bundle), archive_manifest=str(Path(args.archive_case) / 'manifest.json'),
        probes=str(probes), probe_source=str(probe_source), upstream=str(upstream), poly_dir=str(upstream / 'python/poly/poly'),
        original_checkpoint=model['checkpoint'], original_definition=model['original_source'],
        source_factory=FACTORY, facts=facts, restored_bundle_available=(bundle / 'manifest.json').is_file(),
        source_payloads_read=0, sample_ids=saved['sample_ids']), r13


def load_program(source):
    plan, manifest = read(source['plan']), read(Path(source['bundle']) / 'manifest.json')
    proof = dict(status='scalar_sites_verified', source_factory=FACTORY, source_plan=source['plan'],
        original_dsl_scalar_broadcast_interpretation=True, original_plan_modified=False,
        sites=[dict(value_id=o['output'], interpretation='full-slot-scalar',
            source_path=source['facts']['scalar_periodic_runtime_source'], source_line=289)
            for o in plan['initialization'] if o['payload']['kind'] == 'inline'])
    # Private namespaces preserve the old ResNet source gate and its meaning.
    namespace = dict(binding.plan_inventory.__globals__, SOURCE_FACTORY=FACTORY)
    namespace['_source_gates'] = FunctionType(binding._source_gates.__code__, namespace,
        binding._source_gates.__name__, binding._source_gates.__defaults__, binding._source_gates.__closure__)
    inventory = FunctionType(binding.plan_inventory.__code__, namespace, binding.plan_inventory.__name__,
        binding.plan_inventory.__defaults__, binding.plan_inventory.__closure__)
    inventory.__kwdefaults__ = binding.plan_inventory.__kwdefaults__
    metadata = inventory(plan, manifest, source_proof=proof, plan_path=source['plan'])
    if metadata['source_gates']['per_site_provenance_verified'] is not True:
        raise ValueError('All original singleton plaintext sites require periodic encoding evidence')
    values = {v['id']:v for v in plan['values']}
    last = {name:i for i, op in enumerate(plan['execution']) for name in op['inputs'] if values[name]['kind'] == 'ciphertext'}
    return dict(plan=plan, manifest=manifest, values=values, encodes={o['output']:o['payload'] for o in plan['initialization']},
        last_use=last, input=plan['external_inputs'][0], output=plan['final_outputs'][0], metadata=metadata), proof


def audit_and_mask_bias(original, private, to_numpy, *, conv_names=CONVS):
    import numpy as np
    rows = []
    for name in conv_names:
        a, b = original.get_submodule(name).Conv2d, private.get_submodule(name).Conv2d
        if a.bias is None or b.bias is None:
            raise ValueError('Original VGG convolution bias must be inventoried before private masking: ' + name)
        reference = a.bias
        values, copied = to_numpy(a.bias), to_numpy(b.bias)
        if (values.ndim != 1 or values.shape != (a.out_channels,) or not np.isfinite(values).all()
                or not np.array_equal(values, copied)):
            raise ValueError('Original/private convolution bias semantic coverage differs: ' + name)
        rows.append(dict(module=name + '.Conv2d', shape=list(values.shape), count=int(values.size),
            dtype=str(values.dtype), finite=True, nonzero_count=int(np.count_nonzero(values)),
            maximum_absolute=float(np.max(np.abs(values))), L2_norm=float(np.linalg.norm(values)),
            used_by_official_FHE=False, private_reference_bias_removed=True, original_parameter_retained=True))
        b.bias = None
        if a.bias is not reference:
            raise ValueError('Original model bias changed during private adaptation')
    return rows


def physical_activation(values, native_silu, coefficients):
    import numpy as np
    if values.dtype != np.dtype('float64') or not np.isfinite(values).all():
        raise ValueError('Physical reference activation requires finite float64 inputs')
    u = values / 32.0
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        native = native_silu(u)
        checked = polynomial._check_math(polynomial.odd_chebyshev_reference(coefficients, u), native)
        if not checked['passed']:
            raise ValueError('Original Poly tree and independent odd Chebyshev arithmetic differ')
        physical = 32.0 * native
    if not np.isfinite(physical).all():
        raise ValueError('Physical reference activation became nonfinite')
    return physical, checked


def prepare_reference(original, native_silu, coefficients, result):
    import torch
    private = copy.deepcopy(original)
    result['private_model_copies'] = 1
    result['unused_convolution_bias_audit'] = audit_and_mask_bias(original, private, lambda t:t.detach().cpu().numpy())
    rows = result['unused_convolution_bias_audit']
    result['unused_convolution_bias_summary'] = dict(layers=len(rows), parameter_count=sum(r['count'] for r in rows),
        nonzero_count=sum(r['nonzero_count'] for r in rows), maximum_absolute=max(r['maximum_absolute'] for r in rows),
        all_finite=all(r['finite'] for r in rows), arrays_output=False, used_by_official_FHE=False)
    selected = [(name, m) for name, m in private.named_modules() if isinstance(m, torch.nn.SiLU)]
    if len(selected) != 14:
        raise ValueError('Original VGG requires13 convolution SiLU modules and one reused head SiLU module')
    class PhysicalSiLU(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.source_name = name
        def forward(self, z):
            result['activation_attempts'] += 1
            output, checked = physical_activation(z.detach().cpu().numpy(), native_silu, coefficients)
            result['activation_completed'] += 1
            result['activation_math'].append(dict(module=self.source_name, sample_id=result.get('active_sample_id'), **checked))
            return torch.from_numpy(output)
    for name, _ in selected:
        parent, _, attribute = name.rpartition('.')
        setattr(private.get_submodule(parent) if parent else private, attribute, PhysicalSiLU(name))
    result['physical_SiLU_modules'] = len(selected)
    return private.eval()


def accepted_worker(measured):
    allowed_counts = dict(model_forwards=6, torch_model_builds=1, torch_model_forwards=6, checkpoint_reads=1)
    if (measured.get('status') != 'ok' or measured.get('round') != ROUND
            or measured.get('model') != MODEL or measured.get('source_factory') != FACTORY
            or measured.get('official_FHE_arithmetic_math_passed') is not True
            or type(measured.get('elapsed_ns')) is not int or measured['elapsed_ns'] <= 0
            or measured.get('original_model_modified') is not False
            or measured.get('task_accuracy_evaluated') is not False
            or measured.get('task_accuracy_acceptance') is not None
            or any(measured.get(k) is not None for k in replay.FORMAL_FIELDS)
            or any(measured.get(k) != allowed_counts.get(k, 0) for k in replay.ZERO_FIELDS)
            or measured.get('packed_forward_completed') != 3 or measured.get('math_check_completed') != 3
            or measured.get('activation_completed') != 45 or measured.get('op_completed') != 128580
            or measured.get('checkpoint_load_completed') != 1 or measured.get('physical_SiLU_modules') != 14
            or len(measured.get('unused_convolution_bias_audit', [])) != 13
            or len(measured.get('activation_math', [])) != 45
            or any(v.get('passed') is not True for v in measured.get('activation_math', []))
            or [v.get('sample_id') for v in measured.get('probe_rows', [])] != ['train:0', 'train:1', 'train:2']
            or any(v.get('math', {}).get('passed') is not True for v in measured.get('probe_rows', []))):
        raise ValueError('Official VGG three-probe terminal protocol is incomplete')
    return measured


def worker(args):
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=False)
    recorder = StageRecorder(work / 'events.jsonl')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    memory = int(args.memory_gib * GIB)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        memory = min(memory, hard)
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    result = replay.initial_result()
    result.update(round=ROUND, model=MODEL, source_factory=FACTORY, mode=MODE, status='running',
        task_accuracy_acceptance=None, official_FHE_arithmetic_math_passed=False,
        original_Torch_diagnostics=[], unused_convolution_bias_audit=[], activation_math=[],
        activation_attempts=0, activation_completed=0, checkpoint_load_attempts=0,
        checkpoint_load_completed=0, original_model_builds_completed=0, private_model_copies=0,
        original_Torch_forward_attempts=0, original_Torch_forward_completed=0,
        official_reference_forward_attempts=0, official_reference_forward_completed=0,
        original_model_modified=False, task_accuracy_evaluated=False,
        counter_scope='A running checkpoint contains observed lifecycle counters; completed-forward aggregates are finalized in the terminal worker record. Missing terminal records leave actual totals unobserved.',
        scope='Official FHE arithmetic including source-omitted convolution bias, original Poly and physical32; original Torch outputs are separate training-probe diagnostics')
    started = time.perf_counter_ns()
    def checkpoint():
        if (time.perf_counter_ns() - started) / 1e9 > args.seconds:
            raise TimeoutError('Official VGG math worker wall allowance exhausted')
        if artifact_bytes(work.parent) > args.case_gib * GIB or shutil.disk_usage(work).free < args.minimum_free_gib * GIB:
            raise RuntimeError('Official VGG math resource allowance exhausted')
        write_json(work / 'worker-result.json', result)
    constants = None
    try:
        with recorder.span('source-and-restored-metadata'):
            source, r13 = source_gate(args)
            result['source'] = source
            result['source_checked'] = True
            program, proof = load_program(source)
            for name, value in EXPECTED.items():
                if program['metadata'][name] != value:
                    raise ValueError('Original VGG complete plan inventory differs: ' + name)
            result['program_metadata'] = program['metadata']
            result['restored_bundle_metadata'] = replay.check_restored_bundle(program, source['bundle'], source['archive_manifest'])
            write_json(work / 'scalar-source-proof.json', proof)
            checkpoint()
        from .r13_common_models import vgg16_silu_avgpool
        import torch
        import numpy as np
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        with recorder.span('original-factory-and-private-official-reference'):
            result['checkpoint_load_attempts'] = 1
            result['checkpoint_reads'] = None
            result['torch_model_builds'] = None
            checkpoint()
            original_model = vgg16_silu_avgpool().cpu().double().eval()
            result.update(checkpoint_load_completed=1, checkpoint_reads=1, original_model_builds_completed=1, torch_model_builds=1)
            native_silu, coefficients, native_metadata = polynomial.load_native_functions(source['poly_dir'])
            result['original_Poly_metadata'] = native_metadata
            physical_model = prepare_reference(original_model, native_silu, coefficients, result)
            checkpoint()
        with recorder.span('saved-three-training-probes'):
            probe_args = argparse.Namespace(probes=Path(source['probes']), probe_source=Path(source['probe_source']))
            images, labels, probe_source = replay.reference._load_probes(probe_args, r13, result)
            result['probe_source'] = probe_source
        constants = replay.LazyPlaintexts(source['bundle'], program)
        for index in range(3):
            result['active_sample_id'] = 'train:' + str(index)
            checkpoint()
            with recorder.span('official-reference-and-original-Torch-diagnostic', sample_id=result['active_sample_id']):
                inputs = torch.from_numpy(images[index:index + 1])
                with torch.no_grad():
                    result['original_Torch_forward_attempts'] += 1
                    exact = original_model(inputs).detach().cpu().numpy()
                    result['original_Torch_forward_completed'] += 1
                    result['official_reference_forward_attempts'] += 1
                    expected = physical_model(inputs).detach().cpu().numpy()
                    result['official_reference_forward_completed'] += 1
                if exact.shape != (1, 10) or expected.shape != (1, 10) or exact.dtype != np.float64 or expected.dtype != np.float64 or not np.isfinite(exact).all() or not np.isfinite(expected).all():
                    raise ValueError('Both original and official-reference logits must be finite float64 [1,10]')
                result['original_Torch_diagnostics'].append(dict(sample_id=result['active_sample_id'],
                    original_Torch=exact.tolist(), official_FHE_arithmetic_reference=expected.tolist(),
                    maximum_absolute_difference=float(np.max(np.abs(exact - expected))),
                    used_for_task_acceptance=False, source_difference='Official Poly/physical32 and source-omitted conv.bias'))
            with recorder.span('unchanged-complete-ideal-plan', sample_id=result['active_sample_id']):
                result['packed_forward_attempts'] += 1
                packed = replay.reference.pack_input(images[index])
                output = replay.execute_program(program, constants, packed, result, progress=checkpoint)
                decoded = replay.reference.decode_output(output)
                result['packed_forward_completed'] += 1
                result['math_check_attempts'] += 1
                checked = replay.reference._check_math(expected, decoded)
                result['math_check_completed'] += 1
                result['probe_rows'].append(dict(sample_id=result['active_sample_id'], label=int(labels[index]),
                    decoded=decoded.tolist(), official_FHE_arithmetic_reference=expected.tolist(), math=checked))
                checkpoint()
                if not checked['passed']:
                    raise ValueError('Complete official VGG plan fails the unchanged strict mathematical criterion')
        if result['activation_completed'] != 45 or result['op_completed'] != 3 * EXPECTED['execution_count'] or any(
                result['op_completed_by_kind'][k] != 3 * v for k, v in program['metadata']['op_counts'].items()):
            raise ValueError('Every official operation and15 activations must complete on all three saved inputs')
        result.update(status='ok', official_FHE_arithmetic_math_passed=True, original_training_math_verified=True,
            full_execution_scope_ready=False, model_forwards=6, torch_model_forwards=6)
        write_json(work / 'source-math-proof.json', dict(status='strict_math_passed', source_factory=FACTORY,
            source_plan=source['plan'], sample_ids=['train:0', 'train:1', 'train:2'],
            original_poly_preserved=True, original_pack_bb32_preserved=True, original_plan_modified=False,
            absolute_tolerance=1e-12, relative_tolerance=1e-12, all_three_passed=True,
            reference_scope='Official FHE arithmetic; source-omitted conv.bias audited separately',
            task_accuracy_acceptance=None, crypto_executions=0))
    except BaseException as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc)[:4000])
    finally:
        if constants is not None:
            result['constant_reads'] = constants.summary()
            constants.clear()
        result['model_forwards'] = result['original_Torch_forward_completed'] + result['official_reference_forward_completed']
        result['torch_model_forwards'] = result['model_forwards']
        result['elapsed_ns'] = time.perf_counter_ns() - started
        result['worker_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(work / 'worker-result.json', result)
    return result


def execute(args):
    out = Path(args.out).resolve()
    if out.parent != paths.RESULTS_DIR.resolve() or out.exists():
        raise ValueError('A new separate official VGG math case is required')
    args.out = out
    out.mkdir()
    started = time.perf_counter_ns()
    deadline = time.monotonic() + args.seconds + 240
    result = dict(round=ROUND, model=MODEL, mode=MODE, status='running', worker=None, processes=[],
        worker_attempts=0, worker_completed=0, incoming_signal=None, timeout_signal=None,
        guard_trigger=None, monitor_error=None, wall_timeout_observed=False,
        accuracy_acceptance=None, security_acceptance=None, actual_ckks_latency_seconds=None,
        archive_restores=0, api_calls=0, compiler_runs=0, keygen_runs=0, ckks_executions=0,
        limits=dict(worker_AS_gib=args.memory_gib, file_gib=1, case_gib=args.case_gib,
            minimum_free_disk_gib=args.minimum_free_gib, subtree_plus_supervisor_RSS_gib=14,
            minimum_available_RAM_gib=1.25, worker_seconds=args.seconds,
            wrapper_seconds=args.seconds + 120, total_seconds=args.seconds + 240, concurrency=1))
    recorder = StageRecorder(out / 'events.jsonl')
    active = [None]
    stop = threading.Event()
    old = {s:signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    def interrupt(signum, frame):
        supervision.receive_signal(result, signum)
    def monitor():
        while not stop.wait(.2):
            try:
                supervision.check_resources(args, result, active[0], deadline)
            except BaseException as exc:
                if result['guard_trigger'] is None and not result['wall_timeout_observed']:
                    result['monitor_error'] = type(exc).__name__ + ': ' + str(exc)
                try:
                    write_json(out / 'monitor-stop.json', dict(error=str(exc), guard_trigger=result['guard_trigger']))
                finally:
                    os.kill(os.getpid(), signal.SIGTERM)
                return
    watcher = threading.Thread(target=monitor, daemon=True)
    for sig in old:
        signal.signal(sig, interrupt)
    try:
        with supervision.serial_lease(paths.RESULTS_DIR):
            with recorder.span('original-source-and-restored-bundle-admission'):
                source, _ = source_gate(args)
                result['source'] = source
                shutil.copy2(__file__, out / 'driver.py')
                snapshots = out / 'source'
                snapshots.mkdir()
                for module in (binding, replay, polynomial, replay.reference):
                    shutil.copy2(module.__file__, snapshots / Path(module.__file__).name)
                write_json(out / 'source.json', source)
                supervision.check_resources(args, result, None, deadline)
            watcher.start()
            inner = ['-m', __spec__.name, '--worker', '--work-dir', str(out / 'worker'),
                '--compile-case', str(args.compile_case), '--archive-case', str(args.archive_case),
                '--seconds', str(args.seconds), '--memory-gib', str(args.memory_gib), '--case-gib', str(args.case_gib),
                '--minimum-free-gib', str(args.minimum_free_gib)]
            argv = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(args.seconds + 60),
                '--python', str(args.intake_python), '--', *inner]
            row = dict(argv=argv, observation=None, supervision_exception=None)
            result['processes'].append(row)
            result['worker_attempts'] = 1
            supervision.durable(out / 'result.json', result)
            with recorder.span('three-training-probe-official-arithmetic-worker'):
                active[0] = out / 'resources.jsonl'
                try:
                    observation = run_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts',
                        log=out / 'worker.log', resources=active[0], timeout=args.seconds + 120,
                        mem_gib=args.memory_gib, fsize_gib=1,
                        env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
                    row['observation'] = observation
                except BaseException as exc:
                    row['supervision_exception'] = type(exc).__name__ + ': ' + str(exc)
                    row['last_resource_sample'] = supervision.latest_resource(active[0])
                    raise
                finally:
                    active[0] = None
                result['wall_timeout_observed'] |= bool(observation.get('timed_out'))
                if observation.get('signal') is not None:
                    field = 'guard_stop_signal' if result['guard_trigger'] is not None or result['monitor_error'] else 'incoming_signal'
                    result[field] = observation['signal']
                result['worker'] = read(out / 'worker/worker-result.json') if (out / 'worker/worker-result.json').is_file() else None
                measured = result['worker'] or {}
                if not replay._clean(observation):
                    raise RuntimeError('Official VGG worker failed; partial evidence retained without retry')
                accepted_worker(measured)
                result.update(status='ok', worker_completed=1)
    except BaseException as exc:
        result.update(status=supervision.status(result), error=type(exc).__name__ + ': ' + str(exc)[:4000])
    finally:
        stop.set()
        if watcher.is_alive():
            watcher.join(timeout=3)
        for sig in old:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if result['worker'] is None and (out / 'worker/worker-result.json').is_file():
                result['worker'] = read(out / 'worker/worker-result.json')
            observed_worker = result['worker'] or {}
            result['worker_terminal_counters_available'] = (observed_worker.get('status') in ('ok', 'error')
                and type(observed_worker.get('elapsed_ns')) is int)
            result['worker_counter_scope'] = ('Terminal worker lifecycle counts' if result['worker_terminal_counters_available']
                else 'No terminal worker record; saved checkpoint counts are observed lower bounds and actual totals remain unobserved')
            if result['guard_trigger'] is not None or result['monitor_error'] or result['incoming_signal'] is not None:
                result['status'] = supervision.status(result)
            timing = summarize_events(read_events(out / 'events.jsonl'), time.perf_counter_ns())
            result['timing'] = timing
            result['recorded_elapsed_seconds'] = (time.perf_counter_ns() - started) / 1e9
            supervision.durable(out / 'result.json', result)
            metrics = {k:result.get(k) for k in ('status', 'worker_attempts', 'worker_completed', 'accuracy_acceptance', 'security_acceptance', 'actual_ckks_latency_seconds')}
            registry_row = dict(run_id=out.name, round=ROUND, model=MODEL, mode=MODE, git_rev=None,
                metrics=metrics, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
            Path(args.registry).parent.mkdir(parents=True, exist_ok=True)
            with Path(args.registry).open('a+') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                stream.seek(0)
                if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
                    raise RuntimeError('Run id already exists')
                stream.seek(0, os.SEEK_END)
                stream.write(json.dumps(registry_row, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            persist_case_completion(out, round_label=ROUND, started_ns=started, status=result['status'],
                timing=timing, required_artifact_bytes=artifact_bytes(out))
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-case', type=Path, required=True)
    parser.add_argument('--archive-case', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--intake-python', type=Path, default=paths.INTAKE_PYTHON)
    parser.add_argument('--registry', type=Path, default=paths.RESULTS_DIR / 'registry.jsonl')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--memory-gib', type=float, default=4)
    parser.add_argument('--case-gib', type=float, default=.5)
    parser.add_argument('--minimum-free-gib', type=float, default=10)
    args = parser.parse_args(argv)
    if (not all(math.isfinite(v) for v in (args.memory_gib, args.case_gib, args.minimum_free_gib))
            or not 0 < args.seconds <= 1800 or not 0 < args.memory_gib <= 4
            or not 0 < args.case_gib <= .5 or not 4 <= args.minimum_free_gib <= 10):
        parser.error('Use seconds<=1800, AS<=4GiB, own artifacts<=.5GiB, and explicit reserve4..10GiB')
    if args.worker:
        if args.work_dir is None:
            parser.error('--worker requires --work-dir')
        result = worker(args)
    else:
        if args.out is None:
            parser.error('--out is required')
        result = execute(args)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
