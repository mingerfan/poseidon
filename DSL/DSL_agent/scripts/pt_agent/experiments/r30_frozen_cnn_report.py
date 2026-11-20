"""One finite report pass for an independently accepted, compiled fresh R30 CNN.

The source case remains unchanged. Report observations are never planner inputs.
Only trusted original factories and the parsed frozen DSL enter the worker.
"""
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
from types import SimpleNamespace

from .. import paths
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

ROUND = 'R30-FROZEN-REPORT'
MODE = 'fresh-frozen-cnn-report-once'
MODELS = ('resnet20_silu', 'vgg16_silu_avgpool', 'alexnet_silu_avgpool')
REPORT_COUNT = 8000
GIB = 1024**3
CRITERION = '1e-12+1e-12*abs(reference)'


def read(path):
    return json.loads(Path(path).read_text())


def clean(observation):
    return (observation.get('exit_code') == 0 and observation.get('timed_out') is False
            and observation.get('signal') is None and observation.get('signals_sent') == []
            and observation.get('remaining_pids') == [])


def official_normalization(source):
    found = []
    for node in ast.walk(ast.parse(Path(source).read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'Normalize':
            values = {arg.arg: ast.literal_eval(arg.value) for arg in node.keywords}
            if set(values) == {'mean', 'std'}:
                found.append(values)
    if len(found) != 1 or any(len(found[0][name]) != 3 for name in ('mean', 'std')):
        raise ValueError('One original official inference normalization is required')
    return found[0]


def source_gate(source_case, model):
    """Read metadata and parsed source only; do not open weights or samples."""
    from .r13_common_models import FACTORIES
    from .r30_common_model_e2e import audit_agent, FACTORY_PREFIX, PHASES

    case = Path(source_case).resolve()
    result = read(case / 'result.json')
    completion = read(case / 'completion.json')
    if (model not in MODELS or result.get('round') != 'R30' or result.get('model') != model
            or result.get('status') != 'ok' or result.get('mode') != 'fresh-common-model-agent-e2e'
            or result.get('error') is not None or result.get('monitor_error') is not None
            or any(result.get('completed', {}).get(p) != 1 for p in PHASES)
            or completion.get('status') != 'ok' or completion.get('registry_persisted') is not True
            or completion.get('required_artifacts_persisted') is not True):
        raise ValueError('A complete accepted fresh R30 source and native compilation are required')
    for phase in ('trace', 'compile'):
        observations = [row.get('observation') or {} for row in result.get('processes', [])
                        if row.get('phase') == phase]
        record = result.get('phases', {}).get(phase) or {}
        if (len(observations) != 1 or not clean(observations[0])
                or record.get('status') != 'ok' or not clean(record.get('process') or {})):
            raise ValueError('The fresh native trace and compile must both terminate cleanly')
    work = case / 'work'
    info_path = work / 'intake/model_info/model_info.json'
    calibration = work / 'calibration/calibration.json'
    agent_dir = work / 'agent'
    program = agent_dir / 'program.ptdsl'
    worker = read(agent_dir / 'worker-result.json')
    audit = audit_agent(worker, model=model, model_info=info_path, calibration=calibration, agent_dir=agent_dir)
    stored_path = case / 'post-run-quality-audit.json'
    if Path(result.get('quality_audit', '')).resolve() != stored_path:
        raise ValueError('The source must retain its independent formal quality audit')
    stored = read(stored_path)
    if (any(stored.get(key) != value for key, value in audit.items())
            or Path(stored.get('source_result', '')).resolve() != (agent_dir / 'worker-result.json')):
        raise ValueError('The retained independent quality audit differs from the frozen source')
    identity = lambda text: ast.dump(ast.parse(text), include_attributes=False)
    selected = worker['agent']['confirm']['selected']
    if identity(program.read_text()) != identity(selected['source']):
        raise ValueError('Report program AST differs from the confirmed selection')
    compiled_input = case / 'post-freeze/emit/candidate_trace.mlir'
    archived_input = compiled_input.with_name(compiled_input.name + '.gz')
    if not compiled_input.is_file() and not archived_input.is_file():
        raise ValueError('Retain the compiled input MLIR or its reversible gzip archive')
    native_argv = read(work / 'compile/native-argv.json')
    if (not isinstance(native_argv, list)
            or str(case / 'post-freeze/emit/candidate_trace.mlir') not in native_argv):
        raise ValueError('The native compilation must use the current frozen emitted MLIR')
    info = read(info_path)
    source = info.get('source') or {}
    factory = FACTORY_PREFIX + model
    expected = FACTORIES[factory]
    if (source.get('factory') != factory or source.get('type') != 'torch'
            or info.get('task') != 'classification' or info.get('metric') != 'classification_top1'
            or info.get('threshold') != {'acc_drop_pp': .5}
            or info.get('split', {}).get('kind') != 'legacy_cifar'
            or info.get('split', {}).get('report') != list(range(REPORT_COUNT))):
        raise ValueError('The original model policy and fixed CIFAR report split must be retained')
    for field, key in [('original_source', 'source'), ('checkpoint', 'checkpoint'),
                       ('inference_preprocessing_source', 'inference_preprocessing_source')]:
        if Path(source.get(field, '')).resolve() != Path(expected[key]).resolve() or not Path(expected[key]).is_file():
            raise ValueError('The original factory source, weights and preprocessing paths must match')
    normalization = official_normalization(expected['inference_preprocessing_source'])
    if source.get('normalization') != normalization or source.get('preprocessing_provenance') != 'official inference test':
        raise ValueError('The original official inference normalization must be preserved')
    intake = result.get('phases', {}).get('intake') or {}
    original_math = intake.get('graph_math') or {}
    samples = original_math.get('samples') or []
    if (any(intake.get(name) is not True for name in ('import_completed', 'real_probes_passed',
            'graph_math_passed', 'roundtrip_passed', 'static_coverage'))
            or original_math.get('passed') is not True or len(samples) != 3
            or not all(row.get('passed') is True and row.get('finite') is True for row in samples)):
        raise ValueError('The original R13 three-probe Torch-to-graph proof is required')
    probes = work / 'intake/training_probes.npz'
    probe_source = work / 'intake/training_probe_source.json'
    provenance = read(probe_source)
    if (provenance.get('split') != 'train' or provenance.get('indices') != [0, 1, 2]
            or provenance.get('sample_ids') != ['train:0', 'train:1', 'train:2'] or not probes.is_file()):
        raise ValueError('Only the original three saved training probes may precede report access')
    records = Path(result.get('source', {}).get('records', '')).resolve()
    expected_records = paths.WORK_RESULTS / 'agent-cifar-offline-records-r1804/cifar10-records'
    if records != expected_records.resolve():
        raise ValueError('The original saved CIFAR record source must be retained')
    return dict(source_case=str(case), model=model, model_info=str(info_path), program=str(program),
                factory=factory, records=str(records), normalization=normalization,
                probes=str(probes), probe_source=str(probe_source), quality_audit=str(stored_path),
                source_program_ast_matches=True, session_phase='done', clean_native_compile=True,
                compiled_input_retention='raw' if compiled_input.is_file() else 'reversible-gzip',
                compiled_input_logical_path=str(compiled_input),
                compiled_input_retained_path=str(compiled_input if compiled_input.is_file() else archived_input),
                original_r13_math_checks_reused=3, original_sources={k: str(expected[k]) for k in
                    ('source', 'checkpoint', 'inference_preprocessing_source')}, report_count=REPORT_COUNT,
                reference_criterion=CRITERION, current_script_ast_identity_required=False)


def durable_json(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    else:
        write_json(path, value)
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    directory = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def claim_access(out, gate):
    directory = paths.RESULTS_DIR / 'frozen-cnn-report-access'
    directory.mkdir(exist_ok=True)
    marker = directory / (Path(gate['source_case']).name + '.json')
    durable_json(marker, dict(source_case=gate['source_case'], out=str(Path(out).resolve()),
        model=gate['model'], status='prepared', report_started=False, report_complete=False,
        report_samples=0, automatic_retry=False), exclusive=True)
    return marker


def update_access(args, **updates):
    value = read(args.access_marker)
    if (value.get('source_case') != str(args.source_case.resolve())
            or value.get('out') != str(args.out.resolve()) or value.get('model') != args.model):
        raise ValueError('Report access marker belongs to a different frozen source or output')
    value.update(updates)
    durable_json(args.access_marker, value)


def logits_row(array):
    if tuple(getattr(array, 'shape', ())) != (1, 10) or str(getattr(array, 'dtype', '')) != 'float64':
        raise ValueError('The original classifier must return float64[1,10]')
    values = [float(x) for x in array.tolist()[0]]
    return [value if math.isfinite(value) else None for value in values], all(math.isfinite(v) for v in values)


def report_rows(output, read_sample, reference, approximate, progress):
    """Fixed batch-one pass; every approximate failure remains a report observation."""
    exact_correct = approximate_correct = finite_count = 0
    with Path(output).open('x') as stream:
        try:
            for index in range(REPORT_COUNT):
                image, label = read_sample(index)
                progress(report_samples_read=index + 1, active_sample_id='test:' + str(index))
                if type(label) is not int or not 0 <= label < 10:
                    raise ValueError('The original CIFAR label is outside the classifier classes')
                exact, exact_finite = logits_row(reference(image))
                if not exact_finite:
                    raise ValueError('The original classifier reference became nonfinite')
                error = None
                try:
                    values, finite = logits_row(approximate(image))
                except (FloatingPointError, OverflowError) as exc:
                    values, finite, error = [None] * 10, False, type(exc).__name__ + ': ' + str(exc)
                exact_class = max(range(10), key=exact.__getitem__)
                approx_class = max(range(10), key=values.__getitem__) if finite else None
                exact_correct += int(exact_class == label)
                approximate_correct += int(approx_class == label)
                finite_count += int(finite)
                row = dict(sample_id='test:' + str(index), index=index, source_split='test', label=label,
                    reference=exact, approximate=values, reference_finite=True, approximate_finite=finite,
                    reference_class=exact_class, approximate_class=approx_class, error=error)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                if (index + 1) % 100 == 0:
                    os.fsync(stream.fileno())
                progress(report_samples_completed=index + 1, reference_correct=exact_correct,
                         approximate_correct=approximate_correct, finite_samples=finite_count)
        finally:
            stream.flush()
            os.fsync(stream.fileno())
    return dict(report_samples=REPORT_COUNT, report_completed=True, ids_exactly_test_0_through_7999=True,
                reference_correct=exact_correct, approximate_correct=approximate_correct,
                reference_top1_percent=100 * exact_correct / REPORT_COUNT,
                approximate_top1_percent=100 * approximate_correct / REPORT_COUNT,
                accuracy_drop_pp=100 * (exact_correct - approximate_correct) / REPORT_COUNT,
                all_finite=finite_count == REPORT_COUNT, finite_samples=finite_count,
                report_policy_passed=finite_count == REPORT_COUNT and
                    100 * (exact_correct - approximate_correct) / REPORT_COUNT <= .5,
                batch_size=1, independent_report_observation=True, used_for_selection=False)


def load_context(gate):
    import numpy as np
    import torch
    from ..models.info import ModelInfo
    from ..intake.probe import load_callable
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from .r14_common_worker import check_program_signature
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    info = ModelInfo.load(gate['model_info'])
    program = parse(Path(gate['program']).read_text())
    if (not check(program, {name: value.shape for name, value in info.graph.weights.items()}, graph=info.graph).ok
            or check_program_signature(info, program)['passed'] is not True):
        raise ValueError('The frozen parsed program and original model interface differ')
    original, _ = load_callable(gate['factory'])
    original = original.cpu().eval().double()
    with np.load(gate['probes'], allow_pickle=False) as archive:
        probes, indices = archive['input'], archive['indices']
    if probes.dtype != np.float64 or probes.shape != (3, 3, 32, 32) or indices.tolist() != [0, 1, 2]:
        raise ValueError('The saved original training probes changed shape, dtype or indices')
    if not np.isfinite(probes).all():
        raise ValueError('The saved original training probes must be finite')
    return SimpleNamespace(info=info, program=program, original=original, probes=probes, gate=gate)


def reference_output(context, image):
    import torch
    with torch.no_grad():
        return context.original(torch.from_numpy(image)).detach().cpu().numpy()


def approximate_output(context, image):
    from ..backends import BackendError
    from ..tensor_sim import run_tensor
    try:
        return run_tensor(context.program, context.info.graph.weights, image, cache=False)
    except BackendError as exc:
        # The existing Tensor engine wraps this precise numerical guard in
        # BackendError. Other backend failures retain their original exception.
        if str(exc) == 'Mock arithmetic produced a nonfinite value':
            raise FloatingPointError(str(exc)) from exc
        raise


def training_math(context):
    from ..analyzer import reference_forward
    from ..tensor_sim import run_tensor
    from .r15_rule_validation_worker import _packed_logits, check_math
    rows = []
    for index in range(3):
        image = context.probes[index:index + 1]
        graph = reference_forward(context.info.graph, {context.info.graph.inputs[0]: image})[context.info.graph.outputs[0]]
        original = reference_output(context, image)
        tensor_exact = run_tensor(context.program, context.info.graph.weights, image, cache=False, act_mode='exact')
        mock_exact = _packed_logits(context.program, context.info.graph.weights, image, act_mode='exact')
        tensor = approximate_output(context, image)
        mock = _packed_logits(context.program, context.info.graph.weights, image, act_mode='program')
        checks = dict(original_torch_vs_graph=check_math(original, graph),
            graph_vs_tensor_exact=check_math(graph, tensor_exact), graph_vs_mock_exact=check_math(graph, mock_exact),
            tensor_vs_mock_program=check_math(tensor, mock))
        rows.append(dict(sample_id='train:' + str(index), checks=checks))
        if not all(row['passed'] for row in checks.values()):
            raise ValueError('The frozen training-probe mathematical check failed: train:' + str(index))
    return dict(passed=True, criterion=CRITERION, samples=rows, checks_completed=12,
                training_samples=3, report_samples=0)


def read_report_sample(context, index):
    from ..data import cifar
    if not 0 <= index < REPORT_COUNT or Path(context.gate['records']).resolve() != cifar.RECORDS.resolve():
        raise ValueError('The report reader must use the original fixed CIFAR record source')
    pixels, labels = cifar.read_rows('test', [index])
    norm = context.gate['normalization']
    return cifar.normalize(pixels, normalization=(norm['mean'], norm['std'])), int(labels[0])


def worker(args):
    limit = int(args.memory_gib * GIB)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    result = dict(status='running', model=args.model, report_samples_read=0, report_samples_completed=0,
                  report_completed=False, llm_calls=0, compiler_runs=0, ckks_runs=0, batch_size=1,
                  address_space_limit_bytes=limit)
    events = StageRecorder(args.out / 'worker-events.jsonl')
    def save(**values):
        result.update(values)
        write_json(args.out / 'worker-result.json', result)
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError('Frozen report reached its finite wall allowance; preserve partial evidence')
    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, args.seconds)
        with events.span('frozen-report-source-validation'):
            gate = source_gate(args.source_case, args.model)
            update_access(args, status='checking-training-math')
        with events.span('frozen-report-original-model-and-saved-probes'):
            context = load_context(gate)
        with events.span('frozen-report-twelve-training-math-checks'):
            proof = training_math(context)
            save(training_math=proof)
            if proof.get('passed') is not True or proof.get('checks_completed') != 12:
                raise ValueError('Twelve successful training-probe checks must precede report access')
        with events.span('frozen-report-test-0-through-7999'):
            update_access(args, status='reading-report', report_started=True)
            measured = report_rows(args.out / 'report-rows.jsonl',
                lambda index: read_report_sample(context, index),
                lambda image: reference_output(context, image),
                lambda image: approximate_output(context, image), save)
            save(**measured)
            update_access(args, status='complete', report_complete=True, report_samples=REPORT_COUNT)
        save(status='ok')
    except BaseException as exc:
        failure_kind = ('timeout' if isinstance(exc, TimeoutError) else
                        'resource_error' if isinstance(exc, MemoryError) else 'execution_error')
        save(status='error', failure_kind=failure_kind, error=type(exc).__name__ + ': ' + str(exc))
        update_access(args, status='error', report_samples=result['report_samples_read'],
                      error=result['error'], failure_kind=failure_kind, automatic_retry=False)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        save()
    return 0 if result['status'] == 'ok' else 1


def register(out, result):
    from ..registry import git_rev
    row = dict(run_id=out.name, round=ROUND, model=result['model'], mode=MODE, git_rev=git_rev(),
        metrics={key: result.get(key) for key in ('status', 'source_case', 'report_completed',
            'report_samples_read', 'report_samples_completed', 'reference_top1_percent',
            'approximate_top1_percent', 'accuracy_drop_pp', 'all_finite', 'report_policy_passed')},
        artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR / 'registry.jsonl').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        if any(read_row.get('run_id') == out.name for read_row in
               (json.loads(line) for line in stream if line.strip())):
            raise ValueError('The report run id already exists')
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def execute(args):
    from .bert_sst2 import budget_guard
    started = time.perf_counter_ns()
    out = args.out.resolve()
    if out == paths.RESULTS_DIR.resolve() or not out.is_relative_to(paths.RESULTS_DIR.resolve()):
        raise ValueError('The new report output must be a child of PT_AGENT_RESULTS_ROOT')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'tmp').mkdir()
    events = StageRecorder(out / 'events.jsonl')
    result = dict(round=ROUND, mode=MODE, model=args.model, source_case=str(args.source_case.resolve()),
        status='running', report_completed=False, report_samples_read=0, report_samples_completed=0,
        access_marker=None, processes=[], llm_calls=0, compiler_runs=0, ckks_runs=0,
        feedback_to_agent=False, automatic_retry=False, source_case_modified=False)
    try:
        with events.span('frozen-report-source-preflight'):
            gate = source_gate(args.source_case, args.model)
            write_json(out / 'source-preflight.json', gate)
            marker = claim_access(out, gate)
            result['access_marker'] = str(marker)
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(args.seconds + 60),
            '--python', str(args.intake_python), '--', '-B', '-m', __spec__.name,
            '--worker', '--model', args.model, '--source-case', str(args.source_case.resolve()),
            '--out', str(out), '--access-marker', str(marker), '--seconds', str(args.seconds),
            '--memory-gib', str(args.memory_gib)]
        environment = {key: value for key, value in os.environ.items()
            if not key.startswith(('LD_', 'NIX_', '__NIX_', 'HECATE_'))
            and key not in ('HECATE', 'IN_NIX_SHELL', 'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')}
        environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TMPDIR=str(out / 'tmp'))
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            environment[key] = '1'
        with budget_guard(out) as violations, events.span('frozen-report-bounded-worker', argv=command):
            observation = run_process(command, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=out / 'worker.log',
                resources=out / 'resources.jsonl', timeout=args.seconds + 120,
                mem_gib=args.memory_gib, fsize_gib=8, env=environment)
            result['processes'].append(dict(argv=command, observation=observation))
            if (out / 'worker-result.json').is_file():
                result['worker'] = read(out / 'worker-result.json')
            measured = result.get('worker', {})
            if (violations or not clean(observation) or measured.get('status') != 'ok'
                    or measured.get('report_completed') is not True
                    or measured.get('report_samples_read') != REPORT_COUNT
                    or measured.get('report_samples_completed') != REPORT_COUNT
                    or measured.get('ids_exactly_test_0_through_7999') is not True
                    or any(measured.get(key) != 0 for key in ('llm_calls', 'compiler_runs', 'ckks_runs'))):
                raise RuntimeError('The report worker did not complete; retain its access marker and partial outputs')
        result['status'] = 'ok'
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__ + ': ' + str(exc))
    finally:
        if (out / 'worker-result.json').is_file():
            result['worker'] = read(out / 'worker-result.json')
            for key, value in result['worker'].items():
                if key not in ('status', 'error', 'model'):
                    result[key] = value
        outer_events = read_events(out / 'events.jsonl')
        worker_events = read_events(out / 'worker-events.jsonl')
        parent = next((row['event_id'] for row in outer_events
            if row.get('type') == 'begin' and row.get('stage_id') == 'frozen-report-bounded-worker'), None)
        for row in worker_events:
            if row.get('parent_id') is None:
                row['parent_id'] = parent
        timing = summarize_events(outer_events + worker_events, time.perf_counter_ns())
        write_json(out / 'timing.json', timing)
        result['recorded_elapsed_seconds'] = (time.perf_counter_ns() - started) / 1e9
        write_json(out / 'result.json', result)
        register(out, result)
        persist_case_completion(out, round_label=ROUND, started_ns=started, status=result['status'],
            timing=timing, required_artifact_bytes=artifact_bytes(out))
    return 0 if result['status'] == 'ok' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--source-case', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--intake-python', type=Path, default=paths.INTAKE_PYTHON)
    parser.add_argument('--seconds', type=int, default=7200)
    parser.add_argument('--memory-gib', type=float, default=6)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--access-marker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 0 < args.seconds <= 7200 or not 0 < args.memory_gib <= 6:
        parser.error('Use a finite report allowance of at most7200seconds and at most6GiB address space')
    for name in ('source_case', 'out', 'intake_python'):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    if args.worker:
        if args.access_marker is None:
            parser.error('The trusted report worker requires its explicit access marker')
        return worker(args)
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
