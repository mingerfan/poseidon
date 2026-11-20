"""Serial local compiler comparisons on original, already saved model programs.

The outer entry owns RESULTS and registry. Inner phases receive every path by
argv, export trusted lowering directly, and never construct package manifests.
The compiler's existing operator-spec identifier is read from a saved successful
command template. No file digest is calculated or verified here.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import threading
import time

from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .completion_metrics import persist_case_completion

ROUND = 'R29'
MODEL = 'pt_agent.experiments.r13_common_models:resnet20_silu'
MODEL_FACTORIES = (MODEL, 'pt_agent.experiments.r13_common_models:vgg16_silu_avgpool',
                   'pt_agent.experiments.r13_common_models:alexnet_silu_avgpool')
SYSTEMS = ('dacapo-optimized', 'dacapo-depth-dp', 'agent-pars')
GIB = 1024 ** 3
MAX_CASE_GIB = 64
LATENCY_KEYS = dict(add_cc='add_double', sub_cc='add_double', add_cp='add_single', sub_cp='add_single',
                    negate='negate_single', rotate='rotate_single', mul_cc='mul_double', mul_cp='mul_single',
                    rescale='rescale_single', mod_switch='modswitch_single', boot='bootstrap_single')


def _read(path):
    return json.loads(Path(path).read_text())


def _size(path):
    result = 0
    for parent, _, names in os.walk(path):
        for name in names:
            try:
                result += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                pass
    return result


def completion_marker(out, *, started_ns, status, timing, required_artifact_bytes, clock=time.perf_counter_ns):
    """Call after required artifacts and the unique registry row are closed.

    The endpoint is captured before this marker's own serialization. Its final
    byte inventory includes the marker, while result.json's preceding inventory
    explicitly covers the required artifacts alone.
    """
    return persist_case_completion(out, round_label=ROUND, started_ns=started_ns, status=status,
                                   timing=timing, required_artifact_bytes=required_artifact_bytes, clock=clock)


def case_run_id(out, system):
    return Path(out).parent.name + '-' + system


def selected_model(args):
    """Keep the original programmatic ResNet default and explicit CLI choices."""
    factory = getattr(args, 'model_factory', MODEL)
    if factory not in MODEL_FACTORIES:
        raise ValueError('Select one of the three original common-model factories')
    return factory


def validate_model_identity(record, factory):
    if factory not in MODEL_FACTORIES or record.get('source', {}).get('factory') != factory:
        raise ValueError('Original ModelInfo factory differs from the selected original model')
    return record['source']


def validate_rule_source(result, *, result_path, model_info, model_info_record, model_factory=MODEL):
    """Check saved original-source and accepted internal-rule metadata only."""
    original = validate_model_identity(model_info_record, model_factory)
    worker = result.get('worker') or {}
    preparation = worker.get('source_preparation') or {}
    if (result.get('round') != 'R17' or result.get('status') != 'ok'
            or result.get('model') != model_factory or worker.get('model') != model_factory
            or worker.get('status') != 'ok'
            or any(worker.get(key) is not True for key in
                   ('source_checked','evaluation_completed','task_passed','domain_passed','heldout_acceptance','all_finite'))
            or worker.get('completed_samples') != 1000 or worker.get('heldout_samples') != 1000):
        raise ValueError('The selected original model requires its completed accepted R17 heldout rule record')
    if (worker.get('model_source') != original
            or Path(preparation.get('model_info', '')).resolve() != Path(model_info).resolve().parent):
        raise ValueError('The rule quality record refers to a different original ModelInfo or model source')
    r13_path = Path(preparation.get('r13_result', ''))
    r13 = _read(r13_path)
    if (r13.get('round') != 'R13' or r13.get('status') != 'ok' or r13.get('model') != model_factory
            or Path((r13.get('worker') or {}).get('artifacts', {}).get('model_info', '')).resolve()
               != Path(model_info).resolve().parent):
        raise ValueError('Original R13 intake and ModelInfo identities differ')
    program = Path(worker.get('artifacts', {}).get('program', ''))
    if (program.resolve().parent != (Path(result_path).resolve().parent/'worker')
            or not program.is_file()):
        raise ValueError('The accepted rule program must remain in its recorded original worker directory')
    return dict(source_result=str(Path(result_path).resolve()), program=str(program.resolve()),
                model_factory=model_factory, original_r13_result=str(r13_path.resolve()),
                independent_source_checked=True,
                source_quality=dict(validation_metrics=worker.get('validation_metrics'), max_u=worker.get('max_u'),
                    heldout_samples=worker['heldout_samples'], candidate_frozen=worker.get('candidate_frozen'),
                    confirm_samples=worker.get('confirm_samples'), report_samples=worker.get('report_samples')),
                scope='internal fixed-rule candidate; formal accuracy/freeze remain pending')


def validate_agent_source(result, audit, *, result_path, model_info, model_factory=MODEL):
    """Require independent final quality audit as well as the worker's own gate."""
    worker = result.get('worker') or {}
    if model_factory not in MODEL_FACTORIES:
        raise ValueError('Unknown original model factory')
    model = model_factory.split(':', 1)[1]
    program = worker.get('selected_program')
    if (result.get('round') != 'R28' or result.get('status') != 'ok' or result.get('model') != model
            or worker.get('model') != model or worker.get('status') != 'ok' or worker.get('frozen') is not True
            or (worker.get('quality_acceptance') or {}).get('passed') is not True or not program):
        raise ValueError('Agent source lacks a successful frozen quality acceptance')
    probes = worker.get('strict_math_probes')
    if (not isinstance(probes, list) or len(probes) != 3
            or any(row.get('sample_id') != 'test:' + str(9000 + i)
                   or (row.get('exact') or {}).get('passed') is not True
                   or (row.get('approximate') or {}).get('passed') is not True
                   for i, row in enumerate(probes))):
        raise ValueError('Agent source lacks all three strict exact/program math pairs')
    if (audit.get('status') != 'quality_accepted' or audit.get('formal_frozen') is not True
            or audit.get('quality_acceptance_passed') is not True or audit.get('strict_math_passed') is not True
            or audit.get('model') != model
            or Path(audit.get('source_result', '')).resolve() != Path(result_path).resolve()
            or Path(audit.get('selected_program', '')).resolve() != Path(program).resolve()
            or Path(worker.get('source_model_info', '')).resolve() != Path(model_info).resolve()):
        raise ValueError('Independent post-run quality audit or original ModelInfo identity differs')
    return dict(program=str(Path(program).resolve()), source_result=str(Path(result_path).resolve()),
                independent_audit_passed=True, source_quality=worker['quality_acceptance'],
                historical_prediction=dict(boots=worker.get('boots_pred'), latency_us=worker.get('latency_pred_us')))


def compiler_argv(template, mode, *, binary, source, profile, output, debug_snapshots=False):
    """Reuse native metadata and keep debug IR snapshots explicitly optional."""
    if mode not in ('dacapo-optimized', 'dacapo-depth-dp', 'pars'):
        raise ValueError('Unknown compile mode')
    if (not isinstance(template, list) or any(not isinstance(arg, str) for arg in template)
            or template.count('--pars') != 1 or template.count('-o') != 1
            or template[:2] != ['{hecate_opt}', '{input}']):
        raise ValueError('Expected the saved successful pars compiler template')
    identities = [arg for arg in template if arg.startswith('--runtime-plan-operator-spec-sha256=')]
    if len(identities) != 1 or not re.fullmatch(r'--runtime-plan-operator-spec-sha256=sha256:[0-9a-f]{64}', identities[0]):
        raise ValueError('Native operator metadata identifier missing')
    if ('--runtime-plan-context-id=qwen-bootstrap48' not in template
            or '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5' not in template
            or '--waterline=43' not in template):
        raise ValueError('The original common compiler parameter group differs')
    argv = []
    substitutions = {'{hecate_opt}': str(binary), '{input}': str(source),
                     '{package}/config/compiler-config.json': str(profile),
                     '{output_dir}/lowered.mlir': str(output)}
    for arg in template:
        if arg == '--enable-debug-printer' or arg.startswith('--enable-debug-printer='):
            continue
        if arg == '--pars' and mode.startswith('dacapo-'):
            argv.extend(['--dacapo', '--boot-placement=' + ('optimized' if mode.endswith('optimized') else 'depth-dp'),
                         '--output-val=10', '--threshold=0.5', '--dump-pass-pipeline'])
        else:
            for old, new in substitutions.items():
                arg = arg.replace(old, new)
            argv.append(arg)
    if debug_snapshots:
        argv.insert(argv.index('-o'), '--enable-debug-printer')
    if any('{' in arg or '}' in arg for arg in argv):
        raise ValueError('Unmapped compiler command placeholder')
    return argv


@contextmanager
def native_metadata_commands(template):
    """Prevent legacy command builders from recalculating a file digest.

    management_profile and the original cost/plan algorithms still read their
    original JSON and compiler marker. Only command construction is adapted.
    """
    from .. import toolchain
    original = toolchain.compile_argv
    def command(mode, boot, run_dir):
        if boot != 'qwen48':
            raise ValueError('R29 uses the existing common qwen48 compiler group')
        return compiler_argv(template, mode, binary='/hecate-opt', source='/out/candidate_trace.mlir',
                             profile='/out/boot-compiler-config.json', output='/out/lowered.mlir')
    toolchain.compile_argv = command
    try:
        yield
    finally:
        toolchain.compile_argv = original


def plan_observation(plan, profile):
    """Counts, locations and table prediction from one completed RuntimePlan."""
    values = {row['id']: row for row in plan['values']}
    table = {key.removeprefix('earth.'): value for key, value in profile['latencyTable'].items()}
    counts, by_op, untimed = Counter(), Counter(), Counter()
    missing, boots = [], []
    execution = plan['execution']
    for index, step in enumerate(execution):
        op = step.get('op') or step.get('kind')
        counts[op] += 1
        if op == 'boot':
            outputs = ([step['output']] if step.get('output') is not None
                       else step.get('outputs') or [])
            boots.append(dict(execution_index=index, instruction_id=step.get('id'), inputs=step.get('inputs'),
                              outputs=outputs,
                              input_levels=[values.get(value, {}).get('level') for value in step.get('inputs', [])],
                              output_levels=[values.get(value, {}).get('level') for value in outputs],
                              source_location=step.get('source_location', step.get('location'))))
        key = LATENCY_KEYS.get(op)
        if key is None:
            untimed[op] += 1
            if step.get('kind') == 'compute' or op not in ('encode', 'release', 'fence', 'allocate', 'transfer'):
                missing.append(dict(index=index, op=op, reason='operation has no profile entry'))
            continue
        inputs = step.get('inputs') or []
        level = values.get(inputs[0], {}).get('level') if inputs else None
        durations = table.get(key)
        if (type(level) is not int or not isinstance(durations, list) or not 0 <= level < len(durations)
                or not isinstance(durations[level], (int, float)) or not math.isfinite(durations[level])):
            missing.append(dict(index=index, op=op, level=level, reason='latency level is unavailable'))
        else:
            by_op[op] += durations[level]
    return dict(compiled_bootstraps=len(boots), bootstrap_positions=boots, execution_ops=dict(counts),
                execution_steps=len(execution), values=len(values), predicted_latency_us=None if missing else sum(by_op.values()),
                prediction=dict(source='unchanged compiler profile latencyTable', unit='microseconds',
                                scope='table sum for serialized execution compute operations; excludes initialization, keygen and host overhead',
                                measured_local_ckks=False, table_known_sum_us=sum(by_op.values()), by_op_us=dict(by_op),
                                untimed=dict(untimed), missing=missing))


def _worker_inputs(args):
    from ..models.info import ModelInfo
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from .r14_common_worker import check_program_signature
    info = ModelInfo.load(args.model_info)
    if info.source.get('factory') != selected_model(args):
        raise ValueError('Original model source differs')
    program = parse(args.program.read_text())
    checked = check(program, {key: value.shape for key, value in info.graph.weights.items()}, graph=info.graph)
    signature = check_program_signature(info, program)
    if not checked.ok or signature['passed'] is not True:
        raise ValueError('Program typecheck or original interface check failed')
    return info, program, signature


def worker(args):
    """No RESULTS/registry/config.load access; every source is an explicit argument."""
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    target = work / (args.worker_phase + '-result.json')
    if target.exists():
        raise ValueError('Preserve earlier worker phase')
    recorder = StageRecorder(work / (args.worker_phase + '-events.jsonl'))
    result = dict(round=ROUND, mode=args.system, phase=args.worker_phase, model=selected_model(args), status='running', error=None,
                  source_program=str(args.program), source_model_info=str(args.model_info),
                  extra_integrity_checks=0, keygen_runs=0, real_ckks_runs=0, sampled_images=0,
                  compiled_bootstraps=None, predicted_latency_us=None, CKKS_latency=None)
    began = time.perf_counter_ns()
    try:
        template = _read(args.metadata_template)
        with native_metadata_commands(template):
            if args.worker_phase in ('placement', 'emit'):
                with recorder.span('restore_and_typecheck'):
                    info, program, signature = _worker_inputs(args)
                    result['signature'] = signature
                if args.worker_phase == 'placement':
                    from ..plan import plan_program
                    with recorder.span('agent_program_plan'):
                        if program.header.refresh != 'agent' or program.header.placement != 'program-plan':
                            raise ValueError('Frozen Agent program must retain program-plan placement')
                        plan, backend, placement = plan_program(program, info.graph.weights, profile='qwen48')
                        write_json(work / 'refresh-plan.json', plan)
                        result.update(boots_pred=placement.boots, latency_pred_us=placement.latency_us,
                                      boot_nodes=placement.boot_nodes, placement_variant=plan['placement'],
                                      refresh_plan=str(work / 'refresh-plan.json'))
                        del backend, placement
                        gc.collect()
                else:
                    from ..backends import EmitBackend, RefreshBackend
                    from ..lower import lower
                    emit = work / 'emit'
                    with recorder.span('trusted_emit_lower'):
                        backend = EmitBackend(program.header.slots, frontend=args.frontend,
                                              constants=emit / 'constants.npz', save_dir=emit)
                        active = backend
                        if args.system == 'agent-pars':
                            plan = _read(work / 'refresh-plan.json')
                            active = RefreshBackend(backend, policy='plan', plan=plan, **plan['levels'])
                        lower(program, info.graph.weights, active, act_mode='program')
                        result['emit_stats'] = backend.stats()
                        if args.system == 'agent-pars' and result['emit_stats']['counts']['boot'] != plan['boots']:
                            raise ValueError('Emit replay bootstrap count differs from the original plan')
                    with recorder.span('trusted_emit_write'):
                        text, constants = backend.render()
                        estimate = len(text.encode()) + sum(value.nbytes + 1024 for value in constants.values()) + 1024**2
                        result['emit_estimated_bytes'] = estimate
                        if estimate > args.case_gib * GIB - GIB:
                            raise RuntimeError('Trusted Emit output does not fit the case allowance')
                        free = shutil.disk_usage(work).free
                        result['emit_disk_projection'] = dict(free_before_bytes=free,
                            proposed_source_and_constants_bytes=estimate, free_after_estimated_emit_bytes=free-estimate,
                            reserve_bytes=10*GIB, scope='conservative Emit source/archive estimate; compiler artifacts remain separately monitored')
                        if free - estimate < 10*GIB:
                            raise RuntimeError('Trusted Emit output would consume the 10 GiB disk reserve')
                        del text, constants
                        backend.write(emit)
                        result['trace_source'] = str(emit / 'candidate_trace.py')
                        result['constants'] = str(emit / 'constants.npz')
                        del backend, active
                        gc.collect()
            elif args.worker_phase == 'trace':
                from .. import toolchain
                emit = work / 'emit'
                host = toolchain.host_paths()
                env = dict(os.environ, HECATE=str(host['hecate']), HECATE_BUILD=str(host['hecate_build']),
                           LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'])
                with recorder.span('trusted_frontend_trace'):
                    observation = run_process([sys.executable, '-B', str(emit / 'candidate_trace.py')], cwd=emit,
                                              log=work / 'trace-native.log', resources=work / 'trace-native-resources.jsonl',
                                              timeout=args.trace_seconds, mem_gib=args.memory_gib, fsize_gib=8, env=env)
                    result['process'] = observation
                    if observation['exit_code'] or observation['timed_out'] or observation['signal'] is not None or observation['remaining_pids']:
                        raise RuntimeError('Trusted native trace failed')
                    mlir = emit / 'candidate_trace.mlir'
                    if not mlir.is_file() or mlir.stat().st_size == 0:
                        raise RuntimeError('Trace returned no nonempty MLIR')
                    result['mlir'] = dict(path=str(mlir), bytes=mlir.stat().st_size)
            elif args.worker_phase == 'compile':
                from .. import toolchain
                emit = work / 'emit'
                command = compiler_argv(template, 'pars' if args.system == 'agent-pars' else args.system,
                                        binary=toolchain.host_paths()['hecate_opt'], source=emit / 'candidate_trace.mlir',
                                        profile=args.profile, output=emit / 'lowered.mlir',
                                        debug_snapshots=getattr(args, 'debug_snapshots', False))
                result['debug_snapshots_enabled'] = getattr(args, 'debug_snapshots', False)
                write_json(work / 'compile-argv.json', command)
                with recorder.span('native_compiler_7200s'):
                    observation = run_process(command, cwd=emit, log=work / 'compile-native.log',
                                              resources=work / 'compile-native-resources.jsonl', timeout=args.compile_seconds,
                                              mem_gib=args.memory_gib, fsize_gib=8,
                                              env=dict(os.environ, LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
                    result['process'] = observation
                    if observation['exit_code'] or observation['timed_out'] or observation['signal'] is not None or observation['remaining_pids']:
                        raise RuntimeError('Native compiler failed')
                    plans = sorted(emit.glob('lowered.*.runtime-plan.json'))
                    if len(plans) != 1:
                        raise RuntimeError('Expected exactly one complete original RuntimePlan')
                    result['runtime_plan'] = str(plans[0])
                with recorder.span('observe_placement_and_prediction'):
                    observed = plan_observation(_read(plans[0]), _read(args.profile))
                    write_json(work / 'placement-and-prediction.json', observed)
                    result.update(observed)
                    if args.system == 'agent-pars':
                        expected = _read(work / 'refresh-plan.json')['boots']
                        result['bootstrap_count_matches_prediction'] = observed['compiled_bootstraps'] == expected
                        if not result['bootstrap_count_matches_prediction']:
                            raise RuntimeError('Compiled and planned bootstrap counts differ')
            result['status'] = 'ok'
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}'[:4000])
    finally:
        result['elapsed_ns'] = time.perf_counter_ns() - began
        result['timing'] = summarize_events(read_events(recorder.path))
        write_json(target, result)
    return 0 if result['status'] == 'ok' else 1


def _launch_worker(args, system, phase, program, work):
    from .. import paths
    seconds = args.compile_seconds if phase == 'compile' else args.trace_seconds if phase == 'trace' else 1800
    inner = ['-m', 'pt_agent.experiments.r29_local_compile_matrix', '--worker-phase', phase,
             '--system', system, '--program', str(program), '--model-info', str(args.model_info),
             '--model-factory', selected_model(args),
             '--metadata-template', str(args.metadata_template), '--profile', str(args.profile),
             '--frontend', str(args.frontend), '--work-dir', str(work), '--trace-seconds', str(args.trace_seconds),
             '--compile-seconds', str(args.compile_seconds), '--memory-gib', str(args.memory_gib), '--case-gib', str(args.case_gib)]
    if getattr(args, 'debug_snapshots', False):
        inner.append('--debug-snapshots')
    return [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(seconds + 120),
            '--python', str(paths.INTAKE_PYTHON), '--', *inner], seconds + 240


def archive_emit_constants_after_trace(result, work):
    """Reversibly remove only this case's redundant NPZ after a clean trace.

    The native compiler consumes the self-contained dense-constant MLIR. Replaying
    candidate_trace.py later requires the recorded gzip restoration first.
    """
    from .r34_archive_text import archive_one, archive_state
    def clean(observation):
        return (observation.get('exit_code') == 0 and observation.get('timed_out') is False
                and observation.get('signal') is None and observation.get('remaining_pids') == []
                and observation.get('signals_sent') == [])
    trace = result.get('phases', {}).get('trace') or {}
    outer = [row.get('observation') or {} for row in result.get('processes', []) if row.get('phase') == 'trace']
    if (result.get('completed', {}).get('trace') != 1 or trace.get('status') != 'ok'
            or len(outer) != 1 or not clean(outer[0]) or not clean(trace.get('process') or {})
            or result.get('attempted', {}).get('compile') or result.get('completed', {}).get('compile')):
        raise ValueError('Archive only after a complete clean trace and before compiler entry')
    work = Path(work)
    source = work / 'emit/constants.npz'
    mlir = work / 'emit/candidate_trace.mlir'
    if (Path(trace.get('mlir', {}).get('path', '')) != mlir or not mlir.is_file()
            or mlir.stat().st_size == 0
            or Path((result.get('phases', {}).get('emit') or {}).get('constants', '')) != source):
        raise ValueError('Archive paths must belong to this case and its completed trace')
    if (not source.is_file() or source.is_symlink() or source.resolve() != source.absolute()
            or source.stat().st_nlink != 1):
        raise ValueError('The Emit NPZ must remain the original regular unlinked case file')
    target = source.with_name(source.name + '.gz')
    temporary = target.with_name(target.name + '.partial')
    if any(path.exists() or path.is_symlink() for path in (target, temporary)):
        raise ValueError('Preserve previous archive or partial output')
    recovery_path = work / 'emit-constants-recovery.json'
    recovery = dict(status='prepared', source=str(source), archive=str(target),
                    source_bytes=source.stat().st_size, source_mtime_ns=source.stat().st_mtime_ns,
                    restore_argv=['gzip', '-dk', '--', str(target)],
                    scope='Restore before replaying candidate_trace.py; compiler consumes the completed MLIR',
                    extra_integrity_checks=0)
    write_json(recovery_path, recovery)
    with recovery_path.open('rb') as stream:
        os.fsync(stream.fileno())
    directory = os.open(str(work), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    try:
        observed = archive_one(source, target, temporary)
    except BaseException as exc:
        recovery.update(status='error', error=f'{type(exc).__name__}: {exc}',
                        interrupted_state=archive_state(source, target, temporary))
        write_json(recovery_path, recovery)
        raise
    recovery.update(status=observed['status'], observation=observed)
    write_json(recovery_path, recovery)
    return dict(status='ok', archive=observed, recovery_record=str(recovery_path),
                restore_argv=recovery['restore_argv'], trace_replay_requires_restore=observed['status']=='compressed',
                source_constants=str(source), compiled_input=str(mlir), extra_integrity_checks=0)


def run_case(args, system, out):
    """A case is independent; terminal failure never consumes another case's attempt."""
    started = time.perf_counter_ns()
    from .. import paths
    out.mkdir()
    work = out / 'work'
    work.mkdir()
    (work / 'tmp').mkdir()
    recorder = StageRecorder(out / 'events.jsonl')
    result = dict(run_id=case_run_id(out, system), round=ROUND, model=selected_model(args), system=system, status='running', error=None, git_rev=args.git_rev,
                  phases={}, processes=[], attempted={}, completed={}, compiled_bootstraps=None, predicted_latency_us=None,
                  CKKS_latency=None, accuracy_acceptance=None, formal_matrix_complete=False, extra_integrity_checks=0,
                  llm_calls=0, keygen_runs=0, real_ckks_runs=0, sampled_images=0, artifact_bytes=0,
                  native_launcher_existing_check_permitted=True, resource_monitor_error=None,
                  source_model_info=str(args.model_info), parameter_group='original qwen48 compiler JSON; independent of Orbit N17 group')
    result['debug_snapshots_enabled'] = getattr(args, 'debug_snapshots', False)
    result['archive_emit_constants_after_trace'] = getattr(args, 'archive_emit_constants_after_trace', False)
    result['resource_limits'] = {key:getattr(args,key,None) for key in
                                ('memory_gib','case_gib','trace_seconds','compile_seconds')}
    result['completion_path'] = str(out / 'completion.json')
    result['artifact_bytes_scope'] = 'required artifacts before completion marker; final case total is completion.json:case_artifact_bytes'
    stop = threading.Event()
    in_process = threading.Event()
    def interrupted(signum, _frame):
        result['interrupted_signal'] = signum
        raise RuntimeError('Case interrupted by signal ' + str(signum))
    old_handlers = {signum: signal.signal(signum, interrupted) for signum in (signal.SIGTERM, signal.SIGINT)}
    env = {key: value for key, value in os.environ.items() if not key.startswith(('LD_', 'NIX_', '__NIX_', 'HECATE_'))
           and key not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'HECATE', 'IN_NIX_SHELL')}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TMPDIR=str(work / 'tmp'))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    def monitor():
        while not stop.wait(1):
            try:
                amount, free = _size(out), shutil.disk_usage(out).free
                with (out / 'artifact-samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(clock_ns=time.perf_counter_ns(), artifact_bytes=amount, free_bytes=free)) + '\n')
                if amount > args.case_gib * GIB or free < 10 * GIB:
                    raise RuntimeError('Case disk allowance or 10 GiB reserve exhausted')
            except BaseException as exc:
                result['resource_monitor_error'] = f'{type(exc).__name__}: {exc}'
                if in_process.is_set():
                    os.kill(os.getpid(), signal.SIGTERM)
                return
    watcher = threading.Thread(target=monitor, daemon=True)
    try:
        with recorder.span('source_gate'):
            factory = selected_model(args)
            # Old direct Namespace callers retain their original ResNet gate.
            # All CLI invocations explicitly carry the model and use the full gate.
            strict_source = hasattr(args, 'model_factory')
            if strict_source:
                model_info_record = _read(args.model_info)
                validate_model_identity(model_info_record, factory)
            result['source_contract'] = ('explicit-original-factory-and-quality-json' if strict_source
                                         else 'legacy-direct-resnet-source-envelope')
            if system == 'agent-pars':
                if args.agent_result is None or args.agent_audit is None:
                    raise ValueError('Agent compile awaits a successful result and independent quality audit')
                source = validate_agent_source(_read(args.agent_result), _read(args.agent_audit),
                                               result_path=args.agent_result, model_info=args.model_info,
                                               model_factory=factory)
                program = Path(source['program'])
                result['source'] = source
            else:
                source = _read(args.rule_result)
                if strict_source:
                    result['source'] = validate_rule_source(source,result_path=args.rule_result,
                        model_info=args.model_info,model_info_record=model_info_record,model_factory=factory)
                    program = Path(result['source']['program'])
                elif (source.get('round') != 'R17' or source.get('status') != 'ok' or source.get('model') != factory
                        or (source.get('worker') or {}).get('heldout_acceptance') is not True):
                    raise ValueError('The original accepted R17 rule source is required')
                else:
                    program = Path(source['worker']['artifacts']['program'])
                    result['source'] = dict(source_result=str(args.rule_result), program=str(program),
                                           scope='internal fixed-rule candidate; formal accuracy/freeze remain pending')
            if not program.is_file() or not args.model_info.is_file():
                raise FileNotFoundError('Saved original program or ModelInfo is absent')
            amount = sum(path.stat().st_size for path in (program, args.profile, args.metadata_template, Path(__file__)))
            free = shutil.disk_usage(out).free
            result['source_disk_projection'] = dict(free_before_bytes=free, required_copy_bytes=amount,
                                                      reserve_bytes=10*GIB)
            if free - amount < 10*GIB:
                raise RuntimeError('Saved source copies would consume the 10 GiB disk reserve')
            shutil.copyfile(program, work / 'program.ptdsl')
            shutil.copyfile(args.profile, work / 'compiler-profile.json')
            shutil.copyfile(args.metadata_template, work / 'source-compile-template.json')
            shutil.copyfile(Path(__file__), out / 'driver.py')
        watcher.start()
        phases = ['placement', 'emit', 'trace'] if system == 'agent-pars' else ['emit', 'trace']
        if getattr(args, 'archive_emit_constants_after_trace', False):
            phases.append('archive_emit_constants')
        phases.append('compile')
        for phase in phases:
            if result['resource_monitor_error']:
                raise RuntimeError(result['resource_monitor_error'])
            result['attempted'][phase] = 1
            write_json(out / 'progress.json', result)
            if phase == 'archive_emit_constants':
                with recorder.span(phase):
                    in_process.set()
                    try:
                        result['phases'][phase] = archive_emit_constants_after_trace(result, work)
                    finally:
                        in_process.clear()
                    result['completed'][phase] = 1
                if result['resource_monitor_error']:
                    raise RuntimeError(result['resource_monitor_error'])
                continue
            command, timeout = _launch_worker(args, system, phase, work / 'program.ptdsl', work)
            with recorder.span(phase, command=command):
                in_process.set()
                try:
                    obs = run_process(command, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=out / (phase + '.log'),
                                      resources=out / (phase + '-resources.jsonl'), timeout=timeout,
                                      mem_gib=args.memory_gib, fsize_gib=8, env=env)
                finally:
                    in_process.clear()
                result['processes'].append(dict(phase=phase, argv=command, observation=obs))
                phase_file = work / (phase + '-result.json')
                if phase_file.is_file():
                    result['phases'][phase] = _read(phase_file)
                phase_record = result['phases'].get(phase, {})
                if strict_source and phase_record and (
                        phase_record.get('model') != factory
                        or Path(phase_record.get('source_model_info', '')).resolve() != args.model_info.resolve()
                        or Path(phase_record.get('source_program', '')).resolve() != (work/'program.ptdsl').resolve()):
                    raise ValueError('Worker phase returned a different original model or source program')
                if (obs['exit_code'] or obs['timed_out'] or obs['signal'] is not None or obs['remaining_pids']
                        or result['phases'].get(phase, {}).get('status') != 'ok'):
                    raise RuntimeError('Compiler comparison stopped at ' + phase)
                result['completed'][phase] = 1
            if result['resource_monitor_error']:
                raise RuntimeError(result['resource_monitor_error'])
        result.update(status='ok', compiled_bootstraps=result['phases']['compile']['compiled_bootstraps'],
                      predicted_latency_us=result['phases']['compile']['predicted_latency_us'])
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}'[:4000])
    finally:
        for signum in old_handlers:
            signal.signal(signum, signal.SIG_IGN)
        stop.set()
        if watcher.is_alive():
            watcher.join(timeout=3)
        result['recorded_elapsed_ns'] = time.perf_counter_ns() - started
        result['complete_batch_elapsed_seconds'] = None
        events = read_events(out / 'events.jsonl')
        for phase in result['attempted']:
            parent = next((event['event_id'] for event in events if event.get('stage_id') == phase and event.get('type') == 'begin'), None)
            for event in read_events(work / (phase + '-events.jsonl')):
                if event.get('parent_id') is None and event.get('type') in ('begin', 'end'):
                    event['parent_id'] = parent
                events.append(event)
        result['timing'] = summarize_events(events, termination_observed_ns=time.perf_counter_ns())
        result['metrics'] = {key: result[key] for key in ('status', 'system', 'attempted', 'completed', 'compiled_bootstraps',
                           'predicted_latency_us', 'CKKS_latency', 'accuracy_acceptance', 'extra_integrity_checks',
                           'llm_calls', 'keygen_runs', 'real_ckks_runs', 'recorded_elapsed_ns')}
        result['metrics']['completion_path'] = result['completion_path']
        write_json(out / 'progress.json', result)
        for _ in range(12):
            write_json(out / 'result.json', result)
            total = _size(out)
            if total == result['artifact_bytes']:
                break
            result['artifact_bytes'] = total
        row = dict(run_id=result['run_id'], round=ROUND, model=selected_model(args), mode=system, git_rev=args.git_rev, metrics=result['metrics'],
                   artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
        with (paths.RESULTS_DIR / 'registry.jsonl').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            fcntl.flock(stream, fcntl.LOCK_UN)
        result['completion'] = completion_marker(out, started_ns=started, status=result['status'],
                                               timing=result['timing'], required_artifact_bytes=result['artifact_bytes'])
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return result


def outer(args):
    from .. import paths
    if args.out.exists():
        raise ValueError('Preserve earlier matrix evidence')
    args.out.mkdir(parents=True)
    # Read-only git metadata is small; its actual command and resource observation
    # are preserved with the queue rather than hidden in a result field.
    obs = run_process(['git', 'rev-parse', 'HEAD'], cwd=paths.REPO, log=args.out / 'git-revision.log',
                      resources=args.out / 'git-revision-resources.jsonl', timeout=15, mem_gib=.5)
    args.git_rev = (args.out / 'git-revision.log').read_text().strip() if obs['exit_code'] == 0 else None
    results = []
    for system in args.systems:
        record = run_case(args, system, args.out / system)
        results.append(dict(system=system, status=record['status'], result=str(args.out / system / 'result.json'),
                            completion=record.get('completion_path'),
                            complete_case_elapsed_seconds=(record.get('completion') or {}).get('complete_case_elapsed_seconds')))
        write_json(args.out / 'queue-result.json', dict(round=ROUND, model=selected_model(args), results=results,
                   status='ok' if all(row['status'] == 'ok' for row in results) else 'partial', git_observation=obs,
                   CKKS_latency=None, accuracy_acceptance=None))
        if record.get('interrupted_signal') is not None and record.get('resource_monitor_error') is None:
            break
    print(json.dumps(dict(output=str(args.out), cases=results)))
    return 0 if all(row['status'] == 'ok' for row in results) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-phase', choices=('placement', 'emit', 'trace', 'compile'))
    parser.add_argument('--system', choices=SYSTEMS)
    parser.add_argument('--model-factory', choices=MODEL_FACTORIES, default=MODEL)
    for name in ('out', 'work-dir', 'model-info', 'program', 'rule-result', 'agent-result', 'agent-audit',
                 'metadata-template', 'profile', 'frontend'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--systems', nargs='+', choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument('--compile-seconds', type=int, default=7200)
    parser.add_argument('--trace-seconds', type=int, default=1800)
    parser.add_argument('--memory-gib', type=int, default=6)
    parser.add_argument('--case-gib', type=int, default=8)
    parser.add_argument('--debug-snapshots', action='store_true', help='Explicitly retain large intermediate debug IR snapshots')
    parser.add_argument('--archive-emit-constants-after-trace', action='store_true',
                        help='Reversibly archive the generated NPZ after trace and before native compilation')
    args = parser.parse_args(argv)
    if not 1 <= args.compile_seconds <= 7200 or not 1 <= args.trace_seconds <= 1800 or not 4 <= args.memory_gib <= 12:
        parser.error('Use compile<=7200s, trace<=1800s and 4..12 GiB within the 16 GiB host')
    if not 2 <= args.case_gib <= MAX_CASE_GIB or len(set(args.systems)) != len(args.systems):
        parser.error('Use a distinct system list and a 2..64 GiB explicit per-case disk allowance')
    if args.worker_phase:
        if any(getattr(args, key) is None for key in ('system', 'work_dir', 'model_info', 'program', 'metadata_template', 'profile', 'frontend')):
            parser.error('Inner phases require all source and toolchain paths explicitly')
        return worker(args)
    from .. import paths
    args.out = args.out or paths.RESULTS_DIR / 'local-compile-matrix-r29-20261116'
    short = args.model_factory.split(':', 1)[1]
    original_run = {'resnet20_silu':'resnet20-silu', 'vgg16_silu_avgpool':'vgg16-silu-avgpool',
                    'alexnet_silu_avgpool':'alexnet-silu-avgpool'}[short]
    args.model_info = args.model_info or paths.RESULTS_DIR / ('r13-common-'+original_run+'-20261115/worker/model_info/model_info.json')
    rule_run = 'r17-margin-heldout-20261115' if short=='resnet20_silu' else 'r17a-margin-heldout-20261115'
    args.rule_result = args.rule_result or paths.RESULTS_DIR / rule_run / short / 'result.json'
    args.metadata_template = args.metadata_template or paths.RESULTS_DIR / 'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json'
    args.profile = args.profile or paths.PROFILES_DIR / 'qwen48/compiler-config.json'
    args.frontend = args.frontend or paths.DACAPO_ROOT / 'python/hecate/hecate/expr.py'
    for name in ('out','model_info','rule_result','agent_result','agent_audit','metadata_template','profile','frontend'):
        value = getattr(args,name,None)
        if value is not None:
            setattr(args,name,value.resolve())
    return outer(args)


if __name__ == '__main__':
    raise SystemExit(main())
