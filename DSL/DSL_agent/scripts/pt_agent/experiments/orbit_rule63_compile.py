"""One source-bound Orbit Rule63 placement/serialization attempt.

The closed A2b model/calibration/Rule63 proof is reused. A new worker performs
only the own format adapter and original Orbit parser/ILP/decoder/serializer.
Its measured interval and the derived shared-generation clock stay separate.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import FunctionType

from .. import paths
from . import a5_a6_shared_rule63_compile as shared
from . import cold_rule63_whole_model as cold
from . import a_group_future_closed_archive as closed
from .r10_paper_metrics import StageRecorder, write_json

MODULE = 'pt_agent.experiments.orbit_rule63_compile'
ROUND = 'S1-ORBIT-SHARED-RULE63'
MODE = 'same-rule63-qwen48-fixed1to5-orbit-placement-serialization'
MODELS = dict(resnet20='resnet20_silu', vgg16='vgg16_silu_avgpool', alexnet='alexnet_silu_avgpool')
RESNET_DEADLINE = datetime(2026, 11, 17, 11, 35, tzinfo=timezone.utc)
GIB = 1024 ** 3
require, read, raw_stat = shared.require, shared.read, shared.raw_stat


def regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path,
            'An original regular file with an explicit canonical path is required')
    require(path.stat().st_size > 0, 'Required file is empty')
    return raw_stat(path)


def prerequisite_resnet(args):
    if args.short_model == 'resnet20':
        require(datetime.now(timezone.utc) < RESNET_DEADLINE, 'S1 ResNet admission deadline exhausted')
        return None
    case = args.resnet_case
    require(case is not None and case.parent == paths.RESULTS_DIR.resolve(),
            'Later models require an explicit successful ResNet case')
    result, completion = read(case/'result.json'), read(case/'completion.json')
    require(result.get('round') == ROUND and result.get('mode') == MODE
            and result.get('model') == MODELS['resnet20'] and result.get('status') == 'ok'
            and result.get('compile_accepted') is True and result.get('monitor_stopped') is True
            and result.get('attempted') == {'compile': 1} and result.get('completed') == {'compile': 1}
            and completion.get('round') == ROUND and completion.get('status') == 'ok'
            and completion.get('registry_persisted') is True
            and completion.get('required_artifacts_persisted') is True,
            'The source-bound ResNet compilation must be a durable accepted success')
    accepted = datetime.fromisoformat(result['accepted_utc'])
    require(accepted.tzinfo is not None and accepted <= RESNET_DEADLINE,
            'ResNet success must precede the fixed S1 deadline')
    require((case/'completion.json').stat().st_mtime_ns <= int(RESNET_DEADLINE.timestamp())*10**9,
            'Complete ResNet persistence must precede the fixed S1 deadline')
    rows = registry_rows(case.name)
    require(len(rows) == 1 and set(rows[0]) == closed.REGISTRY_FIELDS
            and rows[0]['round'] == ROUND and rows[0]['model'] == MODELS['resnet20']
            and rows[0]['mode'] == MODE and rows[0]['metrics']['status'] == 'ok'
            and Path(rows[0]['artifacts_dir']).resolve() == case,
            'ResNet predecessor requires one matching eight-field registry row')
    shared.metadata_cleanup(result)
    process_rows = result.get('processes') or []
    require(len(process_rows) == 1 and process_rows[0].get('phase') == 'compile'
            and cold.clean(process_rows[0].get('observation') or {}),
            'ResNet predecessor requires one clean actual compilation worker')
    accept_adapter((result.get('phases', {}).get('compile') or {}).get('adapter'), case/'work/compile')
    require(closed.producer_births(case) > 0, 'Closed ResNet producer births are required')
    require((result.get('source') or {}).get('source_parameter_group') == 'ours-N16-boot5'
            and read(case/'source/compiler-profile.json') == read(args.profile),
            'ResNet predecessor must use the identical complete qwen48 table')
    return dict(case=str(case), result=str(case/'result.json'), accepted_utc=result['accepted_utc'])


def registry_rows(run_id):
    rows = []
    with (paths.RESULTS_DIR/'registry.jsonl').open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('run_id') == run_id:
                    rows.append(row)
    return rows


def archive_gate(args, source, source_result, raw, archive):
    """Bind the exact closed resource producer without reopening its archive."""
    case = paths.RESULTS_DIR.resolve()/(source.name+'-raw-npz-resource-archive')
    value, completion = read(case/'result.json'), read(case/'completion.json')
    require(value.get('round') == 'A-GROUP-COLD-CLOSED-RESOURCE-ARCHIVE'
            and value.get('status') == completion.get('status') == 'ok'
            and completion.get('round') == value['round']
            and completion.get('registry_persisted') is True
            and completion.get('required_artifacts_persisted') is True
            and value.get('source_case') == str(source) and value.get('source_kind') == source.name
            and value.get('scientific_status') == 'ok' and value.get('resource_status') == 'archived_or_kept'
            and value.get('in_progress') is None and not value.get('incoming_signal'),
            'Exact source raw archive producer must be durably closed')
    shared.metadata_cleanup(value)
    rows = registry_rows(case.name)
    require(len(rows) == 1 and set(rows[0]) == closed.REGISTRY_FIELDS
            and rows[0].get('round') == value['round'] and rows[0].get('model') == source.name
            and rows[0].get('mode') == 'resource-only-reversible-gzip'
            and rows[0].get('metrics', {}).get('status') == 'ok'
            and Path(rows[0].get('artifacts_dir', '')).resolve() == case,
            'One exact closed archive producer registry row is required')
    projection = source_result['phases']['trace'].get('trace_bytes')
    require(type(projection) is int and 0 < projection <= 16*GIB,
            'Original saved raw projection must fit the finite output file allowance')
    selected = [r for r in value.get('files', []) if r.get('source') == str(raw)]
    require(len(selected) == 1 and selected[0].get('status') == 'compressed'
            and selected[0].get('archive') == str(archive) and selected[0].get('raw_bytes') == projection
            and selected[0].get('compressed_bytes') == archive.stat().st_size,
            'Exact original raw/archive producer size accounting differs')
    archive_producer_proof = None
    if getattr(args, 'archive_supervisor_case', None) is not None:
        from .archive_producer_closure import verify_archive_producer
        archive_producer_proof = verify_archive_producer(case, args.archive_supervisor_case,
            host_pid_namespace=os.readlink('/proc/self/ns/pid'))
        births = archive_producer_proof['closed_producer_births']
    else:
        births = closed.producer_births(case)
    require(births > 0, 'All saved archive producer births must be closed')
    from .completion_metrics import artifact_bytes
    used = artifact_bytes(args.out) if args.out.exists() else 0
    require(used+projection < args.case_gib*GIB
            and shutil.disk_usage(paths.RESULTS_DIR).free >= projection+10*GIB,
            'resource_limit: raw projection plus case allowance and free10GiB required')
    return dict(case=str(case), result=str(case/'result.json'), completion=str(case/'completion.json'),
        closed_producer_births=births, archive_producer_proof=archive_producer_proof,
        projected_new_raw_bytes=projection,
        decoded_size_identity_comparison=False, compressed_content_read=False)


def source_gate(args, *, created=False):
    """Closed scientific metadata and stat provenance; no model or raw read."""
    current = cold.admission(args, created=created)
    source = args.source_case
    require(source == paths.RESULTS_DIR.resolve()/('a2b-'+args.short_model+'-rule63-fresh-cold-depth-20261117'),
            'Only the exact matching original A2b Rule63 case is eligible')
    result, completion = read(source/'result.json'), read(source/'completion.json')
    require(result.get('status') == 'ok' and result.get('round') == cold.ROUND
            and result.get('mode') == cold.MODE and result.get('model') == args.model
            and result.get('run_id') == source.name and result.get('placement') == 'depth'
            and result.get('monitor_stopped') is True and not result.get('guard_error')
            and not result.get('monitor_error') and result.get('expected_bootstraps_match') is True,
            'Original complete successful A2b depth source required')
    require(completion.get('round') == cold.ROUND and completion.get('status') == 'ok'
            and completion.get('registry_persisted') is True
            and completion.get('required_artifacts_persisted') is True,
            'Original A2b result/completion must be durable')
    expected = dict.fromkeys(cold.PHASES, 1)
    require(result.get('attempted') == expected and result.get('completed') == expected
            and all(result.get('phases', {}).get(p, {}).get('status') == 'ok' for p in cold.PHASES),
            'Every original cold phase must complete exactly once')
    rows = registry_rows(source.name)
    require(len(rows) == 1 and set(rows[0]) == closed.REGISTRY_FIELDS,
            'One exact eight-field original source registry row required')
    ledger = rows[0]
    require(all(ledger.get(k) == result[k] for k in ('round', 'model', 'mode', 'git_rev'))
            and ledger.get('metrics', {}).get('status') == 'ok'
            and Path(ledger.get('artifacts_dir', '')).resolve() == source,
            'Original source registry identity differs')
    shared.metadata_cleanup(result)
    processes = result.get('processes', [])
    require([r.get('phase') for r in processes] == list(cold.PHASES)
            and all(cold.clean(r['observation']) for r in processes),
            'All six original worker observations must be cleanly closed')
    generation = result['phases']['generation']
    math_proof = generation.get('training_math') or {}
    count = generation.get('raw_Count', {}).get('stats', {})
    emitted = result['phases']['emit'].get('emit_stats') or {}
    require(generation.get('rule_options') == shared.OPTIONS
            and count.get('counts', {}).get('boot') == 0
            and all(key in count and key in emitted and count[key] == emitted[key]
                    for key in ('counts', 'ops_total', 'distinct_consts'))
            and generation.get('signature', {}).get('passed') is True
            and math_proof.get('samples') == 3 and len(math_proof.get('checks', [])) == 3
            and all(r.get('passed') is True for r in math_proof.get('checks', [])),
            'Original degree63/margin1.5/Count and training mathematics proof differ')
    require(read(source/'source/compiler-profile.json') == read(args.profile),
            'The complete original qwen48 parameters and cost table must be preserved')
    native = result['phases']['compile'].get('native_process') or {}
    require(cold.clean(native), 'Original A2b native phase must be cleanly closed')
    start, compile_start, compile_end, end = (completion.get('started_ns'), native.get('started_ns'),
        native.get('ended_ns'), completion.get('ended_ns'))
    require(all(type(v) is int for v in (start, compile_start, compile_end, end))
            and 0 <= start < compile_start < compile_end <= end
            and processes[-1]['observation']['started_ns'] <= compile_start < compile_end
            <= processes[-1]['observation']['ended_ns'], 'Original same-clock prefix must be ordered')
    raw = source/'work/emit/candidate_trace.mlir'
    archived = raw.with_name(raw.name+'.gz')
    require(not raw.is_symlink() and not archived.is_symlink(), 'Original input links are rejected')
    selected = raw if raw.is_file() else archived
    identity = regular(selected)
    require(not raw.with_name(raw.name+'.gz.partial').exists(), 'Original archive is incomplete')
    archive_proof = archive_gate(args, source, result, raw, archived) if selected == archived else None
    births = closed.producer_births(source)
    require(births > 0, 'Original saved producer births must all be closed')
    from .closed_dacapo_trace_archive import lease_observation
    lease = lease_observation()
    predecessor = prerequisite_resnet(args)
    orbit_python = args.orbit_python
    license_file = args.license_file
    require(orbit_python.is_file() and os.access(orbit_python, os.X_OK) and license_file.is_file()
            and args.orbit_root.is_dir(), 'Previously installed Orbit environment is required')
    return dict(current_metadata=current, source_model=args.model, source_case=str(source), source_result=str(source/'result.json'),
        source_completion=str(source/'completion.json'), source_input=str(selected), source_input_stat=identity,
        source_archive=archive_proof,
        source_input_kind='raw' if selected == raw else 'gzip', source_raw=str(raw),
        source_parameter_group='ours-N16-boot5', source_rule_options=dict(shared.OPTIONS),
        source_registry_run_id=source.name, closed_producer_births=births, lease=lease,
        resnet_predecessor=predecessor, shared_generation_prefix_ns=compile_start-start,
        clock_evidence=dict(clock='Original same-host perf_counter_ns', source_entry_ns=start,
            source_native_compile_started_ns=compile_start, source_native_compile_ended_ns=compile_end,
            source_case_ended_ns=end), source_payload_reads=0, source_profile=read(args.profile))


def snapshot(args, proof):
    target = args.out/'source'; target.mkdir()
    for name in ('orbit_rule63_compile', 'orbit_rule63_adapter', 'bounded_gzip_input',
                 'cold_rule63_whole_model', 'a_group_runs', 'archive_producer_closure'):
        file = Path(__file__).with_name(name+'.py')
        shutil.copyfile(file, target/file.name)
    shutil.copyfile(args.profile, target/'compiler-profile.json')
    emit = args.out/'work/emit'; emit.mkdir()
    original = Path(proof['source_input'])
    require(raw_stat(original) == proof['source_input_stat'], 'Original input stat changed after admission')
    output = emit/'candidate_trace.mlir'
    preparation_started = time.perf_counter_ns()
    if proof['source_input_kind'] == 'raw':
        output.symlink_to(original)
        restored = dict(status='linked_closed_original', archive_footer_checks=0, output_readback_checks=0)
    else:
        from .bounded_gzip_input import restore_raw_member
        def guard(extra):
            require(shutil.disk_usage(args.out).free-extra >= 10*GIB,
                    'resource_limit: restored raw requires free10GiB')
            from .completion_metrics import artifact_bytes
            require(artifact_bytes(args.out)+extra <= args.case_gib*GIB,
                    'resource_limit: restored raw case allowance exhausted')
        restored = restore_raw_member(original, output, maximum_output_bytes=16*GIB, guard=guard)
    proof.update(worker_input=str(output), worker_input_stat=raw_stat(output),
        input_preparation=restored, input_restored=proof['source_input_kind'] == 'gzip',
        input_preparation_started_ns=preparation_started, input_preparation_ended_ns=time.perf_counter_ns())
    proof['input_preparation_s'] = (proof['input_preparation_ended_ns']-preparation_started)/1e9
    write_json(target/'worker-source-proof.json', proof)


def phase_command(args, phase, remaining):
    require(phase == 'compile', 'Only one Orbit compilation worker is supported')
    seconds = min(7200, int(remaining)-180)
    require(seconds > 0, 'Finite compile allowance exhausted')
    return ([str(args.orbit_python), '-B', '-m', MODULE, '--worker', '--model', args.short_model,
        '--work-dir', str(args.out/'work/compile'), '--profile', str(args.profile),
        '--orbit-root', str(args.orbit_root), '--source-proof', str(args.out/'source/worker-source-proof.json'),
        '--native-seconds', str(seconds)], seconds+120)


def accept_adapter(value, work):
    require(isinstance(value, dict) and value.get('status') == 'ok', 'Adapter did not finish successfully')
    require(type(value.get('compiled_bootstraps')) is int and value['compiled_bootstraps'] >= 0,
            'Expanded executed bootstrap count must be an explicit nonnegative integer')
    require(type(value.get('execution_steps')) is int and value['execution_steps'] >= value['compiled_bootstraps'],
            'Expanded serialized operation count must include its bootstrap calls')
    require(all(value.get(key) is True for key in ('compile_verified', 'parameter_contract_verified',
        'bootstrap_weighted_matches_serialized', 'source_operator_constant_bijection_verified',
        'official_decoded_graph_check_passed')), 'Adapter source/parameter/count/serialization proof is incomplete')
    artifacts = value.get('artifacts') or {}
    require(isinstance(artifacts, dict) and bool(artifacts), 'Real serialized adapter artifacts are required')
    for path in artifacts.values():
        if isinstance(path, dict): path = path.get('path')
        require(isinstance(path, str) and Path(path).resolve().is_relative_to(work.resolve())
                and Path(path).is_file() and Path(path).stat().st_size > 0,
                'Adapter artifacts must remain inside this new worker directory')
    prediction = value.get('prediction') or {}
    for name in ('pred_latency_s', 'pred_latency_without_upscale_s', 'known_cost_partial_s'):
        number = prediction.get(name)
        require(number is None or type(number) in (int, float) and math.isfinite(number) and number >= 0,
                'Saved scalar prediction must be finite and nonnegative or explicit null')
    return value


def worker(args):
    """Explicit files only: no global results registry/proc/config access."""
    work = args.work_dir
    started = time.perf_counter_ns()
    result = dict(round=ROUND, mode=MODE, model=args.model, phase='compile', status='running', error=None,
        compiled_bootstraps=None, execution_steps=None, predicted_latency_us=None, prediction=None,
        runtime_plan=None, api_calls=0, CKKS_runs=0, model_forwards=0)
    try:
        proof = read(args.source_proof)
        require(args.source_proof == work.parent.parent/'source/worker-source-proof.json'
                and proof.get('source_model') == args.model
                and proof.get('source_parameter_group') == 'ours-N16-boot5'
                and proof.get('source_rule_options') == shared.OPTIONS
                and proof.get('source_profile') == read(args.profile), 'Pure worker requires the complete admitted source proof')
        source = Path(proof['worker_input'])
        require(source == work.parent/'emit/candidate_trace.mlir'
                and raw_stat(source) == proof['worker_input_stat'], 'Worker input provenance differs')
        from .orbit_rule63_adapter import compile_rule63
        result['compile_started_ns'] = time.perf_counter_ns()
        write_json(work/'phase_result.json', result)
        def checkpoint():
            if (time.perf_counter_ns()-result['compile_started_ns'])/1e9 >= args.native_seconds:
                raise TimeoutError('Orbit compiler allowance exhausted')
        output = compile_rule63(source_mlir=source, profile=read(args.profile), work_dir=work/'orbit-adapter',
            orbit_root=args.orbit_root, active_slots=16384, threads=1, checkpoint=checkpoint)
        result['compile_ended_ns'] = time.perf_counter_ns()
        result['adapter'] = output
        accept_adapter(output, work)
        result.update(status='ok', compiled_bootstraps=output['compiled_bootstraps'],
            execution_steps=output.get('execution_steps'), prediction=output.get('prediction'),
            predicted_latency_us=(output['prediction']['pred_latency_s']*1e6
                if output.get('prediction', {}).get('pred_latency_s') is not None else None))
    except BaseException as exc:
        if result.get('compile_started_ns') is not None:
            result.setdefault('compile_ended_ns', time.perf_counter_ns())
        result.update(status='error', error=type(exc).__name__+': '+str(exc)[:3000])
    finally:
        result['worker_elapsed_s'] = (time.perf_counter_ns()-started)/1e9
        write_json(work/'phase_result.json', result)
    return int(result['status'] != 'ok')


def classification(result):
    category = cold.classify(result)
    if category in ('resource_limit', 'timeout'):
        return category
    for phase in result.get('phases', {}).values():
        if ((phase or {}).get('error') or '').startswith('TimeoutError:'):
            return 'timeout'
    for row in result.get('processes', []):
        code = (row.get('observation') or {}).get('exit_code')
        if type(code) is int and code < 0:
            return 'compiler_crash'
    return category


def paper_record(args, result, completion, timing):
    from .a_group_runs import empty_record
    compiled = result['phases'].get('compile') or {}
    start, end = compiled.get('compile_started_ns'), compiled.get('compile_ended_ns')
    interrupted_clock = False
    if type(start) is int and end is None:
        processes = [r for r in result.get('processes', []) if r.get('phase') == 'compile']
        observed = (processes[-1].get('observation') or {}).get('ended_ns') if processes else None
        if type(observed) is int and observed >= start:
            end, interrupted_clock = observed, True
    duration = (end-start)/1e9 if type(start) is int and type(end) is int and end >= start else None
    source = result.get('source') or {}
    prefix = source.get('shared_generation_prefix_ns')
    prefix = prefix/1e9 if type(prefix) is int else None
    derived = prefix+duration if prefix is not None and duration is not None else None
    prediction = compiled.get('prediction') or {}
    row = empty_record(run_id=args.out.name, task='S1-Orbit', model=args.short_model, system='orbit',
        param_group='ours-N16-boot5', status=classification(result), status_detail=result.get('error'),
        budget_s=7200, compile_s=duration, peak_rss_gib=cold.peak_rss(args),
        boots_compiled=compiled.get('compiled_bootstraps'), exec_ops=compiled.get('execution_steps'),
        pred_latency_s=prediction.get('pred_latency_s'), pred_table=str(args.profile),
        generation_e2e_s=derived, stages_s=dict(shared_generation_prefix=prefix, orbit_compile=duration),
        api_calls=0, source=str(args.out/'result.json'), finished_utc=datetime.now(timezone.utc).isoformat())
    row.update(cold=False, estimated_from_shared_generation=True, derived_e2e_s=derived,
        shared_generation_prefix_s=prefix, source_case=str(args.source_case),
        input_preparation_s=source.get('input_preparation_s'),
        input_preparation_scope='Separate current-case raw-link or gzip-restoration resource clock; excluded from compile_s and derived_e2e_s',
        actual_compile_only_case_seconds=completion['complete_case_elapsed_seconds'],
        compile_scope='Own format conversion + official Orbit parser/ILP/decode/serialization in one measured worker interval',
        generation_e2e_scope='Derived original A2b entry-to-native-start prefix plus this actual Orbit compile interval',
        actual_compile_only_scope='Current input preparation, worker and terminal registry persistence',
        compile_function_completed=compiled.get('compile_ended_ns') is not None,
        compile_clock_scope=('Saved compile-start through observed worker exit after interruption'
            if interrupted_clock else 'Saved compile_rule63 function entry through its return or raised exception'),
        pred_latency_without_upscale_s=prediction.get('pred_latency_without_upscale_s'),
        prediction=prediction, prediction_scope='qwen48 serialized operator table sum excluding relinearize; no CKKS calibration',
        source_generation_clock=source.get('clock_evidence'), compile_accepted=result.get('compile_accepted', False),
        compiled_target_IR_only=True, machinecode_generated=False, backend_native_typecheck_executed=False,
        serialization_scope=(compiled.get('adapter') or {}).get('serialization_scope',
            'No successful adapter serialization observed; Hecate native typecheck and machinecode unexecuted'),
        adapter=(compiled.get('adapter') or {}).get('serialization_scope'), formal_quality=None,
        model_forwards=0, Emit_runs=0, trace_runs=0, CKKS_runs=0, quality_retested=False)
    return row


def execute(args):
    state = {}
    def admit(current, *, created=False):
        state['proof'] = source_gate(current, created=created)
        return state['proof']
    def snap(current): snapshot(current, state['proof'])
    def durable(path, value):
        if Path(path).parent == args.out and Path(path).name in ('result.json', 'progress.json'):
            value.update(cold=False, shared_generation_reused=True, original_intake_reused=True,
                training_calibration_reused=True, new_model_forwards=0, new_Emit_runs=0, new_trace_runs=0,
                source_case=str(args.source_case), compiled_target_IR_only=True, machinecode_generated=False,
                limits=dict(memory_gib=24, physical_rss_gib=14, available_ram_gib=1.25, file_gib=16,
                    case_gib=args.case_gib, free_gib=10, native_seconds=7200, total_seconds=args.seconds))
            if Path(path).name == 'result.json':
                proof = state.get('proof')
                try:
                    if proof:
                        require(raw_stat(proof['source_input']) == proof['source_input_stat'],
                                'Original source raw/archive stat changed during compile')
                        from .closed_dacapo_trace_archive import lease_observation
                        lease_observation(expected=proof['lease']['owner'])
                    compiled = value.get('phases', {}).get('compile') or {}
                    if value['status'] == 'ok':
                        accept_adapter(compiled.get('adapter'), args.out/'work/compile')
                        require(value.get('monitor_stopped') is True, 'Resource monitor must be stopped')
                        if args.short_model == 'resnet20':
                            require(datetime.now(timezone.utc) <= RESNET_DEADLINE, 'S1 ResNet deadline exhausted')
                    from .dacapo_compile_continuation import available_ram
                    from .completion_metrics import artifact_bytes
                    require(available_ram() >= 1.25*GIB and shutil.disk_usage(args.out).free >= 10*GIB
                            and artifact_bytes(args.out) <= args.case_gib*GIB,
                            'resource_limit: final available RAM/disk/case guard')
                except BaseException as exc:
                    value.update(status='error', error=type(exc).__name__+': '+str(exc))
                value['compile_accepted'] = (value['status'] == 'ok' and value.get('completed') == {'compile': 1})
                if value['compile_accepted']:
                    value.setdefault('accepted_utc', datetime.now(timezone.utc).isoformat())
        write_json(path, value)
    namespace = dict(cold.__dict__)
    namespace.update(ROUND=ROUND, MODE=MODE, PHASES=('compile',), admission=admit, snapshot=snap,
        phase_command=phase_command, paper_record=paper_record, write_json=durable)
    # Only a normal function is cloned; original signal/thread cleanup stays intact.
    run = FunctionType(cold.execute.__code__, namespace, cold.execute.__name__,
                       cold.execute.__defaults__, cold.execute.__closure__)
    old_env = os.environ.get('GRB_LICENSE_FILE')
    os.environ['GRB_LICENSE_FILE'] = str(args.license_file)
    try: return run(args)
    finally:
        if old_env is None: os.environ.pop('GRB_LICENSE_FILE', None)
        else: os.environ['GRB_LICENSE_FILE'] = old_env


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    for name in ('source-case', 'out', 'paper-runs', 'resnet-case', 'archive-supervisor-case', 'profile', 'orbit-root',
                 'orbit-python', 'license-file', 'source-proof', 'work-dir'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--seconds', type=int, default=7500)
    parser.add_argument('--native-seconds', type=int, default=7200)
    parser.add_argument('--case-gib', type=int, default=48)
    args = parser.parse_args(argv)
    args.short_model, args.model = args.model, MODELS[args.model]
    args.placement = 'orbit'
    require(180 < args.seconds <= 7500 and 0 < args.native_seconds <= 7200 and 16 <= args.case_gib <= 48,
            'Finite total7500s/native7200s/case16..48GiB required')
    if args.worker:
        require(all(getattr(args, k) is not None for k in ('source_proof', 'profile', 'work_dir', 'orbit_root')),
                'Pure worker paths must all be explicit')
    else:
        require(all(getattr(args, k) is not None for k in ('source_case', 'out', 'paper_runs')),
                'Original source, fresh out and immediate paper ledger are required')
        args.profile = args.profile or paths.PROFILES_DIR/'qwen48/compiler-config.json'
        args.orbit_root = args.orbit_root or paths.ORBIT_ROOT
        args.orbit_python = args.orbit_python or args.orbit_root.parent/'orbit-venv/bin/python'
        args.license_file = args.license_file or args.orbit_root.parent/'orbit-venv/lib/python3.11/site-packages/gurobipy/.libs/gurobi.lic'
        args.inputs = paths.WORK_RESULTS/'agent-cifar-offline-records-r1804/cifar10-records'
        args.selections = paths.WORK_RESULTS/'agent-resnet20-original-accuracy-r1805/frozen-selections.json'
        args.frontend = paths.DACAPO_ROOT/'python/hecate/hecate/expr.py'
        args.metadata_template = paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json'
        if args.short_model == 'resnet20':
            args.seconds = min(args.seconds, int((RESNET_DEADLINE-datetime.now(timezone.utc)).total_seconds()))
            require(args.seconds > 180, 'S1 ResNet deadline leaves no worker/closure allowance')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            # Preserve the existing virtual-environment executable entry path.
            setattr(args, key, value.expanduser().absolute() if key == 'orbit_python' else value.expanduser().resolve())
    return args


def main(argv=None):
    args = arguments(argv)
    if args.worker: return worker(args)
    return int(execute(args)['status'] != 'ok')


if __name__ == '__main__':
    raise SystemExit(main())
