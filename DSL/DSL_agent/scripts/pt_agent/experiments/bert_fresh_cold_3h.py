"""A fresh original BERT/SST2 Agent pipeline with one inclusive three-hour clock.

The original prepare, v11 steps, C2, calibration mathematics and compiler are
delegated. A wall budget may end search at a committed evaluate boundary. No
report command is exposed. Compiler outcomes stay outside the Agent process.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

from .. import paths
from . import bert_sst2 as original
from . import bert_serial_continuation as serial
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events

MODULE = 'pt_agent.experiments.bert_fresh_cold_3h'
ROUND = 'BERT-FRESH-COLD-3H'
MODEL = original.MODEL
MODE = 'fresh-original-model-v11-three-hour-wall-budget-c2-compile-no-report'
GIB = 1024**3
TOTAL_SECONDS = 10800
SEARCH_CEILING = 9000
TAIL_RESERVE = 1800
CLOSURE_RESERVE = 30
CASE_BYTES = 64*GIB
REGISTRY_FIELDS = serial.REGISTRY_FIELDS


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    serial.durable_json(path, value)


def _clean(observed):
    return (observed.get('exit_code') == 0 and observed.get('timed_out') is False
            and observed.get('signal') is None and observed.get('signals_sent') == []
            and observed.get('remaining_pids') == [] and not observed.get('cleanup_unknown'))


def state_summary(case):
    path = Path(case)/'session/SESSION.json'
    if not path.is_file():
        return None
    state = read(path)
    if any(state.get(key) != value for key, value in serial.STATE_POLICY.items()):
        raise ValueError('Keep every original v11 session policy and max_iters10')
    return dict(phase=state.get('phase'), iteration=state.get('iteration'),
                calls=state.get('calls'), failed=state.get('failed'),
                pending=state.get('pending') is not None, pool_size=len(state.get('pool', [])))


def stop_search(case, *, elapsed_seconds):
    """Change only phase at a complete same-run boundary, before one original C2."""
    path = Path(case)/'session/SESSION.json'
    state = read(path)
    if any(state.get(key) != value for key, value in serial.STATE_POLICY.items()):
        raise ValueError('Original BERT policy changed')
    if (state.get('phase') != 'evaluate' or state.get('pending') is not None
            or state.get('failed') is not None or not state.get('pool')
            or not state.get('records') or not state.get('history')
            or type(state.get('calls')) is not int or state['calls'] < 1
            or any((Path(case)/'session'/name).exists() for name in
                   ('AGENT.json', 'FROZEN.json', 'REPORT_ACCESS.json', 'program.ptdsl'))):
        raise ValueError('Wall stop requires one complete fresh evaluate boundary and candidate pool')
    before = copy.deepcopy(state)
    save(Path(case)/'budget-stop-before-session.json', state)
    state['phase'] = 'confirm'
    save(path, state)
    if read(path) != dict(before, phase='confirm'):
        raise ValueError('Budget stop changed more than original phase')
    record = dict(policy='availability-first-wall-stop-at-first-complete-feasible-pool-boundary',
                  elapsed_seconds=elapsed_seconds, only_session_change='phase:evaluate->confirm',
                  completed_calls=before['calls'], max_iters_preserved=before['max_iters'],
                  complete_pool_preserved=True, provider_budget_exhaustion_used=False,
                  abandoned_partial_steps=0, inherited_warm_search=False)
    save(Path(case)/'budget-stop.json', record)
    return record


def source_gate(args):
    source = original.source_preflight(args)
    if args.profile.resolve() != (paths.PROFILES_DIR/'qwen48/compiler-config.json').resolve():
        raise ValueError('Use the unchanged current qwen48 profile')
    if args.frontend.resolve() != (paths.DACAPO_ROOT/'python/hecate/hecate/expr.py').resolve():
        raise ValueError('Use the unchanged trusted frontend')
    if args.out.parent != paths.RESULTS_DIR.resolve() or args.out.exists() or args.checkpoint_out.exists():
        raise ValueError('Fresh cold output and checkpoint case must both be new')
    if args.checkpoint_out.parent != paths.RESULTS_DIR.resolve() or args.checkpoint_out == args.out:
        raise ValueError('A separate fresh checkpoint case is required')
    from .r29_local_compile_matrix import compiler_argv
    compiler_argv(read(args.metadata_template), 'pars', binary='/compiler', source='/input',
                  profile=args.profile, output='/lowered')
    return dict(source, profile=str(args.profile), profile_metadata=read(args.profile),
                frontend=str(args.frontend), metadata_template=str(args.metadata_template),
                warm_model_or_calibration_reuse=False, warm_session_or_candidate_reuse=False,
                total_wall_seconds=TOTAL_SECONDS, search_absolute_ceiling_seconds=SEARCH_CEILING,
                confirmation_and_compile_reserve_seconds=TAIL_RESERVE,
                compiler_build=str(paths.HECATE_BUILD), report_samples=0)


def _numeric_args(args, phase, work):
    argv = ['--phase', phase, '--model', str(args.model), '--inputs', str(args.inputs),
            '--model-dir', str(args.out/'model'), '--session-dir', str(args.out/'session'),
            '--work-dir', str(work), '--ledger', str(args.ledger), '--allowance', str(args.allowance),
            '--data-python', str(args.data_python)]
    return argv


def command(args, phase, work, remaining):
    if phase in ('prepare', 'step', 'confirm'):
        argv = [sys.executable, '-B', '-m', 'pt_agent.experiments.bert_sst2_recorded',
                '--delegate-worker', '--step-seconds', '10800', '--metadata-template', str(args.metadata_template)]
        argv += _numeric_args(args, 'prepare' if phase == 'prepare' else 'step', work)
    elif phase == 'checkpoint':
        return [sys.executable, '-B', '-m', 'pt_agent.experiments.bert_checkpoint_audit',
                '--model', str(args.model), '--prepare-result', str(args.out/'phases/001-prepare/worker_result.json'),
                '--model-info', str(args.out/'model/model_info.json'), '--out', str(args.checkpoint_out),
                '--intake-python', str(args.intake_python)]
    else:
        argv = [sys.executable, '-B', '-m', MODULE, '--worker-phase', phase,
                '--source-case', str(args.out), '--checkpoint-audit', str(args.checkpoint_out/'result.json'),
                '--work-dir', str(work), '--metadata-template', str(args.metadata_template),
                '--profile', str(args.profile), '--frontend', str(args.frontend),
                '--remaining-seconds', str(max(1, int(remaining)))]
        if phase != 'math':
            argv += ['--math-result', str(args.out/'math/math-result.json'),
                     '--model-info', str(args.out/'model/model_info.json'),
                     '--program', str(args.out/'session/program.ptdsl')]
    # Pure intake receives only explicit paths and has no outer registry/config access.
    return [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds',
            str(max(1, int(remaining))), '--python', str(args.intake_python), '--', *argv[1:]]


def worker(args):
    from . import bert_compile_first_deferred_report as deferred
    if args.worker_phase == 'math':
        return deferred.math_worker(args)
    if read(args.math_result).get('compiler_inputs')!=deferred.compiler_metadata(args):
        raise ValueError('Original compiler inputs differ from accepted fresh calibration mathematics')
    with deferred.original.native_metadata_commands(read(args.metadata_template)):
        namespace = deferred.compile_namespace(args.remaining_seconds)
        namespace['CASE_BYTES'] = CASE_BYTES
        # Original worker source/plan/Count/Emit/trace/native implementation stays intact.
        return namespace['worker'](args)


def _budget(args):
    size = artifact_bytes(args.out)+artifact_bytes(args.checkpoint_out)
    if size+GIB//1024 > CASE_BYTES or shutil.disk_usage(args.out).free < 10*GIB:
        raise RuntimeError('Fresh BERT shared64GiB case or10GiB disk reserve exhausted')
    return size


def _registry(args, result):
    from ..registry import git_rev
    row = dict(run_id=args.out.name, round=ROUND, model=MODEL, mode=MODE, git_rev=git_rev(),
               metrics=result['metrics'], artifacts_dir=str(args.out), ts=datetime.now(timezone.utc).isoformat())
    registry = paths.RESULTS_DIR/'registry.jsonl'
    with registry.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        if any(json.loads(line).get('run_id') == args.out.name for line in stream if line.strip()):
            raise ValueError('Preserve the unique fresh cold registry identity')
        stream.write(json.dumps(row, allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def _read_worker(args, phase, work):
    if phase == 'checkpoint':
        return read(args.checkpoint_out/'result.json')
    target = work/('worker_result.json' if phase in ('prepare', 'step', 'confirm') else phase+'-result.json')
    value=read(target) if target.is_file() else None
    # Original _child persists before raising, so failed native calls also have
    # their actual exit/timeout observation even when assignment never returned.
    native=work/(phase+'-process.json')
    if phase in ('trace','compile')and native.is_file():
        value=dict(value or {},process=read(native),native_process_observation_source=str(native))
    return value


def native_artifacts(compiled):
    """Inspect typed native metadata and stat payload paths, without opening blobs."""
    from .r31_lattigo_runtime_plan import content_filename
    plan_path=Path(compiled['runtime_plan']);plan=read(plan_path)
    bundle=plan_path.parent/(plan_path.name.removesuffix('.runtime-plan.json')+'.bundle')
    manifest=read(bundle/'manifest.json')
    if (plan.get('format_version')!=1 or plan.get('finalization')!=[]
            or not plan.get('external_inputs') or not plan.get('final_outputs')
            or (plan.get('plaintext_bundle') or {}).get('id')!=manifest.get('bundle_id')
            or (manifest.get('bundle_format_version'),manifest.get('version'))!=(1,1)):
        raise ValueError('Native plan/bundle metadata contract is incomplete')
    values={row['id']:row for row in plan['values']}
    if len(values)!=len(plan['values']):raise ValueError('Native SSA value IDs must be unique')
    for value in values.values():
        if (value.get('context')!='qwen-bootstrap48' or value.get('ntt')is not True
                or value.get('place')!={'kind':'host','rank':0}
                or type(value.get('level'))is not int or not 0<=value['level']<=8
                or value.get('components')not in ((2,3) if value.get('kind')=='ciphertext' else (1,))):
            raise ValueError('Native value metadata differs from the original qwen48 contract')
    defined=set(plan['external_inputs']);blobs={row['content']:row for row in manifest['blobs']}
    if len(blobs)!=len(manifest['blobs']):raise ValueError('Bundle locators must be unique')
    for content,row in blobs.items():
        path=bundle/'data'/content_filename(content)
        if not path.is_file() or path.stat().st_size!=row['byte_length']:
            raise ValueError('Native constant payload resource metadata is incomplete')
    boots=0
    for row in plan['initialization']+plan['execution']:
        if any(name not in defined for name in row.get('inputs',[])):
            raise ValueError('Native SSA input appears before its definition')
        if row.get('payload',{}).get('kind')=='bundle' and row['payload'].get('content')not in blobs:
            raise ValueError('Native encode references an absent bundle locator')
        outputs=([row['output']] if row.get('output')is not None else row.get('outputs',[]))
        if any(name in defined or name not in values for name in outputs):
            raise ValueError('Native output definition is absent or repeated')
        if row.get('op')=='boot':
            boots+=1
            if any(values[name].get('level')!=5 or values[name].get('scale_log2')!=43 for name in outputs):
                raise ValueError('Actual native bootstrap output contract changed')
        defined.update(outputs)
    if any(name not in defined for name in plan['final_outputs']) or boots!=compiled['compiled_bootstraps']:
        raise ValueError('Native final outputs or actual bootstrap inventory differ')
    return dict(runtime_plan=str(plan_path),bundle=str(bundle),manifest=str(bundle/'manifest.json'),
        payload_count=len(blobs),payload_bytes=sum(row['byte_length']for row in blobs.values()),
        blob_payload_reads=0,integrity_checks=0,typed_ssa_and_bundle_resource_metadata_accepted=True)


def three_hour_observation(result,completion,started,now):
    within=now-started<=TOTAL_SECONDS*10**9
    usable=bool(result['compile_accepted'] and completion['required_artifacts_persisted']is True
        and completion['registry_persisted']is True and completion['ended_ns']-started<=TOTAL_SECONDS*10**9 and within)
    return dict(round=ROUND,model=MODEL,mode=MODE,started_ns=started,observed_ns=now,
        elapsed_seconds=(now-started)/1e9,budget_seconds=10800,compile_accepted=result['compile_accepted'],
        native_completed_ns=result.get('native_completed_ns'),
        required_artifacts_and_registry_completed_ns=completion['ended_ns'],
        completion_elapsed_seconds=completion['complete_case_elapsed_seconds'],
        persisted_pipeline_within_three_hours=usable,
        persisted_pipeline_within_two_hours=bool(usable and now-started<=7200*10**9),
        two_hour_observation_scope='Actual completion and observation within the first7200s of this separately allocated10800s case',
        status='ok'if usable else'error',
        usable_definition='Original C2 accepted; strict calibration3/9 mathematics; full typed RuntimePlan and bundle resource metadata accepted; result/registry/completion persisted within10800s from fresh original-model/input entry',
        report_samples=0,keygen_runs=0,ckks_runs=0,
        scope='One actual fresh intake through required persistence; final observation JSON write follows the recorded endpoint')


def execute(args):
    from ..autonomous_budget import call_records
    from ..registry import git_rev
    from .bert_frozen_compile import physical_memory_observation, _available_ram_bytes, _latest_rss_sample
    # The one cold clock begins before original model/input source preflight.
    started = time.perf_counter_ns(); deadline = time.monotonic()+TOTAL_SECONDS
    source = source_gate(args)
    args.out.mkdir();(args.out/'phases').mkdir();(args.out/'tmp').mkdir()
    recorder = StageRecorder(args.out/'events.jsonl')
    stop = threading.Event(); failures = []; active = [None]
    result = dict(round=ROUND, model=MODEL, mode=MODE, status='running', error=None, source=source,git_rev=git_rev(),
        attempted=dict(prepare=0,checkpoint=0,step=0,confirm=0,math=0,emit=0,trace=0,compile=0),
        completed=dict(prepare=0,checkpoint=0,step=0,confirm=0,math=0,emit=0,trace=0,compile=0),
        processes=[], phases={}, frozen=False, compile_accepted=False, compiled_bootstraps=None,
        runtime_plan=None, prediction=None, generation_e2e_s=None, report_samples=0,
        keygen_runs=0, ckks_runs=0, downloads=0, installations=0, integrity_checks=0,
        three_hour_record=str(args.out/'three-hour-record.json'),three_hour_record_authoritative=True,termination=None,
        wall_policy=dict(total_seconds=10800,search_absolute_ceiling_seconds=9000,
                         search_stop='first complete feasible-pool evaluate boundary',tail_reserve_seconds=1800,
                         rollback_partial_steps=False,max_iters=10,tuner_budget_s=2400),
        limits=dict(address_space_gib=24,physical_rss_gib=14,minimum_available_gib=1.25,
                    file_gib=16,shared_case_gib=64,minimum_free_disk_gib=10))
    save(args.out/'result.json',result)
    for module,name in ((Path(__file__),'fresh-driver-source.py'),(Path(original.__file__),'original-bert-outer.py'),
                        (Path(__file__).with_name('bert_sst2_worker.py'),'original-bert-worker.py')):
        shutil.copyfile(module,args.out/name)
    shutil.copyfile(args.metadata_template,args.out/'compile-metadata-template.json')
    before = len(call_records(args.ledger)); result['api_before'] = before
    handlers = {sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    def interrupt(sig,frame):
        if result.get('termination') is None:result['termination']=dict(cause='incoming_signal',signal=sig)
        raise InterruptedError('Fresh BERT interrupted '+str(sig))
    for sig in handlers:signal.signal(sig,interrupt)
    def monitor():
        while not stop.wait(.2):
            try:
                _budget(args)
                if time.monotonic()>=deadline-CLOSURE_RESERVE:raise TimeoutError('10800second inclusive closure reserve reached')
                observed=physical_memory_observation(_latest_rss_sample(active[0]) if active[0] else None,
                    _available_ram_bytes(),rss_gib=14,minimum_available_gib=1.25)
                if observed['stop']:
                    result['memory_guard_observation']=observed
                    raise MemoryError('Fresh BERT physical memory guard')
            except BaseException as exc:
                failure=dict(cause='wall_budget'if isinstance(exc,TimeoutError)else'physical_memory'if isinstance(exc,MemoryError)else'resource_guard',
                             error=type(exc).__name__+': '+str(exc))
                failures.append(failure);result['termination']=failure
                save(args.out/'guard-stop.json',failure);os.kill(os.getpid(),signal.SIGTERM);return
    watcher=threading.Thread(target=monitor,daemon=True)
    def phase(name,cap):
        remaining=min(cap,deadline-time.monotonic()-CLOSURE_RESERVE)
        if remaining<=0:raise TimeoutError('No inclusive cold allowance remains')
        _budget(args)
        if name in ('prepare','step','confirm'):
            work=args.out/'phases'/(str(len(list((args.out/'phases').iterdir()))+1).zfill(3)+'-'+('step' if name=='confirm' else name))
            work.mkdir();(work/'tmp').mkdir()
        else:
            work=args.out/('math' if name=='math' else 'compile-work' if name in ('emit','trace','compile') else 'checkpoint-command')
            if name!='math':work.mkdir(exist_ok=name in ('emit','trace','compile'))
        argv=command(args,name,work,remaining)
        row=dict(phase=name,argv=argv,worker_result=None);result['processes'].append(row)
        result['active_phase']=name;result['attempted'][name]+=1;save(args.out/'result.json',result)
        api_before=len(call_records(args.ledger))
        try:
            with recorder.span(name):
                active[0]=args.out/(str(len(result['processes'])).zfill(3)+'-'+name+'-resources.jsonl')
                row['resources']=str(active[0])
                row['observation']=run_process(argv,cwd=paths.REPO/'DSL/DSL_agent/scripts',
                    log=args.out/(str(len(result['processes'])).zfill(3)+'-'+name+'-outer.log'),resources=active[0],timeout=remaining,
                    mem_gib=24,fsize_gib=16,env=dict(original.child_environment(args,work),TMPDIR=str(args.out/'tmp')))
                active[0]=None
                row['worker_result']=_read_worker(args,name,work)
                value=row['worker_result'] or {}
                if failures or not _clean(row['observation']) or value.get('status')!='ok':
                    raise RuntimeError('Fresh original '+name+' failed or did not close cleanly')
                if name in ('prepare','step','confirm'):original.check_worker('prepare' if name=='prepare' else 'step',value)
                if name=='checkpoint' and value.get('checkpoint_semantic_coverage_passed')is not True:
                    raise ValueError('Fresh trained checkpoint semantic audit failed')
                if name=='math' and value.get('math_passed')is not True:raise ValueError('Calibration3 mathematics incomplete')
                result['phases'][name]=value;result['completed'][name]+=1
        finally:
            api_after=len(call_records(args.ledger));row['api_before']=api_before;row['api_after']=api_after
            if api_after>api_before:
                row['api_registration']=original.account_api_calls(args.ledger,paths.RESULTS_DIR/'registry.jsonl',
                    work,before=api_before,after=api_after,git_rev=result.get('git_rev','unknown'))
            save(args.out/'result.json',result)
        return value
    try:
        with serial.serial_lease(paths.RESULTS_DIR/'bert-fresh-cold-3h.lock'),recorder.span('fresh_cold_complete'):
            watcher.start()
            phase('prepare',min(3600,SEARCH_CEILING-(time.monotonic()-(deadline-TOTAL_SECONDS))))
            phase('checkpoint',420)
            for _ in range(25):
                state=state_summary(args.out)
                elapsed=(time.perf_counter_ns()-started)/1e9
                if state and state['phase']=='done':break
                if state and state['failed'] is not None:raise RuntimeError('Original Agent session failed: '+str(state['failed']))
                if state and state['phase']=='evaluate' and not state['pending'] and state['pool_size']:
                    result['budget_stop']=stop_search(args.out,elapsed_seconds=elapsed)
                    state=state_summary(args.out)
                if state and state['phase']=='confirm':
                    if result['attempted']['confirm']:raise ValueError('Original C2 may dispatch only once')
                    phase('confirm',min(900,deadline-time.monotonic()-900))
                    continue
                search_left=SEARCH_CEILING-elapsed
                if search_left<=0:raise TimeoutError('Search ended without a complete confirmation boundary')
                phase('step',search_left)
            else:raise RuntimeError('Original finite25step orchestration exhausted')
            frozen=serial.frozen_state(args.out);result['frozen']=frozen['frozen'];result['freeze']=frozen
            if not frozen['frozen']:raise ValueError('Original C2 rejected all candidates')
            confirmation=(read(args.out/'session/AGENT.json').get('confirm')or{})
            selected=confirmation.get('selected')or{}
            matches=[row for row in confirmation.get('evals',[])if
                (row.get('iteration'),row.get('origin'))==(selected.get('iteration'),selected.get('origin'))]
            if len(matches)!=1 or matches[0].get('passed')is not True:
                raise ValueError('One current original C2 selected candidate acceptance is required')
            result['confirmation_quality']={key:matches[0].get(key)for key in
                ('iteration','origin','passed','top1','exact_top1','total','agree','acc_drop_pp')}
            result['search_boundary']=state_summary(args.out)
            phase('math',min(900,deadline-time.monotonic()-CLOSURE_RESERVE))
            for name in ('emit','trace','compile'):
                phase(name,deadline-time.monotonic()-CLOSURE_RESERVE)
            compiled=result['phases']['compile']
            if not compiled.get('runtime_plan') or not Path(compiled['runtime_plan']).is_file():
                raise ValueError('Actual native RuntimePlan is required')
            result['native_artifacts']=native_artifacts(compiled)
            result.update(status='ok',compile_accepted=True,runtime_plan=compiled['runtime_plan'],
                compiled_bootstraps=compiled['compiled_bootstraps'],prediction=compiled.get('prediction'),
                native_completed_ns=time.perf_counter_ns())
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc),compile_accepted=False)
    finally:
        # Keep cancellation masked through child reap, registry and completion.
        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL,0)
        stop.set()
        if watcher.is_alive():watcher.join(timeout=3)
        stopped=not watcher.is_alive();result['monitor_stopped']=stopped
        try:
            if not stopped or failures or time.monotonic()>=deadline:
                result.update(status='error',compile_accepted=False,error=result.get('error') or 'Inclusive wall or monitor closure failed')
            try:_budget(args)
            except Exception as exc:result.update(status='error',compile_accepted=False,error=str(exc))
            result['api_after']=len(call_records(args.ledger));result['api_calls']=result['api_after']-before
            result['timing']=summarize_events(read_events(recorder.path))
            result['metrics']={key:result.get(key) for key in ('status','attempted','completed','compile_accepted',
                'compiled_bootstraps','frozen','api_calls','report_samples','keygen_runs','ckks_runs','termination','three_hour_record')}
            result['recorded_before_persistence_seconds']=(time.perf_counter_ns()-started)/1e9
            save(args.out/'result.json',result);_registry(args,result)
            completion=persist_case_completion(args.out,round_label=ROUND,started_ns=started,status=result['status'],
                timing=result['timing'],required_artifact_bytes=artifact_bytes(args.out))
            now=time.perf_counter_ns()
            record=three_hour_observation(result,completion,started,now)
            save(args.out/'three-hour-record.json',record)
            if args.paper_runs is not None:paper_record(args,result,record)
        finally:
            if stopped:
                for sig,handler in handlers.items():signal.signal(sig,handler)
                # The previous timer is deliberately not restarted after a bounded run.
    return result


def paper_record(args,result,record):
    from .a_group_runs import append_record,empty_record
    from .a_group_profile_sum_policy import profile_sum
    success=record['persisted_pipeline_within_three_hours']
    priced=profile_sum(result.get('prediction'))
    observations=[row.get('observation') or {} for row in result['processes']]
    timed=any(row.get('timed_out') for row in observations) or record['elapsed_seconds']>=10800
    inner_timeouts=[]
    for process in result['processes']:
        observation=process.get('observation') or {};argv=process.get('argv') or []
        if (argv.count('pt_agent.tools.nixrun')==1 and argv.count('--seconds')==1
                and observation.get('exit_code')==124 and observation.get('timed_out')is False
                and observation.get('signal')is None and observation.get('signals_sent')==[]
                and observation.get('remaining_pids')==[]):
            allocated=int(argv[argv.index('--seconds')+1])
            elapsed=(observation['ended_ns']-observation['started_ns'])/1e9
            if allocated<=elapsed<allocated+15:
                inner_timeouts.append(dict(phase=process['phase'],allocated_seconds=allocated,
                                           observed_seconds=elapsed,exit_code=124))
    timed=timed or bool(inner_timeouts)
    termination=result.get('termination')or{}
    native_rows=[row.get('worker_result')or{}for row in result['processes']if row['phase']=='compile']
    compile_row=result['phases'].get('compile') or (native_rows[-1]if native_rows else{})
    native=compile_row.get('process') or {}
    timed=timed or native.get('timed_out')is True
    native_failed=type(native.get('exit_code'))is int and native['exit_code']!=0
    status=('ok'if result.get('status')=='ok'and result.get('compile_accepted')else'timeout'if timed or termination.get('cause')=='wall_budget'
        else'resource_limit'if termination.get('cause')in('physical_memory','resource_guard')
        else'compiler_crash'if native_failed else'driver_error')
    quality=result.get('confirmation_quality')or{}
    stages=Counter()
    for span in result['timing']['spans']:
        if span.get('exclusive_ns')is not None:stages[span['stage_id']]+=span['exclusive_ns']/1e9
    residual=record['completion_elapsed_seconds']-sum(stages.values())
    if result['timing']['attribution_ready']and residual>=0:stages['persistence_and_other']=residual
    sampled_peak=None
    for process in result['processes']:
        resources=Path(process['resources'])if process.get('resources')else None
        if resources is not None and resources.is_file():
            with resources.open()as stream:
                for line in stream:
                    sample=json.loads(line)
                    rows=sample.get('processes')
                    if isinstance(rows,list):
                        rss=sum(row.get('vmrss_kib',0)*1024 for row in rows)
                        sampled_peak=max(sampled_peak or 0,rss)
    row=empty_record(run_id=args.out.name,task='P0-A-BERT-COLD-3H',model='bert_tiny',system='arranger',
        param_group='ours-N16-boot5',status=status,status_detail=result.get('error'),budget_s=10800,
        compile_s=(native['ended_ns']-native['started_ns'])/1e9 if native.get('ended_ns')and native.get('started_ns')else None,
        boots_compiled=result.get('compiled_bootstraps'),
        exec_ops=compile_row.get('execution_steps'),pred_latency_s=priced['pred_latency_s'],
        pred_table=str(args.profile),generation_e2e_s=record['completion_elapsed_seconds'] if result.get('compile_accepted') else None,
        stages_s=dict(stages),peak_rss_gib=sampled_peak/GIB if sampled_peak is not None else None,
        api_calls=result['api_calls'],source=str(args.out/'result.json'),finished_utc=datetime.now(timezone.utc).isoformat(),
        three_hour_record=str(args.out/'three-hour-record.json'),report_samples=0,quality_source='original C2 only; report deferred',
        pred_scope=priced['source_scope'],known_cost_partial_s=priced['known_cost_partial_s'],
        usable_definition=record['usable_definition'],search_stop_policy=result['wall_policy']['search_stop'],
        quality=dict(metric='classification_top1',split='original-confirmation1000',value=quality.get('top1'),
                     original=quality.get('exact_top1'),passed=quality.get('passed'),total=quality.get('total')),
        peak_rss_scope='Maximum simultaneous sum of saved R10 processes[].vmrss_kib across serial phases; outer driver RSS recorded separately by its parent',
        stage_scope='Sum of exclusive event durations by stage; persisted endpoint residual is separate',
        stage_attribution_ready=result['timing']['attribution_ready'],stage_residual_seconds=residual,
        three_hour_status=record['status'],usable_within_three_hours=success,
        usable_within_two_hours=record['persisted_pipeline_within_two_hours'],
        inner_phase_timeouts=inner_timeouts,
        timeout_scope='inner_phase_allocation'if inner_timeouts else'native_or_complete_case'if timed else None,
        complete_10800_seconds_consumed=record['elapsed_seconds']>=10800,
        three_hour_observation_elapsed_seconds=record['elapsed_seconds'],
        post_completion_observation_seconds=record['elapsed_seconds']-record['completion_elapsed_seconds'])
    append_record(args.paper_runs,row)


def parser():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--worker-phase',choices=('math','emit','trace','compile'))
    for key in ('model','inputs','out','checkpoint-out','source-case','checkpoint-audit','work-dir','math-result','model-info','program','paper-runs'):
        cli.add_argument('--'+key,type=Path)
    cli.add_argument('--metadata-template',type=Path,required=True)
    cli.add_argument('--profile',type=Path,default=paths.PROFILES_DIR/'qwen48/compiler-config.json')
    cli.add_argument('--frontend',type=Path,default=paths.DACAPO_ROOT/'python/hecate/hecate/expr.py')
    cli.add_argument('--intake-python',type=Path,default=paths.INTAKE_PYTHON)
    cli.add_argument('--data-python',type=Path,default=paths.PAPER_DATA_PYTHON or paths.ORBIT_ROOT.parent/'paper-data-venv/bin/python')
    cli.add_argument('--ledger',type=Path,default=paths.RESULTS_DIR/'llm_calls.jsonl')
    cli.add_argument('--allowance',type=Path,default=paths.RESULTS_DIR/'autonomous-api-allowance.json')
    cli.add_argument('--remaining-seconds',type=int,default=10800)
    return cli


def main(argv=None):
    cli=parser();args=cli.parse_args(argv)
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().absolute() if key in ('intake_python','data_python') else value.expanduser().resolve())
    required=('source_case','checkpoint_audit','work_dir') if args.worker_phase else ('model','inputs','out','checkpoint_out')
    if any(getattr(args,key) is None for key in required):cli.error('Explicit original and fresh paths are required: '+','.join(required))
    result=worker(args) if args.worker_phase else execute(args)
    print(json.dumps({key:result.get(key) for key in ('status','error','compile_accepted','compiled_bootstraps')}),flush=True)
    accepted=(result['status']=='ok')
    if not args.worker_phase:accepted=accepted and read(args.out/'three-hour-record.json')['persisted_pipeline_within_three_hours']
    return 0 if accepted else 1


if __name__=='__main__':raise SystemExit(main())
