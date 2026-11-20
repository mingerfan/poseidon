"""Detached, finite serial continuation of the retained trained BERT session.

Only existing trusted experiment entry points run. Candidate selection remains
inside the original v11/c2 session; checkpoint auditing, frozen report and
compilation never feed results back to that session.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

from .. import paths
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json, _scan
from .bert_sst2_recorded import DEFAULT_STEP_SECONDS,validate_step_seconds

ROUND='BERT-SERIAL-CONTINUATION'
MODEL='bert-tiny-sst2'
MODE='retained-v11-c2-semantic-freeze-report-compile'
MODULE='pt_agent.experiments.bert_serial_continuation'
CHECKPOINT='pt_agent.experiments.bert_checkpoint_audit'
RECORDED='pt_agent.experiments.bert_sst2_recorded'
COMPILE='pt_agent.experiments.bert_frozen_compile'
GIB=1024**3
CASE_BYTES=512*1024**2
COMPLETION_RESERVE_BYTES=1024**2
MAX_STEPS=25
OVERALL_SECONDS=86400
PHASE_LIMITS=dict(checkpoint=(480,4),step=(7440,4),report=(7440,4),compile=(11220,12))
STATE_POLICY=dict(max_iters=10,prompt_version='r8-v11',tune_rule='v8-6.25',confirm_rule='c2',
    tune_policy='all',tune_search='sweep',engine_profile='tensor',numeric_lock=True,focus_retune=True,
    split_version='sst2-native-v1',placement='program-plan',tuner=True,tuner_budget_s=2400)
ZERO_FIELDS=('driver_model_forwards','driver_sample_reads','driver_checkpoint_reads','driver_provider_calls',
    'driver_keygen_runs','driver_ckks_runs','downloads','installations','integrity_checks')
REGISTRY_FIELDS={'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'}


def read(path):return json.loads(Path(path).read_text())


def same(a,b):return isinstance(a,(str,Path)) and isinstance(b,(str,Path)) and Path(a).resolve()==Path(b).resolve()


def selected_step_seconds(args):
    return validate_step_seconds(getattr(args,'step_seconds',DEFAULT_STEP_SECONDS))


def phase_limits(args,phase):
    selected=selected_step_seconds(args)
    seconds,memory=PHASE_LIMITS[phase]
    return (selected+240,memory)if phase=='step'else(seconds,memory)


def durable_json(path,record):
    write_json(path,record)
    with Path(path).open('rb') as stream:os.fsync(stream.fileno())
    descriptor=os.open(Path(path).parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def session_state(case):
    state=read(Path(case)/'session/SESSION.json')
    if any(state.get(k)!=v for k,v in STATE_POLICY.items()):
        raise ValueError('Retain the original BERT v11/c2 session policy and ten-call budget')
    if (state.get('phase') not in ('evaluate','tune','confirm','done')
            or type(state.get('calls')) is not int or not 0<=state['calls']<=10
            or type(state.get('iteration')) is not int or not 1<=state['iteration']<=11):
        raise ValueError('Retained BERT session checkpoint or call count is invalid')
    return {key:state.get(key) for key in (*STATE_POLICY,'phase','iteration','calls','failed')}


def source_gate(args):
    selected_step_seconds(args)
    from .bert_checkpoint_audit import source_gate as checkpoint_source
    case=args.source_case.resolve();info=read(case/'model/model_info.json')
    original=info.get('source') or {};model=Path(original['local_path']).resolve();inputs=Path(original['inputs_path']).resolve()
    prepared=case/'phases/001-prepare/worker_result.json';state=session_state(case)
    checkpoint=checkpoint_source(model,prepared,case/'model/model_info.json')
    split=info.get('split') or {};sample=read(case/'model/samples/source.json')
    if (split.get('kind')!='sst2_native' or split.get('version')!='sst2-native-v1' or split.get('seed')!=20261024
            or sample.get('partition')!=split or sample.get('native_test_read') is not False
            or not same(sample.get('source'),inputs)
            or any(split.get(key)!=list(range(start,start+count)) for key,start,count in
                [('calibration',0,1000),('heldout',1000,1000),('confirm',2000,1000),('report',3000,872)])):
        raise ValueError('Retain fixed original SST-2 training partitions and the frozen 872-row native validation report')
    template=read(args.metadata_template)
    from .r29_local_compile_matrix import compiler_argv
    compiler_argv(template,'pars',binary='/compiler',source='/input',profile='/profile',output='/lowered')
    if (args.profile.resolve()!=(paths.PROFILES_DIR/'qwen48/compiler-config.json').resolve()
            or args.frontend.resolve()!=(paths.DACAPO_ROOT/'python/hecate/hecate/expr.py').resolve()):
        raise ValueError('Preserve the original qwen48 compiler profile and trusted expression frontend')
    if (case.parent!=paths.RESULTS_DIR.resolve() or args.out.parent!=paths.RESULTS_DIR.resolve()
            or args.checkpoint_out.parent!=paths.RESULTS_DIR.resolve() or args.compile_out.parent!=paths.RESULTS_DIR.resolve()
            or len({case,args.out,args.checkpoint_out,args.compile_out})!=4):
        raise ValueError('Use separate retained source, new driver, independent audit and compile directories')
    for target in (args.out,args.checkpoint_out,args.compile_out):
        if any(target==src or target in src.parents or src in target.parents for src in (model,inputs,case)):
            raise ValueError('Continuation outputs must preserve all original model, input and retained session sources')
    return dict(source_case=str(case),model=str(model),inputs=str(inputs),model_info=str(case/'model/model_info.json'),
        prepare_result=str(prepared),original_checkpoint=checkpoint['checkpoint'],initial_session=state,
        split_version=split['version'],calibration_samples=1000,heldout_samples=1000,confirm_samples=1000,report_samples=872,
        metadata_template=str(args.metadata_template),inferred_from_original_metadata=True,
        report_feedback_permitted=False,compile_feedback_permitted=False,new_model_or_sample_reads=0)


def command(args,source,phase):
    seconds=selected_step_seconds(args)
    if phase=='checkpoint':
        return [sys.executable,'-B','-m',CHECKPOINT,'--model',source['model'],
            '--prepare-result',source['prepare_result'],'--model-info',source['model_info'],
            '--out',str(args.checkpoint_out),'--intake-python',str(args.intake_python)]
    if phase in ('step','report'):
        result=[sys.executable,'-B','-m',RECORDED,'--phase',phase,'--model',source['model'],
            '--inputs',source['inputs'],'--out',source['source_case'],'--intake-python',str(args.intake_python),
            '--data-python',str(args.data_python),'--mem-gib','4','--ledger',str(args.ledger),
            '--allowance',str(args.allowance),'--metadata-template',str(args.metadata_template)]
        if phase=='step'and seconds!=DEFAULT_STEP_SECONDS:result+=['--step-seconds',str(seconds)]
        return result
    if phase=='compile':
        report=read(Path(source['source_case'])/'session/REPORT_ACCESS.json')
        return [sys.executable,'-B','-m',COMPILE,'--source-case',source['source_case'],
            '--checkpoint-audit',str(args.checkpoint_out/'result.json'),'--report-result',report['result'],
            '--out',str(args.compile_out),'--intake-python',str(args.intake_python),
            '--metadata-template',str(args.metadata_template),'--profile',str(args.profile),'--frontend',str(args.frontend)]
    raise ValueError('Unknown trusted continuation phase')


def foreign_heavy_processes():
    """Public process metadata admission; root remains the shared job scheduler."""
    table=_scan();own={os.getpid()}
    while True:
        expanded=own|{pid for pid,row in table.items() if row.get('ppid') in own}
        if expanded==own:break
        own=expanded
    heavy=[]
    for pid,row in table.items():
        if pid in own or row.get('state')=='Z':continue
        try:name=(Path('/proc')/str(pid)/'exe').resolve().name
        except (FileNotFoundError,PermissionError,OSError):continue
        if (name in ('r31-client','dacapo-client','hecate-opt','cc1plus','ld.lld')
                or name.startswith('python') and row.get('vmrss_kib',0)>512*1024):
            heavy.append(dict(pid=pid,starttime_ticks=row.get('starttime'),executable_name=name,
                sampled_rss_kib=row.get('vmrss_kib')))
    return heavy


def check_budget(out):
    if artifact_bytes(out)+COMPLETION_RESERVE_BYTES>CASE_BYTES or shutil.disk_usage(out).free<10*GIB:
        raise RuntimeError('Continuation metadata allowance or disk reserve exhausted')


def persist_terminal_completion(out,*,started_ns,result,allow_resource_failure=False):
    """Use the reserved marker space for closure, including recorded guard stops.

    A failed resource observation can close after its case projection has been
    exceeded. That closure records the excess; successful cases keep their cap.
    Actual filesystem errors propagate and leave the honest partial record.
    """
    before=artifact_bytes(out)
    budget=result.get('terminal_budget') or {}
    failed_closure=(allow_resource_failure and result.get('status')=='error' and bool(budget.get('reasons')))
    if allow_resource_failure and not failed_closure:
        raise ValueError('Resource-failure closure requires a recorded failed budget observation')
    if before+COMPLETION_RESERVE_BYTES>CASE_BYTES and not failed_closure:
        raise RuntimeError('Continuation completion reserve is exhausted')
    marker=persist_case_completion(out,round_label=ROUND,started_ns=started_ns,status=result['status'],
        timing=result['timing'],required_artifact_bytes=before)
    if failed_closure:
        marker['resource_budget']=dict(case_limit_bytes=CASE_BYTES,
            observed_disk_free_bytes=budget['observed_disk_free_bytes'],minimum_free_disk_bytes=10*GIB,
            case_allowance_with_reserved_completion_passed=budget['case_allowance_with_reserved_completion_passed'],
            disk_reserve_passed=budget['disk_reserve_passed'],reasons=budget['reasons'],
            failed_resource_closure=True,successful_case_cap_unchanged=True,
            reserved_completion_bytes=COMPLETION_RESERVE_BYTES,
            actual_case_within_cap=None,actual_case_overshoot_bytes=None)
        # Include this truthful resource-failure metadata in the marker's own
        # fixed point. The completion endpoint still excludes marker writes.
        for _ in range(12):
            actual=artifact_bytes(out)
            marker['case_artifact_bytes']=actual
            marker['resource_budget'].update(actual_case_within_cap=actual<=CASE_BYTES,
                actual_case_overshoot_bytes=max(0,actual-CASE_BYTES))
            write_json(Path(out)/'completion.json',marker)
            if artifact_bytes(out)==actual:break
        else:raise RuntimeError('Continuation failed-resource completion size did not converge')
    actual=artifact_bytes(out)
    if (Path(out,'completion.json').stat().st_size>COMPLETION_RESERVE_BYTES
            or actual>CASE_BYTES and not failed_closure or marker['case_artifact_bytes']!=actual):
        raise RuntimeError('Continuation final artifact or reserved completion size exceeded')
    return marker


def terminal_budget(size,disk_free):
    """Observe admission limits without admitting another phase during closure."""
    case_ok=size+COMPLETION_RESERVE_BYTES<=CASE_BYTES
    disk_ok=disk_free>=10*GIB
    return dict(observed_case_artifact_bytes_before_completion=size,case_limit_bytes=CASE_BYTES,
        observed_case_overshoot_bytes=max(0,size-CASE_BYTES),reserved_completion_bytes=COMPLETION_RESERVE_BYTES,
        case_allowance_with_reserved_completion_passed=case_ok,observed_disk_free_bytes=disk_free,
        minimum_free_disk_bytes=10*GIB,disk_reserve_passed=disk_ok,
        reasons=([] if case_ok else ['metadata case or reserved completion allowance exhausted'])+
                ([] if disk_ok else ['remaining physical disk below10GiB reserve']),
        scope='Actual metadata inventory and disk snapshot during terminal persistence; no child phase admitted')


@contextmanager
def serial_lease(path):
    with Path(path).open('a') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another BERT continuation owns the serial execution lease') from None
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def frozen_state(case):
    case=Path(case);state=session_state(case);session=case/'session';agent=read(session/'AGENT.json')
    if state['phase']!='done':raise ValueError('Original Agent session is not finalized')
    selected=(agent.get('confirm') or {}).get('selected')
    if selected is None:
        if (session/'FROZEN.json').exists() or (session/'program.ptdsl').exists():
            raise ValueError('Unselected candidate has unexpected frozen artifacts')
        if agent.get('status')=='confirm_failed' and state['failed'] is None:return dict(outcome='quality_rejected',frozen=False)
        raise RuntimeError('Original Agent finalized with an error: '+str(agent.get('error')))
    marker=read(session/'FROZEN.json');program=session/'program.ptdsl';confirmation=agent['confirm']
    matches=[row for row in confirmation.get('evals',[]) if (row.get('iteration'),row.get('origin'))==
        (selected.get('iteration'),selected.get('origin'))]
    if (state['failed'] is not None or agent.get('status')!='feasible' or not same(marker.get('program'),program)
            or selected.get('source')!=program.read_text() or marker.get('confirm')!=confirmation
            or marker.get('origin')!=selected.get('origin') or marker.get('iteration')!=selected.get('iteration')
            or any(marker.get(k)!=STATE_POLICY[k] for k in ('prompt_version','tune_rule','confirm_rule','split_version'))
            or confirmation.get('confirm_rule')!='c2' or confirmation.get('applicable') is not True or len(matches)!=1):
        raise ValueError('Original selected source and C2 frozen metadata differ')
    row=matches[0];tail=row.get('tail') or {}
    def acceptable(value,maximum):return type(value) in (int,float) and math.isfinite(value) and value<=maximum
    if (row.get('total')!=1000 or row.get('passed') is not True or row.get('diverged') is not False or row.get('n_diverged')!=0
            or not acceptable(row.get('acc_drop_pp'),.5) or not acceptable(row.get('max_u_calibrated'),.90)
            or not acceptable(row.get('max_u_analytic'),1) or tail.get('passed') is not True or tail.get('total')!=3000
            or tail.get('diverged') is not False or tail.get('n_diverged')!=0
            or not acceptable((tail.get('metrics') or {}).get('acc_drop_pp'),.5)):
        raise ValueError('Original C2 task, calibrated-domain and analytic-domain confirmation did not pass')
    return dict(outcome='frozen',frozen=True,program=str(program),selected_iteration=selected['iteration'],selected_origin=selected['origin'])


def accepted_checkpoint(args,source):
    from .bert_checkpoint_audit import ROUND as expected_round,ZERO_FIELDS as zeros
    record=read(args.checkpoint_out/'result.json');worker=record.get('worker') or {};coverage=worker.get('state_coverage') or {}
    process=record.get('process') or {}
    if (record.get('status')!='ok' or record.get('round')!=expected_round or record.get('model')!=MODEL
            or record.get('checkpoint_semantic_coverage_passed') is not True or worker.get('status')!='ok'
            or coverage.get('passed') is not True or coverage.get('classifier',{}).get('both_original_checkpoint_keys') is not True
            or not same(worker.get('source',{}).get('model_info'),source['model_info'])
            or not same(worker.get('source',{}).get('model'),source['model'])
            or any(worker.get(k)!=0 for k in zeros)
            or any(worker.get(k)!=1 for k in ('checkpoint_load_attempts','checkpoint_load_completed','from_pretrained_attempts','from_pretrained_completed'))
            or process.get('exit_code')!=0 or process.get('timed_out') is not False or process.get('signal') is not None
            or process.get('remaining_pids')!=[]):
        raise ValueError('Same-model trained classifier semantic coverage was not accepted')
    return record


def phase_result(case,phase):
    retained=read(Path(case)/'result.json')
    if retained.get('phase')!=phase or retained.get('model')!=MODEL or retained.get('status')!='ok':
        raise ValueError('The completed original BERT phase result differs')
    from .bert_sst2 import check_worker
    check_worker(phase,retained.get('worker') or {})
    return retained


def accepted_report(args):
    access=read(args.source_case/'session/REPORT_ACCESS.json')
    if access.get('complete') is not True:raise ValueError('Keep the retained incomplete report; a second report read is not scheduled')
    report_path=Path(access['result']).resolve()
    if report_path.parent.parent!=args.source_case/'phases' or not report_path.parent.name.endswith('-report'):
        raise ValueError('Frozen report result must remain inside its original phase directory')
    from .bert_frozen_compile import source_gate as compile_source
    source=compile_source(args.source_case,args.checkpoint_out/'result.json',report_path)
    return dict(result=str(report_path),task_policy_passed=source['report_task_acceptance'],source=source)


def _snapshot_state(args,folder):
    folder.mkdir()
    for relative in ('result.json','session/SESSION.json','session/AGENT.json','session/FROZEN.json','session/REPORT_ACCESS.json'):
        source=args.source_case/relative
        if source.is_file():
            if artifact_bytes(args.out)+source.stat().st_size+COMPLETION_RESERVE_BYTES>CASE_BYTES:
                raise RuntimeError('Continuation source snapshot exceeds the reserved case allowance')
            shutil.copyfile(source,folder/relative.replace('/','-'))


def _launch(args,source,phase,index,recorder,result,deadline):
    if result.get('termination') is not None:raise InterruptedError('Continuation already interrupted')
    foreign=foreign_heavy_processes()
    if foreign:result['blocked_processes']=foreign;raise RuntimeError('Another heavy process is active; keep the serial continuation unstarted')
    check_budget(args.out);work=args.out/'phases'/('%03d-%s'%(index,phase));work.mkdir()
    _snapshot_state(args,work/'before')
    argv=command(args,source,phase);seconds,memory=phase_limits(args,phase)
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('Finite BERT continuation wall allowance exhausted')
    row=dict(phase=phase,argv=argv,source_state_before=session_state(args.source_case),observation=None)
    result['processes'].append(row);result['attempted'][phase]+=1;durable_json(args.out/'result.json',result)
    with recorder.span('trusted_'+phase,step_index=index):
        row['observation']=run_process(argv,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=work/'driver.log',
            resources=work/'resources.jsonl',timeout=min(seconds,remaining),mem_gib=memory,fsize_gib=8,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'))
    observation=row['observation'];row['source_state_after']=session_state(args.source_case)
    if observation.get('signal') is not None:
        if result.get('termination') is None:result['termination']=dict(cause='incoming_signal',signal=observation['signal'])
        else:row['cleanup_signal_after_recorded_termination']=observation['signal']
    durable_json(args.out/'result.json',result)
    if (observation.get('exit_code')!=0 or observation.get('timed_out') or observation.get('signal') is not None
            or observation.get('remaining_pids')):
        raise RuntimeError('Trusted BERT '+phase+' did not complete cleanly; subsequent phases stopped')
    result['completed'][phase]+=1
    if phase in ('step','report'):phase_result(args.source_case,phase)


def execute(args):
    source=source_gate(args)
    if args.out.exists():
        if not args.launched or sorted(p.name for p in args.out.iterdir())!=['driver.log','launch.json']:
            raise ValueError('Use a fresh continuation case and preserve prior attempts')
        launch=read(args.out/'launch.json')
        if not same(launch.get('source_case'),args.source_case):raise ValueError('Detached launch source differs')
    else:args.out.mkdir()
    (args.out/'phases').mkdir();started=time.perf_counter_ns();deadline=time.monotonic()+args.overall_seconds
    recorder=StageRecorder(args.out/'events.jsonl');stop=threading.Event();guard_failure=[]
    result=dict(round=ROUND,model=MODEL,mode=MODE,status='running',error=None,termination=None,source=source,
        processes=[],attempted=dict.fromkeys(PHASE_LIMITS,0),completed=dict.fromkeys(PHASE_LIMITS,0),reused=[],
        frozen=False,report_completed=False,task_report_acceptance=None,compile_completed=False,
        compiled_bootstraps=None,predicted_latency_us=None,CKKS_latency=None,security_acceptance=None,
        driver_artifact_bytes_before_completion=0,overall_seconds=args.overall_seconds,max_step_commands=args.max_steps,
        step_wall_allowance=dict(inner_seconds=selected_step_seconds(args),nix_seconds=selected_step_seconds(args)+60,
            phase_R10_seconds=selected_step_seconds(args)+120,serial_R10_seconds=selected_step_seconds(args)+240,
            step_AS_gib=PHASE_LIMITS['step'][1],tuner_budget_s=STATE_POLICY['tuner_budget_s'],maximum_session_calls=10),
        **dict.fromkeys(ZERO_FIELDS,0))
    result['delegated_api_accounting']=dict(ledger=str(args.ledger),
        scope='Existing BERT phase driver records actual ledger intervals and unique llm-call registry rows. driver_provider_calls counts only this orchestrator, which creates no Provider.',
        duplicate_api_registration_by_this_driver=False)
    result['completion_accounting']=dict(reserved_bytes=COMPLETION_RESERVE_BYTES,total_case_allowance_bytes=CASE_BYTES,
        reserve_charged_before_registry=True,final_case_inventory_in='completion.json',
        failed_resource_closure_allowed=True,successful_case_cap_unchanged=True,
        scope='Driver result inventory precedes completion.json; resource failures retain actual overshoot and disk snapshot while using reserved closure space. Successful cases keep the512MiB cap. Actual filesystem write errors propagate.')
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def interrupted(signum,frame):
        if not guard_failure:result['termination']=dict(cause='incoming_signal',signal=signum)
        raise InterruptedError('BERT continuation interrupted '+str(signum))
    for signum in handlers:signal.signal(signum,interrupted)
    def monitor():
        while not stop.wait(1):
            try:
                check_budget(args.out)
                if time.monotonic()>=deadline:raise TimeoutError('BERT continuation total wall allowance exhausted')
            except BaseException as exc:
                failure=dict(cause='driver_guard',error=type(exc).__name__+': '+str(exc));guard_failure.append(failure)
                result['termination']=failure
                try:durable_json(args.out/'guard-stop.json',failure)
                finally:os.kill(os.getpid(),signal.SIGTERM)
                return
    watcher=threading.Thread(target=monitor,daemon=True)
    try:
        with serial_lease(paths.RESULTS_DIR/'bert-serial-continuation.lock'),recorder.span('complete_bert_serial_continuation'):
            shutil.copyfile(Path(__file__),args.out/'driver-source.py')
            for name in ('bert_sst2.py','bert_sst2_worker.py','bert_sst2_recorded.py','bert_checkpoint_audit.py','bert_frozen_compile.py'):
                shutil.copyfile(Path(__file__).with_name(name),args.out/name)
            shutil.copyfile(args.metadata_template,args.out/'recorded-compile-template.json')
            durable_json(args.out/'source.json',source);_snapshot_state(args,args.out/'initial-state');watcher.start()
            index=1
            if not args.checkpoint_out.exists():_launch(args,source,'checkpoint',index,recorder,result,deadline);index+=1
            else:result['reused'].append(dict(phase='checkpoint',path=str(args.checkpoint_out/'result.json')))
            accepted_checkpoint(args,source)
            access=args.source_case/'session/REPORT_ACCESS.json'
            if access.exists() and read(access).get('complete') is not True:
                raise ValueError('The original report attempt is incomplete; preserve it without another read')
            for step in range(args.max_steps):
                state=session_state(args.source_case)
                if state['phase']=='done':break
                if access.exists():raise ValueError('Report access exists before the original session is finalized')
                _launch(args,source,'step',index,recorder,result,deadline);index+=1
            else:
                if session_state(args.source_case)['phase']!='done':raise RuntimeError('Finite serial step allowance exhausted')
            outcome=frozen_state(args.source_case);result['selection_outcome']=outcome
            if not outcome['frozen']:result.update(status='quality_rejected',frozen=False)
            else:
                result['frozen']=True
                if not access.exists():_launch(args,source,'report',index,recorder,result,deadline);index+=1
                else:result['reused'].append(dict(phase='report',path=read(access).get('result')))
                report=accepted_report(args);result.update(report_completed=True,task_report_acceptance=report['task_policy_passed'])
                if not args.compile_out.exists():_launch(args,source,'compile',index,recorder,result,deadline)
                else:result['reused'].append(dict(phase='compile',path=str(args.compile_out/'result.json')))
                compiled=read(args.compile_out/'result.json')
                if (compiled.get('status')!='ok' or compiled.get('round')!='BERT-FROZEN-COMPILE'
                        or compiled.get('source')!=report['source'] or compiled.get('completed')!=dict(emit=1,trace=1,compile=1)):
                    raise ValueError('Same frozen-source trusted BERT compilation did not complete')
                result.update(status='ok',compile_completed=True,
                    compiled_bootstraps=compiled.get('compiled_bootstraps'),predicted_latency_us=compiled.get('predicted_latency_us'))
            result['final_session']=session_state(args.source_case)
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        stop.set()
        if watcher.is_alive():watcher.join(timeout=3)
        for signum in handlers:signal.signal(signum,signal.SIG_IGN)
        try:
            result['timing']=summarize_events(read_events(args.out/'events.jsonl'))
            result['recorded_before_persistence_seconds']=(time.perf_counter_ns()-started)/1e9
            disk_free=shutil.disk_usage(args.out).free
            for _ in range(20):
                size=artifact_bytes(args.out);result['terminal_budget']=terminal_budget(size,disk_free)
                if result['terminal_budget']['reasons']:
                    result['status']='error'
                    if result.get('error') is None:result['error']='Terminal continuation resource allowance exhausted'
                    if result.get('termination') is None:
                        result['termination']=dict(cause='driver_guard',error='; '.join(result['terminal_budget']['reasons']))
                result['driver_artifact_bytes_before_completion']=size
                result['metrics']={k:result.get(k) for k in ('status','attempted','completed','frozen','report_completed',
                    'task_report_acceptance','compile_completed','compiled_bootstraps','predicted_latency_us','CKKS_latency',
                    'security_acceptance','termination','terminal_budget')+ZERO_FIELDS}
                durable_json(args.out/'result.json',result)
                if artifact_bytes(args.out)==size:break
            else:raise RuntimeError('Continuation terminal metadata size did not converge')
            from ..registry import git_rev
            row=dict(run_id=args.out.name,round=ROUND,model=MODEL,mode=MODE,git_rev=git_rev(),metrics=result['metrics'],
                artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat())
            with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            persist_terminal_completion(args.out,started_ns=started,result=result,
                allow_resource_failure=bool(result['terminal_budget']['reasons']))
        finally:
            for signum,handler in handlers.items():signal.signal(signum,handler)
    return result


def start_detached(args):
    source_gate(args)
    if args.out.exists():raise ValueError('Preserve every prior detached launch directory')
    if foreign_heavy_processes():raise RuntimeError('A heavy process is active; detached continuation was not started')
    args.out.mkdir();record=dict(source_case=str(args.source_case),launch_requested_utc=datetime.now(timezone.utc).isoformat(),
        launcher_pid=os.getpid(),pid=None,credentials_read=False,environment_read=False)
    durable_json(args.out/'launch.json',record)
    argv=[sys.executable,'-B','-m',MODULE,'--launched']
    for name in ('source_case','out','checkpoint_out','compile_out','intake_python','data_python','ledger','allowance','metadata_template','profile','frontend','max_steps','overall_seconds'):
        argv+=['--'+name.replace('_','-'),str(getattr(args,name))]
    if selected_step_seconds(args)!=DEFAULT_STEP_SECONDS:argv+=['--step-seconds',str(selected_step_seconds(args))]
    with (args.out/'driver.log').open('ab') as stream:
        child=subprocess.Popen(argv,cwd=paths.REPO/'DSL/DSL_agent/scripts',stdin=subprocess.DEVNULL,stdout=stream,
            stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    record.update(pid=child.pid,argv=argv);durable_json(args.out/'launch.json',record)
    return dict(status='detached_started',pid=child.pid,out=str(args.out),new_compute_completed=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--checkpoint-out',type=Path);parser.add_argument('--compile-out',type=Path)
    parser.add_argument('--intake-python',type=Path,default=paths.INTAKE_PYTHON)
    parser.add_argument('--data-python',type=Path,default=paths.PAPER_DATA_PYTHON or paths.ORBIT_ROOT.parent/'paper-data-venv/bin/python')
    parser.add_argument('--ledger',type=Path,default=paths.RESULTS_DIR/'llm_calls.jsonl')
    parser.add_argument('--allowance',type=Path,default=paths.RESULTS_DIR/'autonomous-api-allowance.json')
    parser.add_argument('--metadata-template',type=Path,default=paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json')
    parser.add_argument('--profile',type=Path,default=paths.PROFILES_DIR/'qwen48/compiler-config.json')
    parser.add_argument('--frontend',type=Path,default=paths.DACAPO_ROOT/'python/hecate/hecate/expr.py')
    parser.add_argument('--max-steps',type=int,default=MAX_STEPS);parser.add_argument('--overall-seconds',type=int,default=OVERALL_SECONDS)
    parser.add_argument('--step-seconds',type=int,default=DEFAULT_STEP_SECONDS,
        help='Only the original Agent step wall allowance; integer7200..21600, default7200')
    group=parser.add_mutually_exclusive_group();group.add_argument('--detach',action='store_true');group.add_argument('--launched',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    try:selected_step_seconds(args)
    except ValueError as exc:parser.error(str(exc))
    if not 1<=args.max_steps<=MAX_STEPS or not 0<args.overall_seconds<=OVERALL_SECONDS:parser.error('Use finite positive orchestration allowances within their fixed maxima')
    for name in ('source_case','out','ledger','allowance','metadata_template','profile','frontend'):
        setattr(args,name,getattr(args,name).expanduser().resolve())
    for name in ('intake_python','data_python'):setattr(args,name,getattr(args,name).expanduser().absolute())
    for name,suffix in (('checkpoint_out','-checkpoint'),('compile_out','-compile')):
        value=getattr(args,name);setattr(args,name,(paths.RESULTS_DIR/(args.out.name+suffix) if value is None else value).resolve())
    if args.launched:
        deadline=time.monotonic()+5
        while read(args.out/'launch.json').get('pid')!=os.getpid():
            if time.monotonic()>=deadline:raise RuntimeError('Detached launch PID handshake did not complete')
            time.sleep(.01)
    result=start_detached(args) if args.detach else execute(args)
    if not args.launched:print(json.dumps({k:result.get(k) for k in ('status','pid','frozen','report_completed','compile_completed','error')}),flush=True)
    return 0 if result['status'] in ('ok','quality_rejected','detached_started') else 1


if __name__=='__main__':raise SystemExit(main())
