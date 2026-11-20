"""Compile the original frozen SST-2 program through trusted lowering only.

The original session, independent checkpoint audit and one completed frozen
report supply source gates. No Agent, dataset reader or package manifest runs.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

from .. import paths
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample
from .r29_local_compile_matrix import compiler_argv, native_metadata_commands, plan_observation
from .r31_lattigo_runtime_plan import physical_memory_observation

ROUND='BERT-FROZEN-COMPILE'
MODEL='bert-tiny-sst2'
MODE='frozen-program-plan-qwen48-trusted-compile'
GIB=1024**3
PHASES=('emit','trace','compile')
PHASE_SECONDS=dict(emit=1800,trace=1800,compile=7200)
OVERALL_SECONDS=11160
CASE_BYTES=8*GIB
ZERO_FIELDS=('api_calls','llm_calls','model_forwards','raw_sample_reads','confirm_samples','report_samples',
    'keygen_runs','ckks_executions','integrity_checks','downloads','installations')
CRITERION='abs(observed-reference) <= 1e-12 + 1e-12*abs(reference)'


def _read(path):return json.loads(Path(path).read_text())


def _finite(value):return type(value) in (int,float) and math.isfinite(value)


def source_gate(case,checkpoint_audit,report_result):
    case=Path(case).resolve();checkpoint_audit=Path(checkpoint_audit).resolve();report_result=Path(report_result).resolve()
    model_info=case/'model/model_info.json';session=case/'session';program=session/'program.ptdsl'
    info=_read(model_info);state=_read(session/'SESSION.json');agent=_read(session/'AGENT.json')
    frozen=_read(session/'FROZEN.json');access=_read(session/'REPORT_ACCESS.json');report=_read(report_result)
    original=info.get('source',{});split=info.get('split',{});sample_source=_read(case/'model/samples/source.json')
    if (original.get('type')!='hf' or original.get('config',{}).get('model_type')!='bert'
            or original.get('config',{}).get('num_labels')!=2
            or info.get('task')!='classification' or info.get('metric')!='classification_top1'
            or info.get('threshold')!={'acc_drop_pp':.5}
            or split.get('kind')!='sst2_native' or split.get('version')!='sst2-native-v1'
            or split.get('seed')!=20261024
            or any(split.get(name)!=list(range(start,start+n)) for name,start,n in
                [('calibration',0,1000),('heldout',1000,1000),('confirm',2000,1000),('report',3000,872)])
            or sample_source.get('partition')!=split or sample_source.get('native_test_read') is not False
            or Path(sample_source.get('source','')).resolve()!=Path(original.get('inputs_path','')).resolve()):
        raise ValueError('Preserve the original inferred BERT classification model and fixed SST-2 partition')
    if (state.get('phase')!='done' or state.get('prompt_version')!='r8-v11'
            or state.get('tune_rule')!='v8-6.25' or state.get('confirm_rule')!='c2'
            or state.get('numeric_lock') is not True or state.get('focus_retune') is not True
            or state.get('tune_policy')!='all' or state.get('tune_search')!='sweep'
            or state.get('engine_profile')!='tensor'
            or state.get('split_version')!='sst2-native-v1'
            or frozen.get('prompt_version')!='r8-v11' or frozen.get('confirm_rule')!='c2'
            or frozen.get('tune_rule')!='v8-6.25' or frozen.get('split_version')!='sst2-native-v1'
            or Path(frozen.get('program','')).resolve()!=program
            or frozen.get('confirm')!=agent.get('confirm')):
        raise ValueError('Require the completed unchanged v11/c2 BERT frozen session')
    confirmation=agent.get('confirm') or {};selected=confirmation.get('selected') or {}
    if (selected.get('source')!=program.read_text() or confirmation.get('confirm_rule')!='c2'
            or confirmation.get('applicable') is not True
            or frozen.get('iteration')!=selected.get('iteration') or frozen.get('origin')!=selected.get('origin')):
        raise ValueError('Frozen source must exactly match the original confirmed candidate')
    matches=[r for r in confirmation.get('evals',[]) if (r.get('iteration'),r.get('origin'))==
        (selected.get('iteration'),selected.get('origin'))]
    if len(matches)!=1:raise ValueError('Require one actual confirmation for the frozen selected candidate')
    confirmed=matches[0];tail=confirmed.get('tail') or {}
    if (confirmed.get('passed') is not True or confirmed.get('diverged') is not False
            or confirmed.get('n_diverged')!=0 or not _finite(confirmed.get('acc_drop_pp')) or confirmed['acc_drop_pp']>.5
            or not _finite(confirmed.get('max_u_calibrated')) or confirmed['max_u_calibrated']>.90
            or not _finite(confirmed.get('max_u_analytic')) or confirmed['max_u_analytic']>1
            or tail.get('passed') is not True or tail.get('total')!=3000 or tail.get('diverged') is not False
            or tail.get('n_diverged')!=0 or not _finite(tail.get('metrics',{}).get('acc_drop_pp'))
            or tail['metrics']['acc_drop_pp']>.5):
        raise ValueError('Retain the original selected-candidate C2 confirmation and nonreport tail acceptance')
    if (access.get('complete') is not True or access.get('report_samples')!=872
            or Path(access.get('program','')).resolve()!=program or Path(access.get('result','')).resolve()!=report_result
            or report_result.parent.parent!=case/'phases' or not report_result.parent.name.endswith('-report')
            or report.get('model')!=MODEL or report.get('phase')!='report' or report.get('status')!='ok'
            or report.get('frozen') is not True or report.get('report_completed') is not True
            or report.get('report_samples')!=872 or report.get('task_policy')!={'acc_drop_pp':.5}
            or type(report.get('task_policy_passed')) is not bool
            or any(report.get(k)!=0 for k in ('native_test_reads','llm_calls','compiler_runs','keygen_runs',
                'real_ckks_attempts','real_ckks_completed'))):
        raise ValueError('Use the unchanged completed frozen report; its quality outcome does not select a new candidate')
    math_rows=report.get('frozen_math') or []
    if (len(math_rows)!=3 or [r.get('sample') for r in math_rows]!=[0,1,2]
            or any(len(r.get('checks',[]))!=3 or any(c.get('passed') is not True or c.get('criterion')!=CRITERION
                for c in r['checks']) for r in math_rows)):
        raise ValueError('All three frozen Graph/Tensor/Mock mathematical comparison groups must pass')
    audited=_read(checkpoint_audit);worker=audited.get('worker') or {};coverage=worker.get('state_coverage') or {}
    if (audited.get('round')!='BERT-CHECKPOINT-AUDIT' or audited.get('model')!=MODEL or audited.get('status')!='ok'
            or audited.get('checkpoint_semantic_coverage_passed') is not True
            or worker.get('status')!='ok' or coverage.get('passed') is not True
            or coverage.get('classifier',{}).get('both_original_checkpoint_keys') is not True
            or Path(worker.get('source',{}).get('model_info','')).resolve()!=model_info
            or Path(worker.get('source',{}).get('model','')).resolve()!=Path(original.get('local_path','')).resolve()
            or any(worker.get(k)!=0 for k in ('model_forwards','raw_sample_reads','api_calls','integrity_checks'))):
        raise ValueError('Accepted same-model trained-checkpoint semantic coverage is required')
    return dict(source_case=str(case),model_info=str(model_info),program=str(program),session=str(session),
        checkpoint_audit=str(checkpoint_audit),report_result=str(report_result),
        original_model=original['local_path'],inputs=info['inputs'],sequence=info['sequence'],
        split_version=split['version'],selected_iteration=selected['iteration'],selected_origin=selected['origin'],
        confirm=confirmed,report_task_acceptance=report['task_policy_passed'],
        report_validation_metrics=report.get('validation_metrics'),report_outcome_used_for_selection=False,
        mathematical_groups_passed=3,api_feedback_permitted=False)


def _numeric_environment():
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'


def _emit(args,result,recorder):
    # This scope must precede cost imports: cost.PROFILES computes its boot
    # normalization through management_profile -> compile_argv at import time.
    with native_metadata_commands(_read(args.metadata_template)):
        from ..models.info import ModelInfo
        from ..dsl.parser import parse
        from ..dsl.interface import require_interface
        from ..dsl.typecheck import check
        from ..plan import plan_program
        from ..backends import CountBackend,EmitBackend,RefreshBackend
        from ..lower import lower
        with recorder.span('load_original_weights_and_typecheck'):
            info=ModelInfo.load(args.model_info);program=parse(args.program.read_text())
            result['interface']=require_interface(program,info)
            checked=check(program,{k:v.shape for k,v in info.graph.weights.items()},graph=info.graph,
                prompt_version='r8-v11')
            if not checked.ok:raise ValueError('Frozen BERT program does not typecheck on its original graph/weights')
            if program.header.refresh!='agent' or program.header.placement!='program-plan':
                raise ValueError('Retain the original frozen Agent program-plan header')
        with recorder.span('original_program_bootstrap_placement'):
            plan,backend,placement=plan_program(program,info.graph.weights,profile='qwen48')
            write_json(args.work_dir/'refresh-plan.json',plan)
            result.update(planned_bootstraps=plan['boots'],placement_variant=plan['placement'])
            del backend,placement;gc.collect()
        with recorder.span('independent_count_plan_replay'):
            counter=CountBackend(program.header.slots,boot_out_level=plan['levels']['boot_out_level'])
            counted=RefreshBackend(counter,policy='plan',plan=plan,**plan['levels'])
            lower(program,info.graph.weights,counted,act_mode='program')
            result['count_replay']=dict(stats=counter.stats(),refresh=counted.stats())
            if (result['count_replay']['stats']['counts']['boot']!=plan['boots']
                    or result['count_replay']['refresh']['boots']!=plan['boots']):
                raise ValueError('Count replay bootstrap count differs from the original placement')
            del counter,counted;gc.collect()
        emit=args.work_dir/'emit'
        with recorder.span('trusted_emit_lower'):
            backend=EmitBackend(program.header.slots,frontend=args.frontend,constants=emit/'constants.npz',save_dir=emit)
            active=RefreshBackend(backend,policy='plan',plan=plan,**plan['levels'])
            lower(program,info.graph.weights,active,act_mode='program');result['emit_stats']=backend.stats()
            if any(result['emit_stats'][key]!=result['count_replay']['stats'][key]
                   for key in ('counts','ops_total','distinct_consts')):
                raise ValueError('Complete Count/Emit operation and constant inventories differ')
        with recorder.span('trusted_emit_write'):
            text,constants=backend.render();estimated=len(text.encode())+sum(v.nbytes+1024 for v in constants.values())+1024**2
            result['emit_estimated_bytes']=estimated
            if estimated+artifact_bytes(args.work_dir)>CASE_BYTES-GIB or shutil.disk_usage(args.work_dir).free-estimated<10*GIB:
                raise RuntimeError('Trusted BERT Emit disk projection exceeds the explicit case or reserve')
            del text,constants;backend.write(emit)
            result.update(trace_source=str(emit/'candidate_trace.py'),constants=str(emit/'constants.npz'))


def _child(args,phase,argv,recorder,*,cwd,env=None):
    with recorder.span('native_'+phase):
        observed=run_process([str(x) for x in argv],cwd=cwd,log=args.work_dir/(phase+'-native.log'),
            resources=args.work_dir/(phase+'-native-resources.jsonl'),timeout=PHASE_SECONDS[phase],mem_gib=12,fsize_gib=8,env=env)
    write_json(args.work_dir/(phase+'-process.json'),observed)
    if observed.get('exit_code')!=0 or observed.get('timed_out') or observed.get('signal') or observed.get('remaining_pids'):
        raise RuntimeError('Native BERT '+phase+' did not finish cleanly')
    return observed


def worker(args):
    _numeric_environment();args.work_dir.mkdir(parents=True,exist_ok=True)
    target=args.work_dir/(args.worker_phase+'-result.json')
    if target.exists():raise ValueError('Preserve the previous BERT worker phase')
    recorder=StageRecorder(args.work_dir/(args.worker_phase+'-events.jsonl'))
    result=dict(round=ROUND,model=MODEL,mode=MODE,phase=args.worker_phase,status='running',error=None,
        compiled_bootstraps=None,predicted_latency_us=None,CKKS_latency=None,**dict.fromkeys(ZERO_FIELDS,0))
    try:
        with recorder.span('frozen_bert_'+args.worker_phase):
            result['source']=source_gate(args.source_case,args.checkpoint_audit,args.report_result)
            if (Path(result['source']['program'])!=args.program or Path(result['source']['model_info'])!=args.model_info):
                raise ValueError('Every inner phase must use the exact inferred original source paths')
            if args.worker_phase=='emit':_emit(args,result,recorder)
            else:
                from .. import toolchain
                host=toolchain.host_paths();emit=args.work_dir/'emit'
                env=dict(os.environ,HECATE=str(host['hecate']),HECATE_BUILD=str(host['hecate_build']),
                    LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'])
                if args.worker_phase=='trace':
                    result['process']=_child(args,'trace',[sys.executable,'-B',emit/'candidate_trace.py'],recorder,cwd=emit,env=env)
                    mlir=emit/'candidate_trace.mlir'
                    if not mlir.is_file() or mlir.stat().st_size==0:raise ValueError('Trusted trace produced no MLIR')
                    result['mlir']=dict(path=str(mlir),bytes=mlir.stat().st_size)
                else:
                    argv=compiler_argv(_read(args.metadata_template),'pars',binary=host['hecate_opt'],
                        source=emit/'candidate_trace.mlir',profile=args.profile,output=emit/'lowered.mlir',debug_snapshots=False)
                    write_json(args.work_dir/'compile-argv.json',argv)
                    result['process']=_child(args,'compile',argv,recorder,cwd=emit,env=env)
                    plans=sorted(emit.glob('lowered.*.runtime-plan.json'))
                    if len(plans)!=1:raise ValueError('One complete original compiled RuntimePlan is required')
                    observed=plan_observation(_read(plans[0]),_read(args.profile))
                    if observed['compiled_bootstraps']!=_read(args.work_dir/'refresh-plan.json')['boots']:
                        raise ValueError('Native compiled bootstrap count differs from trusted plan replay')
                    result.update(observed,runtime_plan=str(plans[0]))
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    result['timing']=summarize_events(read_events(recorder.path));write_json(target,result)
    return result


def phase_command(args,phase,work):
    command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(PHASE_SECONDS[phase]+60),
        '--python',str(args.intake_python),'--','-B','-m',__spec__.name,'--worker-phase',phase,'--work-dir',str(work)]
    for key in ('source_case','checkpoint_audit','report_result','metadata_template','profile','frontend','model_info','program'):
        command+=['--'+key.replace('_','-'),str(getattr(args,key))]
    return command


def _budget(out):
    if artifact_bytes(out)+65536>CASE_BYTES or shutil.disk_usage(out).free<10*GIB:
        raise RuntimeError('BERT compile case or free disk allowance exhausted')


def execute(args):
    out=args.out.resolve();case=args.source_case.resolve()
    if out.parent!=paths.RESULTS_DIR.resolve() or out.exists() or out in case.parents or case in out.parents or out==case:
        raise ValueError('A fresh separate frozen BERT compile case is required')
    started=time.perf_counter_ns();deadline=time.monotonic()+OVERALL_SECONDS;out.mkdir();work=out/'work';work.mkdir()
    recorder=StageRecorder(out/'events.jsonl');stop=threading.Event();failures=[];active=[None]
    result=dict(round=ROUND,model=MODEL,mode=MODE,status='running',error=None,processes=[],phases={},
        attempted=dict.fromkeys(PHASES,0),completed=dict.fromkeys(PHASES,0),artifact_bytes=0,termination=None,
        compiled_bootstraps=None,predicted_latency_us=None,accuracy_acceptance=None,security_acceptance=None,
        CKKS_latency=None,**dict.fromkeys(ZERO_FIELDS,0))
    old={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def interrupt(sig,frame):
        if not failures:result['termination']=dict(cause='incoming_signal',signal=sig)
        raise InterruptedError('BERT compile interrupted '+str(sig))
    for sig in old:signal.signal(sig,interrupt)
    def monitor():
        while not stop.wait(.2):
            try:
                _budget(out)
                if time.monotonic()>=deadline:raise TimeoutError('BERT frozen compile wall allowance exhausted')
                if active[0] is not None:
                    memory=physical_memory_observation(_latest_rss_sample(active[0]),_available_ram_bytes(),rss_gib=14,minimum_available_gib=1.25)
                    if memory['stop']:
                        result['memory_guard_observation']=memory;raise RuntimeError('BERT compile physical memory guard')
            except BaseException as exc:
                reason=dict(cause='resource_guard',error=type(exc).__name__+': '+str(exc));failures.append(reason);result['termination']=reason
                try:write_json(out/'guard-observation.json',reason)
                finally:os.kill(os.getpid(),signal.SIGTERM)
                return
    watcher=threading.Thread(target=monitor,daemon=True)
    try:
        with recorder.span('complete_frozen_bert_trusted_compile'):
            result['source']=source_gate(case,args.checkpoint_audit,args.report_result)
            args.program=Path(result['source']['program']);args.model_info=Path(result['source']['model_info'])
            if args.profile.resolve()!=(paths.PROFILES_DIR/'qwen48/compiler-config.json').resolve():
                raise ValueError('Retain the original qwen48 compiler profile')
            if args.frontend.resolve()!=(paths.DACAPO_ROOT/'python/hecate/hecate/expr.py').resolve():
                raise ValueError('Retain the original trusted hecate expression frontend')
            template=_read(args.metadata_template)
            compiler_argv(template,'pars',binary='/compiler',source='/input',profile=args.profile,output='/lowered')
            for source,name in ((Path(__file__),'driver-source.py'),(args.program,'frozen-program.ptdsl'),
                (args.metadata_template,'recorded-compile-template.json'),(args.profile,'compiler-config.json')):shutil.copyfile(source,out/name)
            result['task_report_acceptance']=result['source']['report_task_acceptance'];_budget(out);watcher.start()
            for phase in PHASES:
                if failures:raise RuntimeError(failures[0]['error'])
                admission=physical_memory_observation(None,_available_ram_bytes(),rss_gib=14,minimum_available_gib=1.25)
                if admission['stop']:raise RuntimeError('BERT compile available RAM reserve prevents launch')
                _budget(out);argv=phase_command(args,phase,work);result['attempted'][phase]=1
                row=dict(phase=phase,argv=argv);result['processes'].append(row)
                with recorder.span(phase):
                    active[0]=out/(phase+'-resources.jsonl')
                    row['observation']=run_process(argv,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=out/(phase+'.log'),
                        resources=active[0],timeout=min(PHASE_SECONDS[phase]+120,max(.01,deadline-time.monotonic())),
                        mem_gib=12,fsize_gib=8)
                    active[0]=None
                    target=work/(phase+'-result.json')
                    if target.is_file():result['phases'][phase]=_read(target)
                    observed=row['observation'];worker_result=result['phases'].get(phase) or {}
                    if observed.get('signal') and not failures:result['termination']=dict(cause='incoming_signal',signal=observed['signal'])
                    if (failures or observed.get('exit_code')!=0 or observed.get('timed_out') or observed.get('signal')
                            or observed.get('remaining_pids') or worker_result.get('status')!='ok'
                            or worker_result.get('round')!=ROUND or worker_result.get('model')!=MODEL
                            or worker_result.get('mode')!=MODE or worker_result.get('phase')!=phase
                            or worker_result.get('source')!=result['source']
                            or any(worker_result.get(k)!=0 for k in ZERO_FIELDS)):
                        raise RuntimeError('Frozen BERT '+phase+' failed; subsequent phases were not started')
                    result['completed'][phase]=1
            result.update({k:result['phases']['compile'].get(k) for k in ('compiled_bootstraps','predicted_latency_us','runtime_plan','prediction')})
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        stop.set()
        if watcher.is_alive():watcher.join(timeout=3)
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        try:
            result['timing']=summarize_events(read_events(out/'events.jsonl'))
            result['recorded_pre_persistence_seconds']=(time.perf_counter_ns()-started)/1e9
            result['metrics']={k:result.get(k) for k in ('status','attempted','completed','compiled_bootstraps','predicted_latency_us',
                'task_report_acceptance','accuracy_acceptance','security_acceptance','CKKS_latency','termination')+ZERO_FIELDS}
            for _ in range(20):
                write_json(out/'result.json',result);size=artifact_bytes(out)
                if size+65536>CASE_BYTES:result.update(status='error',error='Terminal BERT compile artifacts exceed case allowance');result['metrics']['status']='error'
                if result['artifact_bytes']==size:break
                result['artifact_bytes']=size
            else:raise RuntimeError('Terminal BERT compile artifact size did not converge')
            from ..registry import git_rev
            row=dict(run_id=out.name,round=ROUND,model=MODEL,mode=MODE,git_rev=git_rev(),metrics=result['metrics'],artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
            with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker-phase',choices=PHASES)
    for name in ('source-case','checkpoint-audit'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--report-result',type=Path)
    for name in ('out','work-dir','model-info','program'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--intake-python',type=Path,default=paths.INTAKE_PYTHON)
    parser.add_argument('--profile',type=Path,default=paths.PROFILES_DIR/'qwen48/compiler-config.json')
    parser.add_argument('--frontend',type=Path,default=paths.DACAPO_ROOT/'python/hecate/hecate/expr.py')
    parser.add_argument('--metadata-template',type=Path,default=paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json')
    args=parser.parse_args(argv)
    if args.report_result is None:
        if args.worker_phase:parser.error('Inner phases require the explicit frozen report result path')
        args.report_result=Path(_read(args.source_case/'session/REPORT_ACCESS.json')['result'])
    for key in ('source_case','checkpoint_audit','report_result','out','work_dir','model_info','program','profile','frontend','metadata_template'):
        if getattr(args,key) is not None:setattr(args,key,getattr(args,key).resolve())
    if args.worker_phase and any(getattr(args,k) is None for k in ('work_dir','model_info','program')):parser.error('Inner phases require explicit original source paths')
    if not args.worker_phase and args.out is None:parser.error('Outer requires --out')
    result=worker(args) if args.worker_phase else execute(args)
    print(json.dumps(dict(status=result['status'],error=result['error'])))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
