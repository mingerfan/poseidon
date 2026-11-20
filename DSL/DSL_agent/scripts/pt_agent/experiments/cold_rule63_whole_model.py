"""Fresh original CNN -> training calibration -> fixed Rule63 -> native compile.

Native placement/search is inside compilation. Each invocation starts from the
original model and training samples; historical programs and clocks are unused.
The serial scheduler owns the heavy-work lease; this driver does not take it.
"""
import argparse
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

from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

ROUND='COLD-RULE63'
MODE='fresh-rule63-whole-model-native-compile'
MODELS=('resnet20_silu','vgg16_silu_avgpool','alexnet_silu_avgpool')
PHASES=('intake','calibration','generation','emit','trace','compile')
EXPECTED=dict(zip(MODELS,(230,276,204)))
GIB=1024**3
SIGNALS=(signal.SIGTERM,signal.SIGINT,signal.SIGALRM)


def read(path):
    return json.loads(Path(path).read_text())


def clean(value):
    return (value.get('exit_code')==0 and value.get('timed_out')is False
            and value.get('signal')is None and value.get('remaining_pids')==[]
            and value.get('signals_sent')==[])


def phase_command(args,phase,remaining):
    from .. import paths
    cap=7200 if phase=='compile' else 1800
    seconds=min(cap,int(remaining)-180)
    if seconds<=0:raise TimeoutError('Cold case allowance exhausted before '+phase)
    work=args.out/'work'/phase
    inner=['-B','-m',__spec__.name,'--worker-phase',phase,'--model',args.model,
        '--placement',args.placement,'--work-dir',str(work),'--case-root',str(args.out),
        '--model-info',str(args.out/'work/intake/model_info/model_info.json'),
        '--calibration',str(args.out/'work/calibration/calibration.json'),
        '--program',str(args.out/'work/generation/program.ptdsl'),'--emit-dir',str(args.out/'work/emit'),
        '--inputs',str(args.inputs),'--selections',str(args.selections),'--frontend',str(args.frontend),
        '--profile',str(args.profile),'--metadata-template',str(args.metadata_template),
        '--case-gib',str(args.case_gib),'--native-seconds',str(seconds)]
    if args.diagnostic_elide_final_ir:inner+=['--diagnostic-elide-final-ir']
    return ([sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(seconds+60),
             '--python',str(paths.INTAKE_PYTHON),'--',*inner],seconds+120)


def admission(args,*,created=False):
    """Prerequisite metadata only: no model, calibration, program or blob reads."""
    from .. import paths,toolchain
    from .r13_common_models import FACTORIES
    from .r29_local_compile_matrix import compiler_argv
    if args.out.is_symlink()or(args.out.exists()and not created):raise ValueError('Preserve existing cold case')
    if not args.out.is_relative_to(paths.RESULTS_DIR.resolve())or args.out==paths.RESULTS_DIR.resolve():
        raise ValueError('A fresh RESULTS_DIR child is required')
    factory='pt_agent.experiments.r13_common_models:'+args.model
    original=FACTORIES[factory]
    required=[args.inputs/'dataset.json',args.selections,args.frontend,args.profile,args.metadata_template,
              *(Path(original[k])for k in('source','checkpoint','inference_preprocessing_source'))]
    if any(not p.is_file()for p in required):raise FileNotFoundError('Missing original cold-chain prerequisite')
    host=toolchain.host_paths()
    if not Path(host['hecate_opt']).is_file():raise FileNotFoundError('Current compiler missing')
    config=read(args.profile)
    # Validate the existing metadata recipe and unchanged parameter group.
    compiler_argv(read(args.metadata_template),'dacapo-depth-dp'if args.placement=='depth'else'dacapo-optimized',
        binary=host['hecate_opt'],source=args.out/'work/emit/candidate_trace.mlir',profile=args.profile,
        output=args.out/'work/emit/lowered.mlir')
    expected=dict(bootstrapLevelLowerBound=1,bootstrapLevelUpperBound=5,levelLowerBound=1,
                  levelUpperBound=8,polynomialDegree=65536,rescalingFactor=43,runtime='HEAAN-HEVM')
    if any(config.get(key)!=value for key,value in expected.items()):
        raise ValueError('Original ours-N16-boot5 compiler parameters required')
    if shutil.disk_usage(args.out.parent).free<10*GIB:raise RuntimeError('resource_limit: free10GiB required')
    return dict(factory=factory,records=str(args.inputs),selections=str(args.selections),
        original_sources={k:str(original[k])for k in('source','checkpoint','inference_preprocessing_source')},
        compiler=str(host['hecate_opt']),compiler_profile=str(args.profile),
        compiler_metadata_recipe=str(args.metadata_template),checkpoint_values_read=0)


def snapshot(args):
    target=args.out/'source';target.mkdir()
    package=Path(__file__).parents[1]
    files=[Path(__file__),Path(__file__).with_name('cold_rule63_worker.py'),
        *[Path(__file__).with_name(name+'.py')for name in('r13_common_models','r13_common_worker',
        'r14_common_worker','r17_margin_validation_worker','r15_rule_validation_worker',
        'r29_local_compile_matrix','r30_common_model_e2e','r10_paper_metrics','completion_metrics')],
        package/'planners/rule.py']
    for source in files:shutil.copyfile(source,target/source.name)
    shutil.copyfile(args.profile,target/'compiler-profile.json')
    shutil.copyfile(args.metadata_template,target/'compile-metadata-template.json')


def append_registry(path,row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+')as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id')==row['run_id']for line in stream if line.strip()):
            raise ValueError('Preserve prior registry identity')
        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def classify(result):
    if result.get('guard_error'):return 'resource_limit'
    if result.get('incoming_signal')==signal.SIGALRM:return 'timeout'
    for phase in result['phases'].values():
        process=(phase or{}).get('native_process')or{}
        if process.get('timed_out'):return 'timeout'
        if process and not clean(process):return 'compiler_crash'
    for item in result['processes']:
        process=item['observation']
        if process.get('timed_out')or process.get('signal')==signal.SIGALRM:return 'timeout'
    error=result.get('error')or''
    if 'resource_limit' in error or 'MemoryError' in error:return 'resource_limit'
    if error.startswith('TimeoutError:'):return 'timeout'
    return 'ok'if result['status']=='ok'else'driver_error'


def stage_partition(timing,completion):
    """Disjoint outer worker windows; inner detail clocks never added twice."""
    groups={'intake_and_calibration':('source_preflight','intake','calibration'),
            'program_generation_and_count':('generation',),'emit_and_trace':('emit','trace'),
            'compile':('compile',)}
    spans={r['stage_id']:r for r in timing['spans']if r['stage_id']!='fresh_cold_rule63'}
    used=0;stages={};observed={}
    if not timing['attribution_ready']:return dict(stages_s={},observed_stages_s={},valid=False)
    for group,names in groups.items():
        rows=[spans[name]for name in names if name in spans]
        amount=sum(r['inclusive_ns']for r in rows)
        used+=amount;observed[group]=amount/1e9 if rows else None
        stages[group]=amount/1e9 if len(rows)==len(names)else None
    total=completion['complete_case_elapsed_ns']
    if used>total:raise ValueError('Observed exclusive phase windows exceed case clock')
    stages['placement']=None;stages['other']=(total-used)/1e9
    return dict(stages_s=stages,observed_stages_s=observed,valid=True,
        total_s=total/1e9,placement_in_compile=True,
        scope='Outer disjoint worker windows; placement/search included in compile; missing phases null',
        numeric_observed_stages_plus_other_s=used/1e9+stages['other'])


def peak_rss(args):
    maximum=0
    for phase in PHASES:
        path=args.out/'work'/phase/'resources.jsonl'
        if not path.is_file():continue
        with path.open()as stream:
            for line in stream:
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                if not isinstance(row.get('processes'),list):continue
                total=sum(p.get('vmrss_kib',0)for p in row['processes'])
                total+=(row.get('outer_process')or{}).get('vmrss_kib',0)
                maximum=max(maximum,total*1024)
    return maximum/GIB if maximum else None


def paper_record(args,result,completion,timing):
    from collections import Counter
    from .a_group_runs import empty_record
    compiled=result['phases'].get('compile')or{}
    process=compiled.get('native_process')or{}
    prediction=compiled.get('prediction')or{}
    partition=stage_partition(timing,completion)
    name={'resnet20_silu':'resnet20','vgg16_silu_avgpool':'vgg16','alexnet_silu_avgpool':'alexnet'}[args.model]
    row=empty_record(run_id=args.out.name,task='A2b'if args.placement=='depth'else('A5'if name=='vgg16'else'A6'),
        model=name,system='fixed_rule_depth'if args.placement=='depth'else'dacapo_search_ours',
        param_group='ours-N16-boot5',status=classify(result),status_detail=result.get('error'),budget_s=7200,
        compile_s=(process['ended_ns']-process['started_ns'])/1e9 if process.get('ended_ns')is not None else None,
        peak_rss_gib=peak_rss(args),boots_compiled=compiled.get('compiled_bootstraps'),
        exec_ops=compiled.get('execution_steps'),pred_latency_s=(compiled['predicted_latency_us']/1e6
            if compiled.get('predicted_latency_us')is not None else None),pred_table=str(args.profile),
        generation_e2e_s=completion['complete_case_elapsed_seconds'],stages_s=partition['stages_s'],
        api_calls=0,source=str(args.out/'result.json'),finished_utc=datetime.now(timezone.utc).isoformat())
    row.update(cold=True,historical_elapsed_reused=False,original_intake_reused=False,
        training_calibration_reused=False,complete_case_clock_source=str(args.out/'completion.json'),
        cold_stage_partition=partition,compile_scope='Native compiler child, including native placement/search',
        compile_stage_scope='Outer compile phase including launcher and actual artifact observation',
        expected_depth_bootstraps=EXPECTED[args.model]if args.placement=='depth'else None,
        expected_bootstraps_match=result.get('expected_bootstraps_match'),
        known_cost_partial_s=prediction.get('table_known_sum_us',0)/1e6 if prediction else None,
        missing_price_ops=dict(Counter(r['op']for r in prediction.get('missing',[]))),
        quality_retested=False,training_math_source=str(args.out/'work/generation/phase_result.json'),
        heldout_samples=0,confirm_samples=0,report_samples=0,ckks_inference_s=None,
        raw_Count_source=str(args.out/'work/generation/phase_result.json'),
        runtime_plan=compiled.get('runtime_plan'),acceptance_status=result['status'])
    row['execution_op_counts']=compiled.get('execution_ops')
    return row


def execute(args):
    started=time.perf_counter_ns()
    from .. import paths
    from ..registry import git_rev
    from .dacapo_compile_continuation import latest_resource,available_ram
    if args.out.exists()or args.out.is_symlink():raise ValueError('Preserve prior cold case')
    if not args.out.is_relative_to(paths.RESULTS_DIR.resolve())or args.out==paths.RESULTS_DIR.resolve():
        raise ValueError('Cold output must be a new RESULTS_DIR child')
    # Preconditions failing after entry still receive a terminal case and ledger row.
    args.out.mkdir(parents=True);(args.out/'work').mkdir()
    rec=StageRecorder(args.out/'events.jsonl')
    result=dict(round=ROUND,run_id=args.out.name,model=args.model,mode=MODE,placement=args.placement,
        status='running',error=None,phases={},processes=[],attempted={},completed={},api_calls=0,
        historical_elapsed_reused=False,original_intake_reused=False,training_calibration_reused=False,
        heldout_samples=0,confirm_samples=0,report_samples=0,real_ckks_runs=0,extra_integrity_checks=0,
        git_rev=git_rev(),completion_path=str(args.out/'completion.json'),guard_error=None,monitor_error=None,
        parameter_group='ours-N16-boot5',expected_bootstraps_match=None,monitor_stopped=False)
    stop=threading.Event();lock=threading.Lock();state=dict(active=False,resources=None)
    old={sig:signal.getsignal(sig)for sig in SIGNALS};oldtimer=signal.getitimer(signal.ITIMER_REAL)
    def interrupted(signum,_frame):
        result['incoming_signal']=signum
        raise InterruptedError('Cold pipeline interrupted by signal '+str(signum))
    def monitor():
        while not stop.wait(1):
            try:
                with lock:path=state['resources']
                sample=latest_resource(path);size=artifact_bytes(args.out)
                free=shutil.disk_usage(args.out).free;ram=available_ram()
                if (size>args.case_gib*GIB or free<10*GIB or ram<1.25*GIB
                        or sample and sample['subtree_plus_supervisor_rss_bytes']>14*GIB):
                    result['guard_observation']=dict(artifact_bytes=size,free_bytes=free,available_ram_bytes=ram,
                        simultaneous_subtree_supervisor_rss_bytes=(sample or{}).get('subtree_plus_supervisor_rss_bytes'),
                        limits=dict(case_bytes=args.case_gib*GIB,free_bytes=10*GIB,available_ram_bytes=1.25*GIB,rss_bytes=14*GIB))
                    raise RuntimeError('resource_limit: RSS14GiB / available1.25GiB / case / free10GiB guard')
            except BaseException as exc:
                key='guard_error'if str(exc).startswith('resource_limit:')else'monitor_error'
                result[key]=type(exc).__name__+': '+str(exc)
                with lock:
                    if state['active']:os.kill(os.getpid(),signal.SIGTERM)
                return
    watcher=threading.Thread(target=monitor,daemon=True)
    try:
        for sig in SIGNALS:signal.signal(sig,interrupted)
        signal.setitimer(signal.ITIMER_REAL,args.seconds)
        watcher.start()
        with rec.span('fresh_cold_rule63'):
            with rec.span('source_preflight'):
                result['source']=admission(args,created=True)
                snapshot(args)
                if available_ram()<1.25*GIB:
                    raise RuntimeError('resource_limit: available1.25GiB required before any worker')
            env={k:v for k,v in os.environ.items()if not k.startswith(('LD_','NIX_','__NIX_','HECATE_'))
                 and k not in('HECATE','IN_NIX_SHELL','PYTHONHOME','PYTHONPATH','VIRTUAL_ENV')}
            env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
            for name in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
            for phase in PHASES:
                if result['guard_error']:raise RuntimeError(result['guard_error'])
                if result['monitor_error']:raise RuntimeError(result['monitor_error'])
                work=args.out/'work'/phase;work.mkdir();(work/'tmp').mkdir()
                argv,timeout=phase_command(args,phase,args.seconds-(time.perf_counter_ns()-started)/1e9)
                result['attempted'][phase]=1;write_json(args.out/'progress.json',result)
                with rec.span(phase,command=argv):
                    with lock:state.update(active=True,resources=work/'resources.jsonl')
                    try:
                        observation=run_process(argv,cwd=paths.REPO/'DSL/DSL_agent/scripts',
                            log=work/'worker.log',resources=work/'resources.jsonl',timeout=timeout,
                            mem_gib=24,fsize_gib=16,env=dict(env,TMPDIR=str(work/'tmp')))
                    finally:
                        with lock:state['active']=False
                    result['processes'].append(dict(phase=phase,argv=argv,observation=observation))
                    value=read(work/'phase_result.json')if(work/'phase_result.json').is_file()else None
                    result['phases'][phase]=value
                    if not clean(observation)or not value or value.get('status')!='ok':
                        detail=(value or{}).get('error')or''
                        raise RuntimeError('Fresh '+phase+' failed: '+detail)
                    if result['guard_error']:raise RuntimeError(result['guard_error'])
                    if result['monitor_error']:raise RuntimeError(result['monitor_error'])
                    result['completed'][phase]=1
            compiled=result['phases']['compile']
            result.update(compiled_bootstraps=compiled['compiled_bootstraps'],
                execution_steps=compiled.get('execution_steps'),runtime_plan=compiled.get('runtime_plan'),
                predicted_latency_us=compiled.get('predicted_latency_us'))
            if args.placement=='depth':
                result['expected_bootstraps_match']=compiled['compiled_bootstraps']==EXPECTED[args.model]
                if not result['expected_bootstraps_match']:
                    raise ValueError('Actual compiled boot count differs from expected original depth control; preserved without adjustment')
            result['status']='ok'
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc)[:3000])
    finally:
        # Block late monitor/timer interrupts through every required persistence.
        for sig in SIGNALS:signal.signal(sig,signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL,0);stop.set()
        if watcher.is_alive():watcher.join(timeout=3)
        result['monitor_stopped']=not watcher.is_alive()
        if result['guard_error']:
            result.update(status='error',error=result['guard_error'])
        if result['monitor_error']:
            result.update(status='error',error=result['monitor_error'])
        if not result['monitor_stopped']:
            result.update(status='error',error='Monitor did not stop before terminal persistence')
        timing=summarize_events(read_events(args.out/'events.jsonl'),termination_observed_ns=time.perf_counter_ns())
        write_json(args.out/'timing.json',timing)
        result['recorded_elapsed_seconds']=(time.perf_counter_ns()-started)/1e9
        result['artifact_bytes']=0
        for _ in range(12):
            write_json(args.out/'result.json',result);size=artifact_bytes(args.out)
            if result['artifact_bytes']==size:break
            result['artifact_bytes']=size
        else:raise RuntimeError('Terminal case size did not stabilize')
        append_registry(paths.RESULTS_DIR/'registry.jsonl',dict(run_id=args.out.name,round=ROUND,model=args.model,
            mode=MODE,git_rev=result['git_rev'],metrics=dict(status=result['status'],attempted=result['attempted'],
            completed=result['completed'],api_calls=0,completion_path=result['completion_path']),
            artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat()))
        completion=persist_case_completion(args.out,round_label=ROUND,started_ns=started,status=result['status'],
            timing=timing,required_artifact_bytes=result['artifact_bytes'])
        row=paper_record(args,result,completion,timing);write_json(args.out/'paper-record.json',row)
        if args.paper_runs is not None:
            from .a_group_runs import append_record
            append_record(args.paper_runs,row)
        if result['monitor_stopped']:
            for sig,handler in old.items():signal.signal(sig,handler)
            if oldtimer[0]>0:signal.setitimer(signal.ITIMER_REAL,max(.001,oldtimer[0]-(time.perf_counter_ns()-started)/1e9),oldtimer[1])
    return result


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--placement',choices=('depth','optimized'),required=True)
    parser.add_argument('--worker-phase',choices=PHASES)
    for name in('out','inputs','selections','work-dir','case-root','model-info','calibration','program','emit-dir',
                'profile','frontend','metadata-template','paper-runs'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--seconds',type=int,default=12000)
    parser.add_argument('--native-seconds',type=int,default=7200)
    parser.add_argument('--case-gib',type=int,default=64)
    parser.add_argument('--diagnostic-elide-final-ir',action='store_true')
    args=parser.parse_args(argv)
    if not(180<args.seconds<=12000 and 0<args.native_seconds<=7200 and 16<=args.case_gib<=64):
        parser.error('Use finite case<=12000s, native<=7200s, case16..64GiB')
    if args.worker_phase:
        required=('inputs','selections','work_dir','case_root','model_info','calibration','program','emit_dir',
                  'profile','frontend','metadata_template')
        if any(getattr(args,k)is None for k in required):parser.error('All inner paths are explicit argv')
    else:
        from .. import paths
        if args.out is None:parser.error('--out required')
        defaults=dict(inputs=paths.WORK_RESULTS/'agent-cifar-offline-records-r1804/cifar10-records',
            selections=paths.WORK_RESULTS/'agent-resnet20-original-accuracy-r1805/frozen-selections.json',
            profile=paths.PROFILES_DIR/'qwen48/compiler-config.json',frontend=paths.DACAPO_ROOT/'python/hecate/hecate/expr.py',
            metadata_template=paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json')
        for key,default in defaults.items():setattr(args,key,(getattr(args,key)or default).expanduser().resolve())
        args.out=args.out.expanduser().resolve()
    return args


def main(argv=None):
    args=arguments(argv)
    if args.worker_phase:
        from .cold_rule63_worker import run
        return run(args)
    return int(execute(args)['status']!='ok')


if __name__=='__main__':raise SystemExit(main())
