"""A2: frozen CNN re-placement and pars export under its original DaCapo profile.

Existing R30 quality is inherited. Planning, Count/Emit, trace and compiler work
belong to this finite warm reprofile invocation. No API, sample forward, quality
selection, key construction or CKKS execution is performed. Four-candidate
scores are conservative frontend proxies; effective15 reserves one logical
post-Boot level and rescale16 cost, without asserting an extra physical
PARS operation or native global optimality. Actual RuntimePlan sums provide
the final table predictions.
"""
import argparse
import ast
from datetime import datetime,timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

from .r10_paper_metrics import StageRecorder,read_events,summarize_events,run_process,write_json
from .completion_metrics import artifact_bytes,persist_case_completion

ROUND='A2-NATIVE-PROFILE'
MODE='frozen-program-native-profile-four-placements-pars'
GIB=1024**3
PHASES=('planning','emit','trace','compile','observe')
MODELS=('resnet20_silu','vgg16_silu_avgpool','alexnet_silu_avgpool')


def read(path):return json.loads(Path(path).read_text())


def require(condition,message):
    if not condition:raise ValueError(message)


def reprofile_source(text):
    """Only supported system header literals change; all mathematical AST stays."""
    tree=ast.parse(text);matches=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='fhe'and node.func.attr=='program'):matches.append(node)
    require(len(matches)==1,'Exactly one original fhe.program header required')
    header=matches[0];values={item.arg:item for item in header.keywords}
    require(set(('l_max','boot_out_level','refresh','placement')).issubset(values),'Original managed header is incomplete')
    require(ast.literal_eval(values['refresh'].value)=='agent'and ast.literal_eval(values['placement'].value)=='program-plan','Original frozen program-plan header required')
    values['l_max'].value=ast.Constant(29);values['boot_out_level'].value=ast.Constant(16)
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def compiler_argv(template,source,profile,output):
    """Reuse every original RuntimePlan identifier; compute no file digest."""
    require(isinstance(template,list)and all(isinstance(arg,str)for arg in template),'Saved native argv required')
    require(template.count('--dacapo')==1 and template.count('--boot-placement=optimized')==1
        and template.count('--waterline=40')==1 and template.count('-o')==1,'Original native search argv changed')
    require(sum(arg.startswith('--runtime-plan-operator-spec-sha256=')for arg in template)==1,'Existing operator identifier required')
    argv=[arg for arg in template if arg not in('--dacapo','--boot-placement=optimized')]
    argv[1]=str(source);argv.insert(2,'--pars')
    index=next(i for i,arg in enumerate(argv)if arg.startswith('--ckks-config='));argv[index]='--ckks-config='+str(profile)
    argv[argv.index('-o')+1]=str(output)
    require([a for a in argv if a.startswith('--runtime-plan-')]==[a for a in template if a.startswith('--runtime-plan-')],
        'Original RuntimePlan parameters changed')
    return argv


def observe_runtime(plan,profile,expected_boots):
    from .r29_local_compile_matrix import plan_observation
    observed=plan_observation(plan,profile);values={row['id']:row for row in plan['values']}
    inputs=plan['external_inputs'];input_ids=[row['value']if isinstance(row,dict)else row for row in inputs]
    require(input_ids and all(values[nid]['level']==16 for nid in input_ids),'Actual pars RuntimePlan inputs must enter at16')
    require(all(type(row.get('level'))is int and 0<=row['level']<=29 for row in values.values()),'Actual pars RuntimePlan values must retain native level0..29 metadata')
    require(observed['compiled_bootstraps']==expected_boots,'Compiled bootstrap count differs from selected placement')
    require(all(row['input_levels']==[3]and row['output_levels']==[16]for row in observed['bootstrap_positions']),
        'Actual bootstrap must retain native3->16 levels')
    return observed,dict(input_levels=[values[nid]['level']for nid in input_ids],max_level=max(row['level']for row in values.values()),profile_upper_bound=29)


def _program(args):
    from ..models.info import ModelInfo
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from .r14_common_worker import check_program_signature
    info=ModelInfo.load(args.model_info);program=parse((args.out/'program.ptdsl').read_text())
    require(info.source.get('factory')=='pt_agent.experiments.r13_common_models:'+args.model,'Original model factory differs')
    checked=check(program,{key:value.shape for key,value in info.graph.weights.items()},graph=info.graph)
    require(checked.ok and check_program_signature(info,program)['passed']is True,'Original frozen interface/typecheck failed')
    return info,program


def worker(args):
    for key in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    from . import a2_native_profile_planner as planner
    from . import a2_native_profile_runtime as runtime
    from ..lower import lower
    from ..backends import EmitBackend
    events=StageRecorder(args.phase_dir/'events.jsonl');result=dict(phase=args.worker,status='running',error=None)
    try:
        with events.span(args.worker):
            model=planner.model_for(read(args.profile))
            if args.worker in('planning','emit'):
                info,program=_program(args);run=lambda active:lower(program,info.graph.weights,active,act_mode='program')
                if args.worker=='planning':
                    with events.span('four_placement_algorithms'):
                        plans,summary,selected=runtime.plan(run,program.header.slots,model,time_limit=args.planning_seconds,state_limit=args.state_limit)
                    with events.span('four_complete_Count_and_Emit_replays'):
                        proof=runtime.validate_all(run,program.header.slots,model,plans,frontend=str(args.frontend))
                    for name,plan in plans.items():write_json(args.out/'plans'/(name+'.json'),plan)
                    write_json(args.out/'placement.json',dict(candidates=summary,selected=selected,proofs=proof))
                    result.update(candidates=summary,selected=selected,all_four_verified=True)
                else:
                    chosen=read(args.out/'placement.json')['selected'];plan=read(args.out/'plans'/(chosen+'.json'))
                    emitter=EmitBackend(program.header.slots,frontend=args.frontend,constants=args.out/'emit/constants.npz',save_dir=args.out/'emit')
                    stats,proof=runtime.run_checked(run,program.header.slots,model,plan,emitter)
                    emitter.write(args.out/'emit');result.update(counts=stats['counts'],proof=proof,selected=chosen)
            elif args.worker in('trace','compile'):
                from .. import toolchain
                toolchain.setup_env();host=toolchain.host_paths()
                env=dict(os.environ,LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'],
                    HECATE=str(host['hecate']),HECATE_BUILD=str(host['hecate_build']))
                if args.worker=='trace':
                    argv=[sys.executable,'-B',str(args.out/'emit/candidate_trace.py')];seconds=1800
                else:
                    checked=read(args.out/'admission.json');argv=compiler_argv(checked['template'],
                        args.out/'emit/candidate_trace.mlir',args.profile,args.out/'emit/lowered.mlir');seconds=7200
                    write_json(args.out/'compile-argv.json',argv)
                write_json(args.phase_dir/'native-argv.json',argv)
                observation=run_process(argv,cwd=args.out/'emit',log=args.phase_dir/'native.log',
                    resources=args.phase_dir/'native-resources.jsonl',timeout=seconds,mem_gib=24,fsize_gib=16,env=env)
                result['native_process']=observation
                require(observation['exit_code']==0 and not observation['timed_out']and observation['signal']is None
                    and observation['remaining_pids']==[]and observation['signals_sent']==[],'A2 native '+args.worker+' failed')
                if args.worker=='trace':require((args.out/'emit/candidate_trace.mlir').is_file(),'Trace did not produce MLIR')
            elif args.worker=='observe':
                from .r29_local_compile_matrix import plan_observation
                plans=list((args.out/'emit').glob('lowered.*.runtime-plan.json'));require(len(plans)==1,'One actual compiled RuntimePlan required')
                compiled_plan=read(plans[0]);selected=read(args.out/'placement.json');expected=selected['candidates'][selected['selected']]['boots']
                ours,level_audit=observe_runtime(compiled_plan,read(args.profile),expected)
                result['native_level_audit']=level_audit
                native=plan_observation(read(args.native_plan),read(args.profile))
                result.update(runtime_plan=str(plans[0]),compiled=ours,native_reference=native,
                    prediction_scope='Same unchanged original external HEAAN-HEVM latencyTable for both compiled execution streams; no measured CKKS latency')
            else:raise ValueError('Trace/compile are supervised directly by the outer driver')
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc)[:4000])
    finally:
        result['timing']=summarize_events(read_events(args.phase_dir/'events.jsonl'),time.perf_counter_ns())
        write_json(args.phase_dir/'result.json',result);gc.collect()
    return result


def admission(args):
    source=read(args.source_case/'result.json');audit=read(args.source_case/'post-run-quality-audit.json')
    require(source.get('status')=='ok'and source.get('model')==args.model,'Original successful R30 source required')
    require(audit.get('status')=='quality_accepted'and all(audit.get(k)is True for k in('formal_frozen','quality_acceptance_passed','strict_math_passed')),'Original frozen quality proof required')
    program=args.source_case/'work/agent/program.ptdsl';require(Path(audit['selected_program']).resolve()==program.resolve(),'Original selected program path differs')
    original_text=program.read_text();worker_record=read(args.source_case/'work/agent/worker-result.json')
    selected=worker_record.get('agent',{}).get('confirm',{}).get('selected',{})
    require(worker_record.get('frozen')is True and isinstance(selected.get('source'),str)
        and ast.dump(ast.parse(original_text))==ast.dump(ast.parse(selected['source'])),
        'Saved original confirmed source differs from the frozen program')
    native=read(args.native_case/'result.json');require(native.get('status')=='ok','Successful original native reference required')
    checkpoint=source.get('source',{}).get('original_sources',{}).get('checkpoint')
    native_checkpoint=native.get('sources',{}).get('checkpoint')
    if isinstance(native_checkpoint,dict):native_checkpoint=native_checkpoint.get('path')
    require(checkpoint and native_checkpoint and Path(checkpoint).resolve()==Path(native_checkpoint).resolve(),'Native reference and original frozen graph checkpoint identities differ')
    expected={'resnet20_silu':'dacapo-original-resnet20-silu','vgg16_silu_avgpool':'dacapo-original-vgg16-silu-avgpool','alexnet_silu_avgpool':'dacapo-original-alexnet-silu-avgpool'}
    require(native.get('model')==expected[args.model],'Native reference model differs')
    profile=args.native_case/'compiler-config.json';template=read(args.native_case/'compile-argv.json')
    from .a2_native_profile_planner import model_for
    model_for(read(profile));require(Path(template[0]).is_file(),'Existing original hecate-opt unavailable')
    require(next(arg for arg in template if arg.startswith('--ckks-config='))=='--ckks-config='+str(profile),'Original config argv identity differs')
    plans=list((args.native_case/'work/compile').glob('lowered.*.runtime-plan.json'));require(len(plans)==1,'Original unique RuntimePlan required')
    info=args.source_case/'work/intake/model_info/model_info.json';require(info.is_file()and program.is_file(),'Frozen original graph/program missing')
    return dict(model_info=str(info),program=str(program),profile=str(profile),native_plan=str(plans[0]),
        template=template,source_quality=audit,source_case=str(args.source_case),native_case=str(args.native_case),
        original_r30_bootstraps=source['compiled_bootstraps'],native_original_bootstraps=native['bootstrap_count'])


def _append(path,row):
    with Path(path).open('a')as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def execute(args):
    from .. import paths
    from .heir_initializer_probe import durable_json
    from .r27_orbit_lazy_backend import _latest_rss_sample,_available_ram_bytes
    require(args.run,'Actual A2 execution requires --run')
    root=paths.RESULTS_DIR.resolve();require(not args.out.exists()and args.out.is_relative_to(root)and args.out!=root,'Fresh owned output required')
    args.out.mkdir();(args.out/'plans').mkdir();(args.out/'emit').mkdir();(args.out/'phases').mkdir()
    started=time.perf_counter_ns();events=StageRecorder(args.out/'events.jsonl');active=[None];stop=threading.Event();failure=[]
    result=dict(round=ROUND,model=args.model,mode=MODE,status='running',error=None,phases=[],attempted={},completed={},
        param_group='dacapo-native',api_calls=0,model_forwards=0,sample_reads=0,keygen_runs=0,real_ckks_attempts=0,
        quality_inherited=True,generation_e2e_s=None,cold_end_to_end_s=None,compiled_bootstraps=None,pred_latency_s=None,
        limits=dict(native_seconds=7200,process_AS_gib=24,subtree_RSS_gib=14,case_gib=args.case_gib,file_gib=16,minimum_free_gib=10,concurrency=1))
    def check():
        if failure:raise RuntimeError(failure[0])
        sample=_latest_rss_sample(active[0])if active[0]and active[0].exists()else None
        if sample and sample['subtree_rss_bytes']>14*GIB:raise RuntimeError('resource_limit: subtree_RSS14GiB')
        if _available_ram_bytes()<1.25*GIB:raise RuntimeError('resource_limit: host_available1.25GiB')
        if shutil.disk_usage(args.out).free<10*GIB:raise RuntimeError('resource_limit: free_disk10GiB')
        if artifact_bytes(args.out)>args.case_gib*GIB-GIB:raise RuntimeError('resource_limit: case artifact reserve')
    def monitor():
        while not stop.wait(.2):
            try:check()
            except BaseException as exc:
                failure.append(type(exc).__name__+': '+str(exc));result['guard_error']=failure[-1]
                os.kill(os.getpid(),signal.SIGTERM);return
    watcher=threading.Thread(target=monitor,daemon=True);old={sig:signal.getsignal(sig)for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    def interrupt(signum,frame):result['incoming_signal']=signum;raise InterruptedError('A2 supervisor received signal'+str(signum))
    for sig in old:signal.signal(sig,interrupt)
    try:
        with events.span('warm_frozen_native_profile_recompile'):
            checked=admission(args);result['source']=checked;durable_json(args.out/'source.json',checked)
            (args.out/'program.ptdsl').write_text(reprofile_source(Path(checked['program']).read_text()))
            metadata=dict(checked);write_json(args.out/'admission.json',metadata);watcher.start()
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
            for key in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
            for phase in PHASES:
                check();directory=args.out/'phases'/phase;directory.mkdir();active[0]=directory/'resources.jsonl'
                seconds=args.planning_seconds+900 if phase=='planning'else 7200 if phase=='compile'else 1800
                inner=['-B','-m','pt_agent.experiments.a2_native_profile_driver','--worker',phase,'--model',args.model,
                    '--out',str(args.out),'--phase-dir',str(directory),'--model-info',checked['model_info'],
                    '--profile',checked['profile'],'--native-plan',checked['native_plan'],'--frontend',str(args.frontend),
                    '--planning-seconds',str(args.planning_seconds),'--state-limit',str(args.state_limit)]
                argv=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(seconds+60),
                    '--python',str(paths.INTAKE_PYTHON),'--',*inner]
                row=dict(phase=phase,argv=argv,status='running');result['phases'].append(row);result['attempted'][phase]=1;durable_json(args.out/'result.json',result)
                with events.span(phase):
                    observation=run_process(argv,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=directory/'worker.log',
                        resources=active[0],timeout=seconds+120,mem_gib=24,fsize_gib=16,env=env)
                    row['process']=observation
                    if (directory/'result.json').exists():row['worker']=read(directory/'result.json')
                    require(observation['exit_code']==0 and not observation['timed_out']and observation['signal']is None and observation['remaining_pids']==[]and observation['signals_sent']==[],'A2 '+phase+' process failed')
                    require(row.get('worker',{}).get('status')=='ok','A2 '+phase+' worker failed: '+str(row.get('worker',{}).get('error')))
                    if phase=='trace':require((args.out/'emit/candidate_trace.mlir').is_file(),'Trace did not produce MLIR')
                    row['status']='ok';result['completed'][phase]=1
                active[0]=None
            observe=result['phases'][-1]['worker'];result.update(status='ok',compiled_bootstraps=observe['compiled']['compiled_bootstraps'],
                pred_latency_s=None if observe['compiled']['predicted_latency_us']is None else observe['compiled']['predicted_latency_us']/1e6,
                native_reference_pred_latency_s=None if observe['native_reference']['predicted_latency_us']is None else observe['native_reference']['predicted_latency_us']/1e6)
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc)[:4000])
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL,0);stop.set()
        if watcher.is_alive():watcher.join(timeout=3)
        result['monitor_stopped']=not watcher.is_alive()
        if failure or not result['monitor_stopped']:result.update(status='error',error=failure[0]if failure else'A2 monitor did not close')
        if result['status']!='ok':
            last=result['phases'][-1]if result['phases']else{}
            observation=last.get('worker',{}).get('native_process')or last.get('process')or{}
            result['failure_class']=('resource_limit'if failure or 'resource_limit:'in(result.get('error')or'') else'timeout'if observation.get('timed_out')
                else'compiler_crash'if last.get('phase')=='compile'and observation.get('exit_code')not in(None,0)
                else'driver_error')
        result['timing']=summarize_events(read_events(args.out/'events.jsonl'),time.perf_counter_ns())
        native_compile=next((row.get('worker',{}).get('native_process')for row in result['phases']if row['phase']=='compile'),None)
        result['compile_s']=None if not native_compile else(native_compile['ended_ns']-native_compile['started_ns'])/1e9
        result['pred_table']='DaCapo original HEAAN-HEVM latencyTable'
        durable_json(args.out/'result.json',result)
        _append(root/'registry.jsonl',dict(run_id=args.out.name,round=ROUND,model=args.model,mode=MODE,git_rev=None,
            metrics={k:result[k]for k in('status','attempted','completed','compiled_bootstraps','pred_latency_s')},artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat()))
        marker=persist_case_completion(args.out,round_label=ROUND,started_ns=started,status=result['status'],timing=result['timing'],required_artifact_bytes=artifact_bytes(args.out))
        result['complete_case_seconds']=marker['complete_case_elapsed_seconds']
        if result['monitor_stopped']:
            for sig,handler in old.items():signal.signal(sig,handler)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--worker',choices=PHASES)
    for key in('source-case','native-case','out','phase-dir','model-info','profile','native-plan','frontend'):p.add_argument('--'+key,type=Path)
    p.add_argument('--planning-seconds',type=int,default=600);p.add_argument('--state-limit',type=int,default=4096);p.add_argument('--case-gib',type=int,default=64);p.add_argument('--run',action='store_true')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().resolve())
    if args.frontend is None:
        from .. import paths
        args.frontend=paths.DACAPO_ROOT/'python/hecate/hecate/expr.py'
    result=worker(args)if args.worker else execute(args)
    print(json.dumps({k:result.get(k)for k in('status','error','compiled_bootstraps')}));return int(result['status']!='ok')


if __name__=='__main__':raise SystemExit(main())
