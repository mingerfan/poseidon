"""Compile a C2-frozen BERT after calibration-only math, with report deferred.

No original report marker or report payload is opened. Original numerical
implementations and compiler parameters are delegated under a new source gate.
The outer scheduler owns the shared heavy job order and absolute deadline.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import fcntl
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import textwrap
import threading
import time
import types
from types import SimpleNamespace

from .. import paths
from . import bert_serial_continuation as serial
from . import bert_frozen_compile as original
from .r10_paper_metrics import StageRecorder,run_process,read_events,summarize_events,write_json
from .completion_metrics import artifact_bytes,persist_case_completion

ROUND='BERT-COMPILE-FIRST-DEFERRED-REPORT'
MODEL=serial.MODEL
MODE='original-c2-calibration3-math-trusted-compile-deferred-report'
MODULE='pt_agent.experiments.bert_compile_first_deferred_report'
GIB=1024**3
CASE_BYTES=32*GIB
RESERVE=1024**2
OVERALL_SECONDS=12000
MATH_SECONDS=900
COUNTERS=('graph_reference','tensor_exact','mock_exact','tensor_program','mock_program')
ZERO=('new_agent_steps','new_tune_commands','new_confirmation_commands','api_calls','report_samples',
      'heldout_samples','confirm_samples','native_test_reads','new_tokenizer_calls','downloads','installations','integrity_checks')


def read(path):return json.loads(Path(path).read_text())
def save(path,record):serial.durable_json(path,record)


def compiler_metadata(args):
    if (args.profile!=paths.PROFILES_DIR/'qwen48/compiler-config.json'
            or args.frontend!=paths.DACAPO_ROOT/'python/hecate/hecate/expr.py'):
        raise ValueError('Preserve original qwen48 profile and trusted frontend')
    template=read(args.metadata_template)
    original.compiler_argv(template,'pars',binary='/compiler',source='/input',profile=args.profile,output='/lowered')
    return dict(profile=str(args.profile),frontend=str(args.frontend),metadata_template=str(args.metadata_template),
                profile_metadata=read(args.profile),recorded_argv=template)


def metadata_source(case,checkpoint_audit):
    """Only explicit retained metadata/stat; no dataset/report/weight read."""
    case=Path(case).resolve();checkpoint_audit=Path(checkpoint_audit).resolve()
    outcome=serial.frozen_state(case)
    if outcome.get('frozen') is not True:raise ValueError('Complete accepted original C2 freezing is required')
    info_path=case/'model/model_info.json';info=read(info_path);src=info.get('source') or {}
    from .bert_checkpoint_audit import source_gate as checkpoint_source
    checkpoint=checkpoint_source(src['local_path'],case/'phases/001-prepare/worker_result.json',info_path)
    split=info.get('split') or {};samples=read(case/'model/samples/source.json')
    if (split.get('kind')!='sst2_native' or split.get('version')!='sst2-native-v1' or split.get('seed')!=20261024
            or samples.get('partition')!=split or samples.get('native_test_read') is not False
            or not serial.same(samples.get('source'),src.get('inputs_path'))
            or any(split.get(key)!=list(range(start,start+count)) for key,start,count in
                [('calibration',0,1000),('heldout',1000,1000),('confirm',2000,1000),('report',3000,872)])):
        raise ValueError('Retain the exact original trained BERT/SST2 partition and input provenance')
    audited=serial.accepted_checkpoint(SimpleNamespace(checkpoint_out=checkpoint_audit.parent),
        dict(model_info=str(info_path),model=checkpoint['model']))
    calibration=case/'model/samples/calibration.npz'
    if not calibration.is_file() or calibration.stat().st_size<=0:raise ValueError('Original saved calibration inputs are required')
    return dict(source_case=str(case),model_info=str(info_path),program=outcome['program'],session=str(case/'session'),
        original_model=checkpoint['model'],checkpoint_audit=str(checkpoint_audit),original_checkpoint=checkpoint,
        frozen=outcome,session_state=serial.session_state(case),
        selected_source=Path(outcome['program']).read_text(),inputs=info['inputs'],sequence=info['sequence'],
        task=info['task'],metric=info['metric'],task_policy=info['threshold'],split_version=split['version'],
        split=split,sample_source=samples,calibration=dict(path=str(calibration),bytes=calibration.stat().st_size,
            partition='calibration',row_positions=[0,1,2],sample_indices=split['calibration'][:3],stored_rows=1000),
        semantic_checkpoint_accepted=audited['checkpoint_semantic_coverage_passed'],
        report_task_acceptance=None,report_validation_metrics=None,report_deferred=True,
        report_outcome_used_for_selection=False,api_feedback_permitted=False)


def _math_api():
    from .bert_sst2_worker import initialize_numeric,strict_math
    initialize_numeric()
    import numpy as np
    from ..models.info import ModelInfo
    from ..dsl.parser import parse
    from ..dsl.interface import require_interface
    from ..dsl.typecheck import check
    from ..analyzer import reference_forward
    from ..transformer_validation import execute
    return dict(np=np,load=ModelInfo.load,parse=parse,interface=require_interface,check=check,
                reference=reference_forward,execute=execute,strict=strict_math)


def math_worker(args):
    work=args.work_dir;work.mkdir(parents=True,exist_ok=False);target=work/'math-result.json'
    recorder=StageRecorder(work/'math-events.jsonl')
    result=dict(round=ROUND,model=MODEL,mode=MODE,phase='math',status='running',error=None,math_passed=False,
        attempted=dict.fromkeys(COUNTERS,0),completed=dict.fromkeys(COUNTERS,0),strict_checks_attempted=0,
        strict_checks_completed=0,frozen_math=[],**dict.fromkeys(ZERO,0))
    try:
        with recorder.span('calibration_only_frozen_math'):
            result['source']=metadata_source(args.source_case,args.checkpoint_audit);save(target,result)
            result['compiler_inputs']=compiler_metadata(args)
            # Install existing identifier metadata before all cost/numeric imports.
            with original.native_metadata_commands(read(args.metadata_template)):
                api=_math_api();np=api['np'];source=result['source']
                with recorder.span('original_model_info_and_program'):
                    info=api['load'](source['model_info']);program=api['parse'](source['selected_source'])
                    api['interface'](program,info)
                    checked=api['check'](program,{key:value.shape for key,value in info.graph.weights.items()},
                                         graph=info.graph,prompt_version='r8-v11')
                    if not checked.ok:raise ValueError('Original frozen program no longer typechecks')
                with recorder.span('saved_first3_calibration_inputs'):
                    with np.load(source['calibration']['path'],allow_pickle=False) as archive:
                        arrays={name:np.array(archive[name][:3]) for name in info.graph.inputs}
                    if set(arrays)!={param.name for param in program.params}:
                        raise ValueError('Saved calibration inputs and original program signature differ')
                    for name,value in arrays.items():
                        if (value.dtype!=np.float64 or value.shape!=(3,*info.graph.tensors[name].shape[1:])
                                or not np.isfinite(value).all()):
                            raise ValueError('Require finite original float64 first3 input rows and exact inferred shapes')
                    result['probe_source']=source['calibration']
                    result['input_metadata']={name:dict(shape=list(value.shape),dtype=str(value.dtype)) for name,value in arrays.items()}
                    result['calibration_probes']=3;result['storage_note']='NPZ may decompress complete saved calibration members before slicing; only3 logical samples are evaluated'
                for index in range(3):
                    values={name:value[index:index+1] for name,value in arrays.items()};outputs={}
                    row=dict(sample=index,sample_index=source['calibration']['sample_indices'][index],checks=[])
                    result['frozen_math'].append(row)
                    for kind in COUNTERS:
                        result['attempted'][kind]+=1;save(target,result)
                        with recorder.span(kind,probe=index):
                            if kind=='graph_reference':out=api['reference'](info.graph,values)[info.graph.outputs[0]]
                            else:out=api['execute'](program,info.graph.weights,values,
                                engine='tensor' if kind.startswith('tensor') else 'mock',exact=kind.endswith('exact'))
                        outputs[kind]=out;result['completed'][kind]+=1;save(target,result)
                    for ref,observed in [('graph_reference','tensor_exact'),('graph_reference','mock_exact'),('tensor_program','mock_program')]:
                        result['strict_checks_attempted']+=1;save(target,result)
                        measured=api['strict'](outputs[ref],outputs[observed]);row['checks'].append(dict(reference=ref,observed=observed,**measured))
                        result['strict_checks_completed']+=1;save(target,result)
                        if measured['passed'] is not True:raise ValueError('Original frozen calibration mathematics failed')
                result['source_recheck']=metadata_source(args.source_case,args.checkpoint_audit)
                if result['source_recheck']!=source:raise ValueError('Original frozen source changed during mathematics')
                result.update(status='ok',math_passed=True)
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    result['timing']=summarize_events(read_events(recorder.path));save(target,result)
    return result


def compile_source(case,checkpoint_audit,math_result):
    source=metadata_source(case,checkpoint_audit);proof=read(math_result)
    rows=proof.get('frozen_math') or []
    if (proof.get('round')!=ROUND or proof.get('model')!=MODEL or proof.get('mode')!=MODE or proof.get('phase')!='math'
            or proof.get('status')!='ok' or proof.get('math_passed') is not True or proof.get('source')!=source
            or proof.get('source_recheck')!=source or proof.get('probe_source')!=source['calibration']
            or proof.get('attempted')!=dict.fromkeys(COUNTERS,3) or proof.get('completed')!=dict.fromkeys(COUNTERS,3)
            or proof.get('strict_checks_attempted')!=9 or proof.get('strict_checks_completed')!=9
            or proof.get('calibration_probes')!=3 or not isinstance(proof.get('compiler_inputs'),dict)
            or any(proof.get(key)!=0 for key in ZERO)
            or len(rows)!=3 or [row.get('sample') for row in rows]!=[0,1,2]
            or [row.get('sample_index') for row in rows]!=source['calibration']['sample_indices']
            or any(len(row.get('checks',[]))!=3 or
                [(check.get('reference'),check.get('observed')) for check in row['checks']]!=
                    [('graph_reference','tensor_exact'),('graph_reference','mock_exact'),('tensor_program','mock_program')]
                or any(check.get('passed') is not True or check.get('criterion')!=original.CRITERION
                    or any(type(check.get(key)) not in (int,float) or not math.isfinite(check[key]) or check[key]<0
                           for key in ('max_absolute_error','max_tolerance_ratio'))
                    or check['max_tolerance_ratio']>1 for check in row['checks']) for row in rows)):
        raise ValueError('Require complete same-source calibration3/9checks before every compiler phase')
    return dict(source,math_result=str(Path(math_result).resolve()),mathematical_groups_passed=3,compiler_inputs=proof['compiler_inputs'],
                report_result=None,report_samples=0,math_scope='Fixed saved calibration rows0..2 only')


def _math_argument(function):
    """Change one trusted argument attribute, retaining the numerical body."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    found=[node for node in ast.walk(tree) if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name)
           and node.value.id=='args' and node.attr=='report_result']
    if len(found)!=1:raise ValueError('Original compiler argument seam changed')
    found[0].attr='math_result';ast.fix_missing_locations(tree)
    return tree


def compile_namespace(seconds=original.OVERALL_SECONDS):
    if type(seconds) is not int or not 1<=seconds<=OVERALL_SECONDS:
        raise ValueError('Use finite remaining compile allowance1..12000seconds')
    values=dict(vars(original),ROUND=ROUND,MODEL=MODEL,MODE=MODE,CASE_BYTES=CASE_BYTES,
                OVERALL_SECONDS=min(original.OVERALL_SECONDS,seconds))
    for name,function in vars(original).items():
        if isinstance(function,types.FunctionType) and function.__globals__ is vars(original):
            clone=types.FunctionType(function.__code__,values,name,function.__defaults__,function.__closure__)
            clone.__kwdefaults__=copy.copy(function.__kwdefaults__);values[name]=clone
    for name in ('worker','execute'):
        exec(compile(_math_argument(getattr(original,name)),original.__file__,'exec'),values)
    def phase_command(args,phase,work):
        argv=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(original.PHASE_SECONDS[phase]+60),
            '--python',str(args.intake_python),'--','-B','-m',MODULE,'--worker-phase',phase,'--work-dir',str(work)]
        for name in ('source_case','checkpoint_audit','math_result','metadata_template','profile','frontend','model_info','program'):
            argv+=['--'+name.replace('_','-'),str(getattr(args,name))]
        return argv
    def bounded_run(argv,**kwargs):
        kwargs.update(mem_gib=24,fsize_gib=16)
        return run_process(argv,**kwargs)
    # The original CLI can infer a report marker; it is never exposed by this
    # namespace. Every actual entry goes through this module's explicit math ABI.
    values.pop('main',None)
    values.update(source_gate=compile_source,phase_command=phase_command,run_process=bounded_run)
    return values


def _case_size(args):return artifact_bytes(args.out)+artifact_bytes(args.compile_out)


def _budget(args):
    if _case_size(args)+RESERVE>CASE_BYTES or shutil.disk_usage(args.out).free<10*GIB:
        raise RuntimeError('Compile-first shared case or disk reserve exhausted')


def terminal_policy(args,result,deadline,failures):
    reasons=[]
    if failures:reasons.append(dict(cause='guard_failure',details=copy.deepcopy(failures)))
    if time.monotonic()>=deadline:reasons.append(dict(cause='total_wall_allowance'))
    try:_budget(args)
    except (OSError,RuntimeError) as exc:reasons.append(dict(cause='terminal_budget',error=str(exc)))
    observed=original.physical_memory_observation(None,original._available_ram_bytes(),
        rss_gib=14,minimum_available_gib=1.25)
    if observed['stop']:reasons.append(dict(cause='terminal_memory_guard',observation=observed))
    result['terminal_rejections']=reasons
    if reasons:
        result.update(status='error',error='Terminal compile-first policy rejected: '+json.dumps(reasons))
        if result.get('termination') is None:result['termination']=dict(cause='terminal_policy',reasons=reasons)
    result['compile_accepted']=(result['status']=='ok' and result.get('math_passed')is True
        and result.get('compile_completed')is True and result.get('completed')==dict(math=1,compile=1))


def _argv(args,phase,math_result,remaining):
    argv=[sys.executable,'-B','-m',MODULE]
    if phase=='math':
        argv=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(MATH_SECONDS+60),
            '--python',str(args.intake_python),'--','-B','-m',MODULE,'--worker-phase','math','--work-dir',str(args.out/'math')]
    else:argv+=['--delegate-compile','--out',str(args.compile_out),'--math-result',str(math_result),
               '--remaining-seconds',str(max(1,int(remaining)))]
    for name in ('source_case','checkpoint_audit','intake_python','metadata_template','profile','frontend'):
        argv+=['--'+name.replace('_','-'),str(getattr(args,name))]
    return argv


def execute(args):
    if (args.out.exists() or args.compile_out.exists() or args.out.parent!=paths.RESULTS_DIR.resolve()
            or args.compile_out.parent!=paths.RESULTS_DIR.resolve() or args.out==args.compile_out
            or any(args.source_case==out or args.source_case in out.parents or out in args.source_case.parents for out in (args.out,args.compile_out))):
        raise ValueError('Use explicit fresh independent metadata and compiler cases')
    started=time.perf_counter_ns();deadline=time.monotonic()+OVERALL_SECONDS
    args.out.mkdir();recorder=StageRecorder(args.out/'events.jsonl');stop=threading.Event();failures=[];active=[None]
    result=dict(round=ROUND,model=MODEL,mode=MODE,status='running',error=None,termination=None,
        attempted=dict(math=0,compile=0),completed=dict(math=0,compile=0),processes=[],
        math_passed=False,compile_completed=False,compile_accepted=False,active_phase='admission',report_deferred=True,report_task_acceptance=None,
        report_validation_metrics=None,accuracy_acceptance=None,security_acceptance=None,CKKS_latency=None,
        compiled_bootstraps=None,predicted_latency_us=None,**dict.fromkeys(ZERO,0))
    handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def interrupt(sig,frame):
        if not failures:result['termination']=dict(cause='incoming_signal',signal=sig)
        raise InterruptedError('Compile-first interrupted '+str(sig))
    for sig in handlers:signal.signal(sig,interrupt)
    def monitor():
        while not stop.wait(.2):
            try:
                _budget(args)
                if time.monotonic()>=deadline:raise TimeoutError('Compile-first total12000seconds exhausted')
                if active[0] is not None:
                    observed=original.physical_memory_observation(original._latest_rss_sample(active[0]),original._available_ram_bytes(),
                        rss_gib=14,minimum_available_gib=1.25)
                    if observed['stop']:result['memory_guard_observation']=observed;raise RuntimeError('Compile-first physical memory guard')
            except BaseException as exc:
                failure=dict(cause='resource_guard',error=type(exc).__name__+': '+str(exc));failures.append(failure);result['termination']=failure
                try:save(args.out/'guard-stop.json',failure)
                finally:os.kill(os.getpid(),signal.SIGTERM)
                return
    watcher=threading.Thread(target=monitor,daemon=True)
    try:
        with serial.serial_lease(paths.RESULTS_DIR/'bert-serial-continuation.lock'),recorder.span('compile_first_complete'):
            result['source']=metadata_source(args.source_case,args.checkpoint_audit)
            compiler_metadata(args)
            for path,name in ((Path(__file__),'driver-source.py'),(Path(original.__file__),'original-compiler-source.py'),
                (Path(serial.__file__),'original-serial-source.py'),
                (Path(serial.__file__).with_name('bert_sst2_worker.py'),'original-math-phase-source.py'),
                (args.metadata_template,'recorded-compile-template.json')):
                shutil.copyfile(path,args.out/name)
            watcher.start()
            proof=args.out/'math/math-result.json'
            for phase in ('math','compile'):
                result['active_phase']=phase
                foreign=serial.foreign_heavy_processes()
                if foreign:result['blocked_processes']=foreign;raise RuntimeError('Another heavy process is active')
                _budget(args);remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('No compile-first wall allowance remains')
                if phase=='compile':compile_source(args.source_case,args.checkpoint_audit,proof)
                argv=_argv(args,phase,proof,remaining);row=dict(phase=phase,argv=argv);result['processes'].append(row)
                result['attempted'][phase]=1;save(args.out/'result.json',result)
                with recorder.span(phase):
                    active[0]=args.out/(phase+'-resources.jsonl')
                    row['observation']=run_process(argv,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=args.out/(phase+'.log'),
                        resources=active[0],timeout=min(MATH_SECONDS+120 if phase=='math' else original.OVERALL_SECONDS+60,remaining),
                        mem_gib=24,fsize_gib=16)
                    active[0]=None
                    observation=row['observation']
                    if observation.get('signal') and not failures:result['termination']=dict(cause='incoming_signal',signal=observation['signal'])
                    if (failures or observation.get('exit_code')!=0 or observation.get('timed_out')
                            or observation.get('signal') or observation.get('remaining_pids')):
                        raise RuntimeError('Compile-first '+phase+' did not finish cleanly')
                    accepted=compile_source(args.source_case,args.checkpoint_audit,proof)
                    if phase=='math':result['math_passed']=True;result['math_result']=str(proof)
                    else:
                        compiled=read(args.compile_out/'result.json');completion=read(args.compile_out/'completion.json')
                        if (compiled.get('status')!='ok' or compiled.get('round')!=ROUND or compiled.get('mode')!=MODE
                                or compiled.get('model')!=MODEL or compiled.get('source')!=accepted
                                or compiled.get('completed')!=dict(emit=1,trace=1,compile=1)
                                or completion.get('status')!='ok' or completion.get('registry_persisted') is not True
                                or completion.get('required_artifacts_persisted') is not True):
                            raise ValueError('Actual same-source complete Emit/trace/native compilation required')
                        result.update(compile_completed=True,compile_case=str(args.compile_out),
                            compiled_bootstraps=compiled.get('compiled_bootstraps'),runtime_plan=compiled.get('runtime_plan'))
                    result['completed'][phase]=1;save(args.out/'result.json',result)
            result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc),error_stage=result['active_phase'])
    finally:
        stop.set()
        if watcher.is_alive():watcher.join(timeout=3)
        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        try:
            result['timing']=summarize_events(read_events(recorder.path))
            terminal_policy(args,result,deadline,failures)
            result['metrics']={key:result.get(key) for key in ('status','attempted','completed','math_passed','compile_completed','compile_accepted',
                'compiled_bootstraps','report_deferred','report_task_acceptance','accuracy_acceptance','security_acceptance','CKKS_latency','termination',*ZERO)}
            for _ in range(20):
                size=artifact_bytes(args.out);result['artifact_bytes_before_completion']=size
                result['shared_case_bytes_before_completion']=_case_size(args)
                if result['shared_case_bytes_before_completion']+RESERVE>CASE_BYTES:
                    result.update(status='error',error='Terminal shared case allowance exhausted',compile_accepted=False)
                    result['metrics'].update(status='error',compile_accepted=False)
                save(args.out/'result.json',result)
                if artifact_bytes(args.out)==size:break
            else:raise RuntimeError('Compile-first terminal metadata inventory did not converge')
            from ..registry import git_rev
            row=dict(run_id=args.out.name,round=ROUND,model=MODEL,mode=MODE,git_rev=git_rev(),metrics=result['metrics'],
                artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat())
            with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            persist_case_completion(args.out,round_label=ROUND,started_ns=started,status=result['status'],timing=result['timing'],
                                    required_artifact_bytes=artifact_bytes(args.out))
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
    return result


def parser():
    cli=argparse.ArgumentParser(description=__doc__)
    group=cli.add_mutually_exclusive_group()
    group.add_argument('--worker-phase',choices=('math','emit','trace','compile'));group.add_argument('--delegate-compile',action='store_true')
    for name in ('source-case','checkpoint-audit'):cli.add_argument('--'+name,type=Path,required=True)
    for name in ('out','compile-out','work-dir','math-result','model-info','program'):cli.add_argument('--'+name,type=Path)
    cli.add_argument('--intake-python',type=Path,default=paths.INTAKE_PYTHON)
    cli.add_argument('--metadata-template',type=Path,required=True)
    cli.add_argument('--profile',type=Path,default=paths.PROFILES_DIR/'qwen48/compiler-config.json')
    cli.add_argument('--frontend',type=Path,default=paths.DACAPO_ROOT/'python/hecate/hecate/expr.py')
    cli.add_argument('--remaining-seconds',type=int,default=original.OVERALL_SECONDS)
    return cli


def main(argv=None):
    cli=parser();args=cli.parse_args(argv)
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().absolute() if key=='intake_python' else value.expanduser().resolve())
    if not 1<=args.remaining_seconds<=OVERALL_SECONDS:cli.error('Use a finite remaining allowance1..12000seconds')
    if args.worker_phase:
        if args.work_dir is None:cli.error('Worker requires explicit work directory')
        if args.worker_phase=='math':result=math_worker(args)
        else:
            if any(getattr(args,key) is None for key in ('math_result','model_info','program')):cli.error('Compiler worker requires explicit math/source paths')
            if read(args.math_result).get('compiler_inputs')!=compiler_metadata(args):raise ValueError('Original compiler parameters/metadata differ from accepted mathematics')
            result=compile_namespace(args.remaining_seconds)['worker'](args)
    elif args.delegate_compile:
        if args.out is None or args.math_result is None:cli.error('Compile delegate requires fresh output and actual math result')
        if read(args.math_result).get('compiler_inputs')!=compiler_metadata(args):raise ValueError('Original compiler parameters/metadata differ from accepted mathematics')
        result=compile_namespace(args.remaining_seconds)['execute'](args)
    else:
        if args.out is None or args.compile_out is None:cli.error('Outer requires explicit fresh outer and compiler cases')
        result=execute(args)
    print(json.dumps({key:result.get(key) for key in ('status','error','math_passed','compile_completed','compiled_bootstraps')}),flush=True)
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
