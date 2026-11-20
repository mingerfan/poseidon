"""A1: original deterministic BERT R0 with Chebyshev63 activations, depth-DP.

Only the trusted rule generator runs. Its generated DSL is parsed, never eval'd.
The original transformer rule ignores its caller's deg argument; this explicit
variant changes only act(...,method='cheb',deg=...) to63. Other R0 choices and
calibration domains stay original. No quality dataset or Provider is accessed.
"""
import argparse
import ast
from datetime import datetime, timezone
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
import types

from .. import paths
from . import r29_local_compile_matrix as original
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample
from .r31_lattigo_runtime_plan import physical_memory_observation

MODULE='pt_agent.experiments.a1_bert_fixed_current'
ROUND='A1-BERT-FIXED-CURRENT'
MODEL='bert-tiny-sst2'
MODE='deterministic-R0-cheb63-depth-dp-qwen48-current-build'
GIB=1024**3
PHASES=('prepare','emit','trace','compile')
ZERO=('api_calls','llm_calls','model_forwards','raw_sample_reads','heldout_samples','confirm_samples',
      'report_samples','keygen_runs','ckks_executions','extra_integrity_checks','downloads','installations')


def read(path):return json.loads(Path(path).read_text())


def source_gate(args):
    """Bind the already compiled9180 comparison's actual model/profile paths."""
    current=read(args.comparison_case/'result.json')
    completion=read(args.comparison_case/'completion.json')
    source=current.get('source') or {};native=(current.get('phases') or {}).get('compile') or {}
    model_info=Path(source.get('model_info','')).resolve()
    info=read(model_info);config=(info.get('source') or {}).get('config') or {}
    command=read(args.comparison_case/'work/compile-argv.json')
    if (current.get('status')!='ok' or completion.get('status')!='ok'
            or completion.get('required_artifacts_persisted') is not True
            or completion.get('registry_persisted') is not True
            or current.get('compiled_bootstraps')!=9180 or native.get('status')!='ok'
            or Path(native.get('runtime_plan','')).resolve()!=Path(current.get('runtime_plan','')).resolve()
            or not Path(current.get('runtime_plan','')).is_file()
            or config.get('model_type')!='bert' or config.get('num_hidden_layers')!=2
            or config.get('hidden_size')!=128 or config.get('num_labels')!=2
            or info.get('task')!='classification'):
        raise ValueError('Require the successful current9180 original BERT model and compiled comparison')
    if (args.profile!=(paths.PROFILES_DIR/'qwen48/compiler-config.json').resolve()
            or args.frontend!=(paths.DACAPO_ROOT/'python/hecate/hecate/expr.py').resolve()
            or command.count('--pars')!=1 or '--waterline=43' not in command
            or '--ckks-config='+str(args.profile) not in command
            or '--runtime-plan-context-id=qwen-bootstrap48' not in command):
        raise ValueError('Preserve current compiled comparison qwen48 and trusted frontend')
    calibration=model_info.parent/'calibration.json'
    if not calibration.is_file():raise ValueError('Original saved calibration statistics are required')
    original.compiler_argv(read(args.metadata_template),'dacapo-depth-dp',binary=command[0],
        source='/input',profile=args.profile,output='/output')
    return dict(comparison_case=str(args.comparison_case),comparison_runtime_plan=current['runtime_plan'],
        model_info=str(model_info),original_model=info['source']['local_path'],
        model_source=info['source'],inputs=info['inputs'],sequence=info['sequence'],
        calibration=str(calibration),profile=str(args.profile),frontend=str(args.frontend),
        metadata_template=str(args.metadata_template),hecate_opt=command[0],
        original_rule='planners.rule.plan(deg=63,placement=depth-dp,refresh=compiler)',
        transformer_degree_dispatch_caveat='Original transformer dispatch ignores deg; explicit act-only63 rewrite follows',
        quality_evaluated=False,accuracy=None)


def rule63_source(text):
    """Only trusted act degree keywords change; no source evaluation."""
    tree=ast.parse(text);changes=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!='act':continue
        fields={item.arg:item for item in node.keywords}
        if (set(('method','deg','fn','domain'))-set(fields)
                or ast.literal_eval(fields['method'].value)!='cheb'
                or ast.literal_eval(fields['fn'].value) not in ('gelu','tanh')):
            raise ValueError('Expected original R0 Chebyshev GELU/tanh activations')
        before=ast.literal_eval(fields['deg'].value)
        if type(before) is not int or before<1:raise ValueError('Invalid original activation degree')
        changes.append(dict(line=node.lineno,fn=ast.literal_eval(fields['fn'].value),
                            original_degree=before,degree=63,domain=ast.literal_eval(fields['domain'].value)))
        fields['deg'].value=ast.Constant(value=63)
    if not changes:raise ValueError('Require original R0 BERT activations')
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)+'\n',changes


def _inputs(args):
    from ..models.info import ModelInfo
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from ..dsl.interface import require_interface
    info=ModelInfo.load(args.model_info);program=parse(args.program.read_text())
    signature=require_interface(program,info)
    checked=check(program,{key:value.shape for key,value in info.graph.weights.items()},graph=info.graph,prompt_version='r8-v11')
    if not checked.ok or program.header.refresh!='compiler' or program.header.placement!='depth-dp':
        raise ValueError('Original rule BERT interface/type/depth header differs')
    return info,program,signature


def recipe_metadata(program):
    """Keep approximation scalars and named references; no weight payloads."""
    from ..dsl.ir import WeightRef,ValueRef
    names={'fn','method','deg','domain','axis','keep','shift','inv_scale','steps','eps'}
    prefixes=('rsqrt_','exp_','inv_','step_inv_','final_inv_')
    def scalar(value):
        if value is None or type(value) in (str,int,float,bool):return value
        if isinstance(value,(tuple,list)):return [scalar(item) for item in value]
        raise TypeError('Approximation recipe requires scalar/list metadata')
    rows=[]
    for op in program.ops:
        if op.kind not in ('act','softmax','layernorm','rmsnorm'):continue
        attrs={key:scalar(value) for key,value in op.attrs.items()
               if key in names or key.startswith(prefixes)}
        references={key:dict(kind='weight' if isinstance(value,WeightRef) else 'value',name=value.name)
                    for key,value in op.attrs.items() if isinstance(value,(WeightRef,ValueRef))}
        rows.append(dict(op_id=op.id,line=op.lineno,kind=op.kind,attrs=attrs,references=references))
    return rows


def prepare(args):
    source=source_gate(args)
    with original.native_metadata_commands(read(args.metadata_template)):
        from ..models.info import ModelInfo
        from ..planners.rule import plan
        from ..backends import CountBackend
        from ..lower import lower
        info=ModelInfo.load(source['model_info'])
        text=plan(info.graph,read(source['calibration']),deg=63,slots=16384,placement='depth-dp',refresh='compiler')
        text,changes=rule63_source(text)
        args.program.write_text(text)
        del info;gc.collect()
        info,program,signature=_inputs(args)
        counter=CountBackend(program.header.slots)
        lower(program,info.graph.weights,counter,act_mode='program')
        recipe=recipe_metadata(program)
        return dict(round=ROUND,model=MODEL,mode=MODE,phase='prepare',status='ok',error=None,
            source=source,recipe=recipe,activation_degree_changes=changes,raw_count=counter.stats(),
            program_ast=ast.dump(ast.parse(text),include_attributes=False),signature=signature,**dict.fromkeys(ZERO,0))


def worker(args):
    from .bert_sst2_worker import initialize_numeric
    initialize_numeric();args.work_dir.mkdir(parents=True,exist_ok=True)
    if args.work_dir!=args.out/'work' or args.program!=args.work_dir/'rule63.ptdsl':
        raise ValueError('Preserve explicit fresh-case work and trusted Rule63 source paths')
    target=args.work_dir/(args.worker_phase+'-result.json')
    if target.exists():raise ValueError('Preserve previous worker result')
    if args.worker_phase=='prepare':
        recorder=StageRecorder(args.work_dir/'prepare-events.jsonl')
        result=dict(round=ROUND,model=MODEL,mode=MODE,phase='prepare',status='error',error=None,**dict.fromkeys(ZERO,0))
        try:
            with recorder.span('original_R0_rule63_generation_and_raw_Count'):result=prepare(args)
        except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
        result['timing']=summarize_events(read_events(recorder.path));write_json(target,result)
        return 0 if result['status']=='ok' else 1
    source=source_gate(args)
    if Path(source['model_info'])!=args.model_info:raise ValueError('Original model path changed')
    prepared=read(args.work_dir/'prepare-result.json')
    if (prepared.get('status')!='ok' or prepared.get('source')!=source
            or prepared.get('program_ast')!=ast.dump(ast.parse(args.program.read_text()),include_attributes=False)):
        raise ValueError('Retain the exact prepared deterministic Rule63 source AST and original source')
    def finite_process(argv,**kwargs):
        kwargs['fsize_gib']=16
        return run_process(argv,**kwargs)
    def same_compiler(template,mode,**kwargs):
        if str(kwargs['binary'])!=source['hecate_opt']:
            raise ValueError('Actual compiler binary differs from current9180 comparison')
        return original.compiler_argv(template,mode,**kwargs)
    values=dict(vars(original),ROUND=ROUND,_worker_inputs=_inputs,selected_model=lambda unused:MODEL,
                run_process=finite_process,compiler_argv=same_compiler)
    delegated=types.FunctionType(original.worker.__code__,values,'worker',original.worker.__defaults__,original.worker.__closure__)
    delegated.__kwdefaults__=original.worker.__kwdefaults__
    result=delegated(args)
    row=read(target);row.update(mode=MODE,source=source,**dict.fromkeys(ZERO,0))
    if args.worker_phase=='emit' and row.get('status')=='ok':
        count=read(args.work_dir/'prepare-result.json')['raw_count']
        if any(row['emit_stats'][key]!=count[key] for key in ('counts','ops_total','distinct_consts')):
            row.update(status='error',error='Complete raw Count/Emit operation or constant inventory differs');result=1
    if args.worker_phase=='compile' and row.get('status')=='ok':
        command=read(args.work_dir/'compile-argv.json')
        if command[0]!=source['hecate_opt']:
            row.update(status='error',error='Actual compiler binary differs from current9180 comparison');result=1
    write_json(target,row)
    return result


def phase_command(args,phase):
    inner=['-B','-m',MODULE,'--worker-phase',phase,'--out',str(args.out),'--work-dir',str(args.work_dir),
        '--program',str(args.program),'--model-info',str(args.model_info),
        '--comparison-case',str(args.comparison_case),'--profile',str(args.profile),
        '--frontend',str(args.frontend),'--metadata-template',str(args.metadata_template),
        '--memory-gib',str(args.memory_gib),'--case-gib',str(args.case_gib),
        '--trace-seconds',str(args.trace_seconds),'--compile-seconds',str(args.compile_seconds)]
    seconds=args.compile_seconds if phase=='compile' else args.trace_seconds if phase=='trace' else 1800
    return [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(seconds+120),
            '--python',str(args.intake_python),'--',*inner],seconds+240


def budget(args):
    if artifact_bytes(args.out)+1024**2>args.case_gib*GIB or shutil.disk_usage(args.out).free<10*GIB:
        raise RuntimeError('A1 case byte allowance or10GiB free disk reserve exhausted')


def execute(args):
    if args.out.parent!=paths.RESULTS_DIR.resolve() or args.out.exists():raise ValueError('Use a fresh direct RESULTS case')
    args.out.mkdir();args.work_dir=args.out/'work';args.work_dir.mkdir()
    args.program=args.work_dir/'rule63.ptdsl'
    began=time.perf_counter_ns();deadline=time.monotonic()+args.seconds
    recorder=StageRecorder(args.out/'events.jsonl');stop=threading.Event();failures=[];active=[None]
    result=dict(round=ROUND,model=MODEL,mode=MODE,status='running',error=None,phases={},processes=[],
        attempted=dict.fromkeys(PHASES,0),completed=dict.fromkeys(PHASES,0),
        compiled_bootstraps=None,predicted_latency_us=None,runtime_plan=None,accuracy=None,
        limits=dict(address_space_gib=args.memory_gib,physical_subtree_rss_gib=14,minimum_available_gib=1.25,
                    file_gib=16,case_gib=args.case_gib,free_disk_gib=10,seconds=args.seconds),**dict.fromkeys(ZERO,0))
    old={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def interrupt(sig,frame):raise InterruptedError('A1 interrupted with signal '+str(sig))
    def monitor():
        while not stop.wait(.2):
            try:
                budget(args)
                if time.monotonic()>=deadline:raise TimeoutError('A1 whole-case wall allowance exhausted')
                memory=physical_memory_observation(_latest_rss_sample(active[0]) if active[0] else None,
                    _available_ram_bytes(),rss_gib=14,minimum_available_gib=1.25)
                if memory['stop']:
                    result['memory_guard_observation']=memory;raise RuntimeError('A1 physical memory reserve or RSS allowance')
            except BaseException as exc:
                failures.append(type(exc).__name__+': '+str(exc));write_json(args.out/'guard-observation.json',dict(error=failures[-1]))
                os.kill(os.getpid(),signal.SIGTERM);return
    watcher=threading.Thread(target=monitor,daemon=True)
    try:
        for sig in old:signal.signal(sig,interrupt)
        with recorder.span('current_BERT_Rule63_prepare_emit_trace_compile'):
            source=source_gate(args);result['source']=source;args.model_info=Path(source['model_info'])
            for file,name in ((Path(__file__),'driver-source.py'),(args.profile,'compiler-config.json'),
                (args.metadata_template,'recorded-compile-template.json')):shutil.copyfile(file,args.out/name)
            watcher.start()
            for phase in PHASES:
                if failures:raise RuntimeError(failures[0])
                budget(args)
                if source_gate(args)!=source:raise ValueError('Original source metadata changed')
                command,seconds=phase_command(args,phase);row=dict(phase=phase,argv=command)
                result['processes'].append(row);result['attempted'][phase]=1
                active[0]=args.out/(phase+'-resources.jsonl')
                with recorder.span(phase):row['observation']=run_process(command,cwd=paths.REPO/'DSL/DSL_agent/scripts',
                    log=args.out/(phase+'.log'),resources=active[0],timeout=min(seconds,max(.01,deadline-time.monotonic())),
                    mem_gib=args.memory_gib,fsize_gib=16)
                active[0]=None
                target=args.work_dir/(phase+'-result.json')
                if target.is_file():result['phases'][phase]=read(target)
                child=result['phases'].get(phase) or {};obs=row['observation']
                if (failures or obs.get('exit_code')!=0 or obs.get('timed_out') or obs.get('signal')
                        or obs.get('remaining_pids') or child.get('status')!='ok' or child.get('source')!=source
                        or child.get('round')!=ROUND or child.get('model')!=MODEL or child.get('mode')!=MODE):
                    raise RuntimeError('A1 '+phase+' failed; preserve this attempt and stop')
                result['completed'][phase]=1
            result.update({key:result['phases']['compile'].get(key) for key in
                ('compiled_bootstraps','predicted_latency_us','runtime_plan','prediction','execution_steps','execution_ops')})
            result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        stop.set()
        if watcher.is_alive():watcher.join(timeout=3)
        result['monitor_stopped']=not watcher.is_alive()
        if failures or not result['monitor_stopped'] or time.monotonic()>deadline:
            result.update(status='error',error=result['error'] or 'A1 final resource/wall/monitor closure failed')
        try:budget(args)
        except Exception as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
        result['compile_accepted']=result['status']=='ok' and result['completed']==dict.fromkeys(PHASES,1)
        result['timing']=summarize_events(read_events(recorder.path))
        native=(result['phases'].get('compile') or {}).get('process') or {}
        result['compile_s']=((native['ended_ns']-native['started_ns'])/1e9
            if type(native.get('started_ns')) is int and type(native.get('ended_ns')) is int else None)
        result['generation_e2e_scope']='warm original ModelInfo/calibration entry through closed required artifacts/registry; original intake excluded'
        result['metrics']={key:result.get(key) for key in ('status','attempted','completed','compile_accepted',
            'compiled_bootstraps','predicted_latency_us','compile_s','accuracy')+ZERO}
        write_json(args.out/'result.json',result)
        from ..registry import git_rev
        row=dict(run_id=args.out.name,round=ROUND,model=MODEL,mode=MODE,git_rev=git_rev(),metrics=result['metrics'],
            artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat())
        with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        persist_case_completion(args.out,round_label=ROUND,started_ns=began,status=result['status'],
            timing=result['timing'],required_artifact_bytes=artifact_bytes(args.out))
        if result['monitor_stopped']:
            for sig,handler in old.items():signal.signal(sig,handler)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--comparison-case',type=Path,default=paths.RESULTS_DIR/'bert-sst2-three-round-frozen-compile-20261117')
    p.add_argument('--metadata-template',type=Path,default=paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json')
    p.add_argument('--profile',type=Path,default=paths.PROFILES_DIR/'qwen48/compiler-config.json')
    p.add_argument('--frontend',type=Path,default=paths.DACAPO_ROOT/'python/hecate/hecate/expr.py')
    p.add_argument('--intake-python',type=Path,default=paths.INTAKE_PYTHON)
    p.add_argument('--worker-phase',choices=PHASES)
    for name in ('work-dir','program','model-info'):p.add_argument('--'+name,type=Path)
    p.add_argument('--memory-gib',type=int,choices=(24,),default=24)
    p.add_argument('--case-gib',type=int,choices=(32,),default=32)
    p.add_argument('--trace-seconds',type=int,default=1800)
    p.add_argument('--compile-seconds',type=int,default=7200)
    p.add_argument('--seconds',type=int,default=13080)
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().absolute() if key=='intake_python' else value.expanduser().resolve())
    if not 0<args.trace_seconds<=1800 or not 0<args.compile_seconds<=7200 or not 0<args.seconds<=13080:
        p.error('Use finite trace<=1800/compile<=7200/whole<=13080 seconds')
    args.system='dacapo-depth-dp';args.debug_snapshots=False
    if args.worker_phase:
        if any(getattr(args,key) is None for key in ('program','model_info','work_dir')):p.error('Explicit inner paths required')
        return worker(args)
    result=execute(args)
    print(json.dumps({key:result.get(key) for key in ('status','error','compiled_bootstraps','compile_accepted')}),flush=True)
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
