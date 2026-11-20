"""Seven single-semantic layer0 Qwen Count/placement observations, no Agent.

Uses the original32-token source and original mapped weights. Existing complete
component planner/verify/Count is reused once per variant, with32 boundary boots
recorded separately. The source is parsed only; no model forward, dataset,
tokenizer, Provider, native compiler, C2 or CKKS entry runs.
"""
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

from .. import paths
from .r10_paper_metrics import StageRecorder,read_events,run_process,summarize_events,write_json
from .completion_metrics import artifact_bytes,persist_case_completion
from .r27_orbit_lazy_backend import _available_ram_bytes,_latest_rss_sample
from .r31_lattigo_runtime_plan import physical_memory_observation

MODULE='pt_agent.experiments.a7_qwen_layer0_sweep'
ROUND='A7-QWEN-LAYER0-SEMANTIC-SWEEP'
MODE='original-layer0-32token-retained13-seven-Count-placement-variants'
GIB=1024**3
VARIANTS=('baseline','softmax-exp7','softmax-exp15','rms-remez15','rms-remez31','silu15','silu31')
ZERO=('api_calls','llm_calls','model_forwards','raw_sample_reads','tokenizer_calls','heldout_samples',
      'confirm_samples','report_samples','compiler_runs','keygen_runs','ckks_executions','extra_integrity_checks')


def read(path):return json.loads(Path(path).read_text())


def source_gate(args):
    info=read(args.model_info/'model_info.json');config=info['source'].get('config') or {}
    if (config.get('model_type')!='qwen2' or config.get('num_hidden_layers')!=24
            or config.get('hidden_size')!=896 or config.get('vocab_size')!=151936
            or [(r['name'],r['shape']) for r in info['inputs']]!=
               [('inputs_embeds',[1,32,896]),('attention_mask',[1,32]),('sel',[1,32])]
            or args.profile!=(paths.PROFILES_DIR/'qwen48/compiler-config.json').resolve()
            or args.model_info!=(paths.RESULTS_DIR/'qwen-normalization-rebased-20261115/work/import/model_info').resolve()
            or args.source_program!=(paths.RESULTS_DIR/'qwen-guarded-reply13-rss16-recovery-20261116/session/sources/iteration-13.ptdsl').resolve()
            or not args.source_program.is_file()):
        raise ValueError('Preserve original trained Qwen2-0.5B,32-token signature and qwen48')
    bound=read(args.baseline_binding)
    if bound.get('format')!='qwen-layer-source-binding-v1' or bound.get('role')!='layer' or bound.get('index')!=0:
        raise ValueError('Require original saved typed layer0 source metadata')
    return dict(model_info=str(args.model_info),source_program=str(args.source_program),
        model_source=info['source'],inputs=info['inputs'],profile=str(args.profile),
        source_candidate=13,source_use='parsed unfrozen semantic baseline only',
        frozen=False,C2_inherited=False,quality_inherited=False,layer=0,prefill_tokens=32,
        baseline_binding=str(args.baseline_binding),
        omitted_variants='steps1..4 require rederived exp/inverse domains; not changed in this batch')


def variant_program(program,partition,name):
    """Change only the selected layer0 semantic fields; retain every domain."""
    if name not in VARIANTS:raise ValueError('Unknown finite A7 variant')
    ids=set(partition.layers[0].original_op_ids[3:]);changed=[];ops=[]
    for op in program.ops:
        attrs=dict(op.attrs);before=dict(attrs)
        if op.id in ids:
            if name.startswith('softmax-exp') and op.kind=='softmax':
                if attrs.get('exp_method')!='cheb':raise ValueError('Original retained13 softmax uses Chebyshev exp')
                attrs['exp_deg']=int(name.removeprefix('softmax-exp'))
            elif name.startswith('rms-remez') and op.kind=='rmsnorm':
                if attrs.get('rsqrt_method')!='newton' or attrs.get('rsqrt_deg0')!=63 or attrs.get('rsqrt_iters')!=4:
                    raise ValueError('Original retained13 RMSNorm must be Newton63 with4 iterations')
                attrs.update(rsqrt_method='remez',rsqrt_deg=int(name.removeprefix('rms-remez')),
                             rsqrt_deg0=None,rsqrt_iters=None)
            elif name.startswith('silu') and op.kind=='act' and attrs.get('fn')=='silu':
                if attrs.get('method')!='cheb' or attrs.get('deg')!=63:raise ValueError('Original SiLU Chebyshev63 required')
                attrs['deg']=int(name.removeprefix('silu'))
        if attrs!=before:
            changes={key:dict(before=before.get(key),after=attrs.get(key)) for key in set(before)|set(attrs)
                     if before.get(key)!=attrs.get(key)}
            changed.append(dict(original_op_id=op.id,kind=op.kind,changes=changes))
        ops.append(replace(op,attrs=attrs))
    expected=0 if name=='baseline' else 2 if name.startswith('rms-remez') else 1
    if len(changed)!=expected:raise ValueError('Preserve exact original layer0 nonlinear point inventory')
    return replace(program,ops=tuple(ops)),changed


def scalar_result(name,measurement,changes,elapsed):
    core=measurement['internal'];raw=measurement['complete_count'];boundary=measurement['boundary_boots']
    if (measurement.get('role')!='layer' or measurement.get('index')!=0 or boundary!=32
            or measurement.get('output_shape')!=[1,32,896] or measurement.get('output_levels')!=[5]*32
            or raw['counts']['boot']!=core['boots_pred']+32):
        raise ValueError('Preserve actual layer0 layout and32 real Count boundary bootstraps')
    groups=dict(act=0,softmax=0,norm=0,linear=0,other=0,boundary=32)
    kinds={row['id']:row['kind'] for row in measurement['source']['original_ops']}
    for row in measurement['statements']:
        kind=kinds[row['id']];group=('act' if kind=='act' else 'softmax' if kind=='softmax' else
            'norm' if kind in ('rmsnorm','layernorm') else 'linear' if kind in ('linear','matmul') else 'other')
        groups[group]+=row['boots']
    if sum(groups.values())!=raw['counts']['boot']:raise ValueError('Statement and boundary boot attribution differs')
    return dict(model='qwen2_0_5b_block',variant=name,changed_field='baseline' if not changes else
        'softmax.exp_deg' if name.startswith('softmax') else 'rmsnorm.rsqrt_method_and_degree' if name.startswith('rms') else 'silu.deg',
        value=None if not changes else int(''.join(c for c in name if c.isdigit())),
        depth_by_component=dict.fromkeys(['act','softmax','norm','linear','other']),
        depth_by_component_scope='Per-component internal depth not exposed by unchanged estimator; left unmeasured',
        max_component_live_depth=core['max_depth'],boots_pred_by_component=groups,
        boots_pred=raw['counts']['boot'],internal_boots_pred=core['boots_pred'],boundary_boots=32,
        latency_pred_s=core['latency_pred_us']/1e6+measurement['boundary_boots']*measurement['boot_us']/1e6,
        latency_prediction_scope='Original qwen48 component-plan table estimate plus32 Count boundary boots; local CKKS unmeasured',
        quality=dict(metric=None,split=None,value=None,original=None),max_u=None,feasible=None,
        placement_feasible=True,quality_evaluated=False,candidate_frozen=False,C2_inherited=False,
        changes=changes,complete_count=raw,plan_artifact=measurement['plan_artifact'],
        elapsed_seconds=elapsed,source_binding=measurement['source'],distinct_consts=len(measurement['constants']))


def run_variants(args,program,weights,*,api=None):
    from . import qwen_layered_structure as layered
    from . import qwen_component_cost as component
    from .qwen_agent_cost_lifetime import Diagnostics
    source=source_gate(args);shapes={k:v.shape for k,v in weights.items()}
    base=layered.split_program(program,shapes);rows=[]
    if layered.source_binding(base.layers[0])!=read(args.baseline_binding):
        raise ValueError('Fresh parsed layer0 source differs from original saved semantic source binding')
    for name in VARIANTS:
        write_json(args.work_dir/'progress.json',dict(status='running',active_variant=name,completed_variants=[r['variant'] for r in rows]))
        began=time.perf_counter_ns();variant,changes=variant_program(program,base,name)
        partition=layered.split_program(variant,shapes)
        callback=layered.callbacks(partition,weights)[0][0]
        directory=args.work_dir/name;directory.mkdir(exist_ok=False)
        measure=component.OriginalComponents('qwen48',directory,Diagnostics(directory/'events.jsonl')) if api is None else api(name,directory)
        measured=measure.measure(callback);measured['boot_us']=measure.boot_us
        row=scalar_result(name,measured,changes,(time.perf_counter_ns()-began)/1e9)
        row.update(source=source,status='ok',**dict.fromkeys(ZERO,0))
        write_json(directory/'result.json',row);rows.append(row)
        write_json(args.work_dir/'progress.json',dict(status='running',completed_variants=[r['variant'] for r in rows]))
        del row,measured,measure,callback,partition,variant;gc.collect()
    return rows


def worker(args):
    result=dict(round=ROUND,mode=MODE,status='running',error=None,variants=[],**dict.fromkeys(ZERO,0))
    recorder=StageRecorder(args.work_dir/'worker-events.jsonl')
    try:
        result['source']=source_gate(args)
        from .r29_local_compile_matrix import native_metadata_commands
        with native_metadata_commands(read(args.metadata_template)):
            from .bert_sst2_worker import initialize_numeric
            initialize_numeric()
            from ..models.mapped_info import load_mapped_info
            from ..dsl.parser import parse
            from ..dsl.interface import require_interface
            from ..dsl.typecheck import check
            with recorder.span('map_original_weights_and_parse_source'):
                info=load_mapped_info(args.model_info);program=parse(args.source_program.read_text())
                require_interface(program,info)
                checked=check(program,{k:v.shape for k,v in info.graph.weights.items()},graph=info.graph,prompt_version='r8-v11')
                if not checked.ok or len(info.graph.weights)!=559:raise ValueError('Original full source/types/559 read-only weights required')
                result['weight_mapping_scope']=dict(members=559,direct_readonly_mapping=True,full_tensor_copies=0)
            with recorder.span('seven_fresh_layer0_semantic_measurements'):
                result['variants']=run_variants(args,program,info.graph.weights)
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    result['completed_variants']=[name for name in VARIANTS if (args.work_dir/name/'result.json').is_file()]
    result['timing']=summarize_events(read_events(recorder.path));write_json(args.work_dir/'worker-result.json',result)
    return 0 if result['status']=='ok' else 1


def append_sweep(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        for line in stream:
            old=json.loads(line)
            if old.get('run_id')==row['run_id'] and old.get('variant')==row['variant']:
                if old!=row:raise ValueError('Preserve an existing A7 variant publication')
                return False
        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    return True


def execute(args):
    if args.out.parent!=paths.RESULTS_DIR.resolve() or args.out.exists():raise ValueError('Use a fresh direct RESULTS case')
    args.out.mkdir();args.work_dir=args.out/'work';args.work_dir.mkdir()
    started=time.perf_counter_ns();deadline=time.monotonic()+args.seconds;stop=threading.Event();failures=[]
    resource_path=args.out/'worker-resources.jsonl';recorder=StageRecorder(args.out/'events.jsonl')
    result=dict(round=ROUND,mode=MODE,model='qwen2_0_5b_block',status='running',error=None,
        attempted=0,completed=0,variants=[],**dict.fromkeys(ZERO,0))
    old={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def interrupt(sig,frame):raise InterruptedError('A7 interrupted '+str(sig))
    def budget():
        if artifact_bytes(args.out)+1024**2>4*GIB or __import__('shutil').disk_usage(args.out).free<10*GIB:
            raise RuntimeError('A7 case4GiB or free10GiB reserve exhausted')
    def monitor():
        while not stop.wait(.2):
            try:
                budget()
                if time.monotonic()>=deadline:raise TimeoutError('A7 one-hour allowance exhausted')
                obs=physical_memory_observation(_latest_rss_sample(resource_path),_available_ram_bytes(),rss_gib=14,minimum_available_gib=1.25)
                if obs['stop']:result['guard_observation']=obs;raise RuntimeError('A7 physical RSS/reserve limit')
            except BaseException as exc:
                failures.append(type(exc).__name__+': '+str(exc));os.kill(os.getpid(),signal.SIGTERM);return
    watcher=threading.Thread(target=monitor,daemon=True)
    try:
        for sig in old:signal.signal(sig,interrupt)
        result['source']=source_gate(args);budget();watcher.start()
        command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(args.seconds-15),
            '--python',str(args.intake_python),'--','-B','-m',MODULE,'--worker','--out',str(args.out),
            '--work-dir',str(args.work_dir),'--model-info',str(args.model_info),'--source-program',str(args.source_program),
            '--baseline-binding',str(args.baseline_binding),
            '--metadata-template',str(args.metadata_template),'--profile',str(args.profile),'--sweep-output',str(args.sweep_output)]
        result['command']=command;result['attempted']=1
        with recorder.span('finite_original_layer0_seven_variant_sweep'):
            result['process']=run_process(command,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=args.out/'worker.log',
                resources=resource_path,timeout=max(.01,deadline-time.monotonic()-10),mem_gib=24,fsize_gib=1)
        target=args.work_dir/'worker-result.json'
        if target.is_file():result['worker']=read(target)
        o=result['process'];w=result.get('worker') or {}
        if (failures or o.get('exit_code')!=0 or o.get('timed_out') or o.get('signal')
                or o.get('remaining_pids') or o.get('signals_sent') or w.get('status')!='ok'):
            raise RuntimeError('A7 worker did not complete all seven variants; retain partial observations')
        result.update(status='ok',completed=1)
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        stop.set()
        if watcher.is_alive():watcher.join(timeout=3)
        result['monitor_stopped']=not watcher.is_alive()
        if failures or not result['monitor_stopped'] or time.monotonic()>deadline:result.update(status='error',error=result['error'] or 'A7 terminal resource/monitor failure')
        try:budget()
        except Exception as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
        for name in VARIANTS:
            path=args.work_dir/name/'result.json'
            if not path.is_file():continue
            row=read(path)
            if row.get('status')=='ok' and row.get('source')==result.get('source'):
                publication=dict(row,run_id=args.out.name,source_result=str(path));append_sweep(args.sweep_output,publication)
                result['variants'].append(dict(variant=name,path=str(path),boots_pred=row['boots_pred']))
        result['timing']=summarize_events(read_events(recorder.path));result['metrics']={k:result[k] for k in ['status','attempted','completed','variants']+list(ZERO)}
        write_json(args.out/'result.json',result)
        from ..registry import git_rev
        record=dict(run_id=args.out.name,round=ROUND,model='qwen2_0_5b_block',mode=MODE,git_rev=git_rev(),metrics=result['metrics'],artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat())
        with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        persist_case_completion(args.out,round_label=ROUND,started_ns=started,status=result['status'],timing=result['timing'],required_artifact_bytes=artifact_bytes(args.out))
        from .a_group_runs import empty_record,append_record
        process=result.get('process') or {};status=('ok' if result['status']=='ok' else 'timeout'
            if process.get('timed_out') or any(e.startswith('TimeoutError:') for e in failures)
            else 'resource_limit' if failures else 'driver_error')
        run=empty_record(run_id=args.out.name,task='A7',model='qwen2_0_5b_block',system='arranger',param_group='ours-N16-boot5',
            status=status,status_detail=result['error'] or 'Seven fresh layer0 Count/placement variants; quality and native compilation unmeasured',
            budget_s=args.seconds,api_calls=0,source=str(args.out/'result.json'),finished_utc=record['ts'])
        run.update(variant_results=result['variants'],scope='Count and source-bound original placement/verify only;32 boundary boots per variant',quality_evaluated=False)
        append_record(args.sweep_output.parent/'runs.jsonl',run)
        if result['monitor_stopped']:
            for sig,handler in old.items():signal.signal(sig,handler)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sweep-output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--work-dir',type=Path)
    p.add_argument('--model-info',type=Path,default=paths.RESULTS_DIR/'qwen-normalization-rebased-20261115/work/import/model_info')
    p.add_argument('--source-program',type=Path,default=paths.RESULTS_DIR/'qwen-guarded-reply13-rss16-recovery-20261116/session/sources/iteration-13.ptdsl')
    p.add_argument('--baseline-binding',type=Path,default=paths.RESULTS_DIR/'qwen-retained13-component-expanded-wall-recovery-20261117/session/component-cost-1/layer0/source-binding.json')
    p.add_argument('--metadata-template',type=Path,default=paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json')
    p.add_argument('--profile',type=Path,default=paths.PROFILES_DIR/'qwen48/compiler-config.json')
    p.add_argument('--intake-python',type=Path,default=paths.INTAKE_PYTHON)
    p.add_argument('--seconds',type=int,default=3600)
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().absolute() if key=='intake_python' else value.expanduser().resolve())
    if not 60<=args.seconds<=3600:p.error('Use total60..3600seconds')
    if args.worker:
        if args.work_dir!=args.out/'work':p.error('Explicit owned inner work directory required')
        return worker(args)
    result=execute(args);print(json.dumps({k:result[k] for k in ['status','error','variants']}),flush=True)
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
