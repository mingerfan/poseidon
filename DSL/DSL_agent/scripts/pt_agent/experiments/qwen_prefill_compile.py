"""Compile one complete original Qwen prefill after independent C2 freezing.

This entry reuses a saved Agent session. It never invokes the Agent, reads its
report partition, selects another candidate, or composes the two-token selector.
Compact projections wrap placement, Count replay and trusted Emit identically.
"""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import textwrap
import time

from .. import paths
from . import qwen_bounded_feedback as bounded
from . import qwen_bridge_agent as bridge
from . import qwen_guarded_replay as guarded
from . import qwen_agent_cost_lifetime as lifetime
from . import qwen_agent_compact_record as compact_record
from . import qwen_agent_grouped_record as grouped_record
from . import qwen_retained_component_replay as retained_component
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

ROUND='QWEN-FROZEN-PREFILL-COMPILE'
MODEL='Qwen2-0.5B'
MODE='full24-frozen-single-prefill-compact-qwen48'
MODULE='pt_agent.experiments.qwen_prefill_compile'
GIB=1024**3
PHASE_SECONDS=dict(placement=7200,emit=7200,trace=1800,compile=7200)
ZERO=('api_calls','agent_runs','new_calibration_forwards','model_forwards','sample_reads',
    'report_samples','confirm_samples','keygen_runs','real_ckks_attempts','real_ckks_completed',
    'intermediate_decryptions','downloads','installations','extra_integrity_checks')
FORMAL=('security_acceptance','accuracy_acceptance','standard_lm_quality','CKKS_latency','predicted_latency')
TERMINAL_RESERVE=1024**2


def read(path):return json.loads(Path(path).read_text())


def json_value(value):
    """Normalize JSON object keys/tuples exactly as the retained result writer."""
    return json.loads(json.dumps(value,allow_nan=False))


def source_arguments(case):
    """Parse one saved explicit worker argv; parsing never calls that entry."""
    case=Path(case).resolve();command=read(case/'command-1.json')
    if not isinstance(command,list)or any(not isinstance(item,str)for item in command):
        raise ValueError('Require the saved original explicit worker argv list')
    allowed={bounded.MODULE:bounded,bridge.MODULE:bridge,guarded.MODULE:guarded,
             lifetime.MODULE:lifetime, compact_record.MODULE:compact_record, grouped_record.MODULE:grouped_record,
             retained_component.MODULE:retained_component}
    matches=[(i,item)for i,item in enumerate(command)if item in allowed and i>0 and command[i-1]=='-m']
    if len(matches)!=1:raise ValueError('One trusted Qwen Agent worker module is required')
    index,name=matches[0];entry=allowed[name]
    parser=bridge.original.parser()if entry is bridge else entry.parser()
    args=parser.parse_args(command[index+1:]);args.out=None
    if args.worker_phase not in ('calibration','prompt')or Path(args.session).resolve()!=case/'session':
        raise ValueError('Bind the saved worker argv to this original session')
    return entry,args


def frozen_source(case):
    case=Path(case).resolve();result=read(case/'result.json');complete=read(case/'completion.json')
    entry,args=source_arguments(case);state=read(case/'session/SESSION.json')
    outcome=entry.session_outcome(case/'session')
    if (result.get('round')!=entry.ROUND or result.get('mode')!=entry.MODE or result.get('model')!=MODEL
            or result.get('status')!='ok'or result.get('candidate_frozen')is not True
            or complete.get('status')!='ok'or complete.get('registry_persisted')is not True
            or complete.get('required_artifacts_persisted')is not True or state.get('failed')is not None
            or outcome.get('done')is not True or outcome.get('frozen')is not True or outcome.get('status')!='ok'):
        raise ValueError('Require the closed independently C2-confirmed original full Qwen session')
    args.session=case/'session';args.calibration=case/'calibration.json'
    if not args.calibration.is_file()or not (case/'unit-width-bridge.json').is_file():
        raise ValueError('Retain the measured23-prefix calibration and exact bridge artifact')
    source=entry.source_gate(args)
    if (source.get('original_layers')!=24 or source.get('graph_operation_counts',{}).get('softmax')!=24
            or source.get('graph_operation_counts',{}).get('shuffle')!=48):
        raise ValueError('All original24 decoder blocks must remain')
    if result.get('source')!=json_value(source):
        # Display-only source snapshots carry identical original source records.
        raise ValueError('The current original model/input metadata differs from the accepted Agent source')
    if Path(args.profile).resolve()!=(paths.PROFILES_DIR/'qwen48/compiler-config.json').resolve():
        raise ValueError('Retain the original qwen48 compiler profile path')
    return dict(source_case=str(case),entry_module=entry.MODULE,original_source=json_value(source),
        model_info=str(args.model_info),original_model=str(args.model),program=str(case/'session/program.ptdsl'),
        session=str(case/'session'),calibration=str(args.calibration),bridge=str(case/'unit-width-bridge.json'),
        metadata_template=str(args.metadata_template),frontend=str(args.frontend),profile=str(args.profile),
        inherited_calls=state['calls'],selected_iteration=outcome['selected']['iteration'],
        selected_origin=outcome['selected']['origin'],candidate_frozen=True,source_program_ast_verified=True,
        full_model_layers=24,forward_calls=1,sequence_capacity=32,encrypted_selector_included=False,
        token_embedding_feedback_included=False,report_outcome_used_for_selection=False,
        scope='One complete original24-layer prefill plus full vocabulary head; saved confirmed candidate only')


def _load_program(args,result):
    entry,original=source_arguments(args.source_case)
    original.session=args.source_case/'session';original.calibration=args.source_case/'calibration.json'
    # The original private hook applies interface/typecheck, same-source C2 and
    # exact stable-bridge attribute gates before any arithmetic lowering.
    info,program=entry._namespace()['_program'](original,result)
    if (program.header.slots!=16384 or len(info.graph.weights)!=559
            or [p.name for p in program.params]!=['inputs_embeds','attention_mask','sel']
            or tuple(program.params[0].shape)!=(1,32,896)):
        raise ValueError('Retain all559 original arrays and the complete16384-slot32x896 prefill signature')
    return info,program


def _same_lowering(program,weights,backend):
    from ..kernels.compact_linear import compact_transformer_projections
    from ..lower import lower
    with _projection_backend_scope(),compact_transformer_projections():
        return lower(program,weights,backend,act_mode='program')


def _projection_backend_allowed(backend):
    """Preserve the arithmetic wrapper; inspect its original inner type only."""
    from ..backends import CountBackend,RefreshBackend
    from ..backends.cost_dag import CostDag
    from ..backends.streaming_mock import StreamingMockBackend
    from ..backends.streaming_emit import StreamingEmitBackend
    inner=backend.inner if isinstance(backend,RefreshBackend)else backend
    return isinstance(inner,(CostDag,CountBackend,StreamingMockBackend,StreamingEmitBackend))


def _projection_delegate(original,predicate):
    """Clone the original projection, changing its single type guard only."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    function=tree.body[0]
    expected=ast.parse('not isinstance(backend,(CountBackend,StreamingMockBackend,StreamingEmitBackend))',mode='eval').body
    guards=[node for node in ast.walk(function)if isinstance(node,ast.If)
        and ast.dump(node.test,include_attributes=False)==ast.dump(expected,include_attributes=False)]
    if len(guards)!=1:raise ValueError('Original streaming projection type guard changed')
    guards[0].test=ast.parse('not _prefill_projection_backend(backend)',mode='eval').body
    ast.fix_missing_locations(tree)
    namespace=dict(original.__globals__,_prefill_projection_backend=predicate)
    exec(compile(tree,inspect.getsourcefile(original)or __file__,'exec'),namespace)
    delegated=namespace[original.__name__]
    delegated.__defaults__=original.__defaults__;delegated.__kwdefaults__=original.__kwdefaults__
    return delegated


@contextmanager
def _projection_backend_scope(module=None,predicate=None):
    """Scoped interface adaptation; all operators still receive RefreshBackend."""
    if module is None:
        module=importlib.import_module('..kernels.streaming_linear',__package__)
    original=module.streaming_projection
    module.streaming_projection=_projection_delegate(original,predicate or _projection_backend_allowed)
    try:yield
    finally:module.streaming_projection=original


def _ephemeral_cost_dag(base,numeric):
    """Original CostDag arithmetic; writable temporaries bypass identity cache."""
    class EphemeralCostDag(base):
        def const(self,vector):
            if isinstance(vector,numeric.ndarray)and not vector.flags.writeable:
                vector=numeric.array(vector,copy=True)
            return self._const(vector,keep_data=True)
    return EphemeralCostDag


def _planner_namespace(plan_module=None,dag_class=None,count_class=None,numeric=None):
    if plan_module is None:
        from .. import plan as plan_module
    if dag_class is None:
        from ..backends.cost_dag import CostDag as dag_class
    if count_class is None:
        from ..backends.streaming_count import StreamingCountBackend as count_class
    if numeric is None:
        import numpy as numeric
    return bridge._clone_namespace(plan_module,dict(CostDag=_ephemeral_cost_dag(dag_class,numeric),CountBackend=count_class))


def _placement(args,result,recorder):
    from ..kernels.compact_linear import compact_transformer_projections
    from ..backends import RefreshBackend
    from ..backends.streaming_count import StreamingCountBackend
    planner=_planner_namespace()
    info,program=_load_program(args,result)
    with recorder.span('full24_compact_program_placement'),_projection_backend_scope(),compact_transformer_projections():
        plan,dag,placed=planner['plan_program'](program,info.graph.weights,profile='qwen48',frontend_normalize=True)
        write_json(args.codegen/'refresh-plan.json',plan)
        result.update(planned_bootstraps=plan['boots'],placement_variant=plan['placement'])
        del dag,placed;gc.collect()
        result['plan_verification']=planner['verify_program'](program,info.graph.weights,plan,profile='qwen48')
    with recorder.span('independent_full24_compact_count'):
        counter=StreamingCountBackend(program.header.slots,boot_out_level=plan['levels']['boot_out_level'])
        active=RefreshBackend(counter,policy='plan',plan=plan,**plan['levels'])
        output,layouts=_same_lowering(program,info.graph.weights,active)
        result.update(count_stats=counter.stats(),refresh_stats=active.stats(),output_blocks=len(output))
        result['constant_lifetime_policy']='Private CostDag readonly-copy operands and original StreamingCountBackend; constant names retained, arrays live only with current operands'
        result['projection_dispatch_policy']='Original streaming projection body with one private type-guard adaptation for supported RefreshBackend inner; the unchanged outer wrapper performs every arithmetic operation'
        if len(output)!=19:raise ValueError('Retain all19 compact blocks of the original151936-vocabulary output')
        result['output_layouts']=[str(layout)for layout in layouts]
        if counter.stats()['counts']['boot']!=plan['boots']or active.stats()['boots']!=plan['boots']:
            raise ValueError('Original placement and compact Count replay bootstrap counts differ')
        write_json(args.codegen/'count-replay.json',dict(stats=counter.stats(),refresh=active.stats()))
    result['placement_completed']=True


def _emit(args,result,recorder):
    from ..backends import RefreshBackend
    from ..backends.streaming_emit import StreamingEmitBackend
    info,program=_load_program(args,result);plan=read(args.codegen/'refresh-plan.json')
    counted=read(args.codegen/'count-replay.json')['stats']
    backend=StreamingEmitBackend(program.header.slots,directory=args.codegen/'emit',frontend=args.frontend,
        max_constant_bytes=int(args.case_gib*GIB-GIB))
    try:
        with recorder.span('full24_compact_streaming_emit'):
            active=RefreshBackend(backend,policy='plan',plan=plan,**plan['levels'])
            output,layouts=_same_lowering(program,info.graph.weights,active)
            stats=backend.stats();result.update(emit_stats=stats,output_blocks=len(output))
            if len(output)!=19:raise ValueError('Retain the complete original151936-vocabulary output')
            if any(stats[key]!=counted[key]for key in ('counts','ops_total','distinct_consts'))or active.stats()['boots']!=plan['boots']:
                raise ValueError('The same complete compact Count/Emit schedule must match')
            result.update(trace_source=str(backend.write().resolve()),emit_completed=True)
    finally:
        if not backend.finished:backend.close()


def native_command(args,host):
    emit=args.codegen/'emit'
    if args.worker_phase=='trace':return [sys.executable,'-B',str(emit/'candidate_trace.py')]
    from .r29_local_compile_matrix import compiler_argv
    from .r30_common_model_e2e import diagnostic_compile_argv
    argv=compiler_argv(read(args.metadata_template),'pars',binary=host['hecate_opt'],
        source=emit/'candidate_trace.mlir',profile=args.profile,output=emit/'lowered.mlir')
    return diagnostic_compile_argv(argv,elide_final_ir=args.diagnostic_elide_final_ir)


def compiled_inventory(path,expected_bootstraps):
    """Read complete instruction metadata with bounded memory and no blobs."""
    from .native_component_matrix import array_items
    counts={};operations=0
    for row in array_items(path,'execution'):
        if row.get('kind')!='compute' or not isinstance(row.get('op'),str):
            raise ValueError('Require native compute instructions in the complete prefill plan')
        counts[row['op']]=counts.get(row['op'],0)+1;operations+=1
    outputs=list(array_items(path,'final_outputs'));inputs=list(array_items(path,'external_inputs'))
    if counts.get('boot',0)!=expected_bootstraps or len(outputs)!=19 or not inputs:
        raise ValueError('Compiled prefill must retain planned bootstraps and19 vocabulary blocks')
    return dict(compiled_operation_count=operations,compiled_bootstraps=counts.get('boot',0),
        compiled_op_counts=counts,compiled_inputs=len(inputs),compiled_outputs=len(outputs),
        RuntimePlan_reader='bounded64KiB JSON instruction decoder; no full plan object or plaintext blob reads')


def worker(args):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    args.work_dir.mkdir(parents=True,exist_ok=False);args.codegen.mkdir(exist_ok=True)
    recorder=StageRecorder(args.work_dir/'events.jsonl');started=time.perf_counter_ns()
    result=dict(round=ROUND,model=MODEL,mode=MODE,phase=args.worker_phase,status='running',error=None,
        **dict.fromkeys(ZERO,0),**dict.fromkeys(FORMAL))
    write_json(args.work_dir/'worker-result.json',result)
    try:
        result['source']=frozen_source(args.source_case)
        for name in ('metadata_template','frontend','profile'):setattr(args,name,Path(result['source'][name]))
        from .r29_local_compile_matrix import native_metadata_commands
        with native_metadata_commands(read(args.metadata_template)),recorder.span('frozen_full24_'+args.worker_phase):
            if args.worker_phase in ('placement','emit'):
                import torch
                torch.set_num_threads(1);torch.set_num_interop_threads(1)
                (_placement if args.worker_phase=='placement'else _emit)(args,result,recorder)
            else:
                from .. import toolchain
                host=toolchain.host_paths();env=dict(os.environ,HECATE=str(host['hecate']),HECATE_BUILD=str(host['hecate_build']),
                    LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'])
                command=native_command(args,host);write_json(args.work_dir/'native-argv.json',command)
                result['process']=run_process(command,cwd=args.codegen/'emit',log=args.work_dir/'native.log',
                    resources=args.work_dir/'native-resources.jsonl',timeout=args.phase_seconds,mem_gib=args.memory_gib,fsize_gib=64,env=env)
                process=result['process']
                if process['exit_code']!=0 or process['timed_out']or process['signal']is not None or process['remaining_pids']:
                    raise RuntimeError('Native '+args.worker_phase+' did not complete')
                if args.worker_phase=='trace':
                    target=args.codegen/'emit/candidate_trace.mlir'
                    if not target.is_file()or target.stat().st_size==0:raise ValueError('Trusted trace must save nonempty complete MLIR')
                    result.update(trace_completed=True,mlir=dict(path=str(target),bytes=target.stat().st_size))
                else:
                    plans=list((args.codegen/'emit').glob('lowered.*.runtime-plan.json'))
                    if len(plans)!=1:raise ValueError('One full prefill RuntimePlan is required')
                    observed=compiled_inventory(plans[0],read(args.codegen/'refresh-plan.json')['boots'])
                    result.update(compile_completed=True,runtime_plan=str(plans[0]),**observed,
                        diagnostic_final_ir=args.diagnostic_elide_final_ir)
            result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        result['elapsed_ns']=time.perf_counter_ns()-started;write_json(args.work_dir/'worker-result.json',result)
    return result


def resource_limits(args):
    if (type(args.memory_gib)not in(int,float)or not math.isfinite(args.memory_gib)or not 12<=args.memory_gib<=64
            or type(args.case_gib)is not int or not 32<=args.case_gib<=128):
        raise ValueError('Use finite AS12..64GiB and case32..128GiB; physical RSS remains14GiB')
    if type(args.seconds)is not int or not 1<=args.seconds<=10800:
        raise ValueError('The complete post-freeze batch has a finite allowance through10800 seconds')
    return dict(address_space_gib=args.memory_gib,subtree_rss_gib=14,minimum_available_gib=1.25,
        case_bytes=args.case_gib*GIB,minimum_free_bytes=10*GIB,file_gib=64,phases=PHASE_SECONDS.copy(),
        source_metadata_seconds=600,complete_batch_seconds=args.seconds)


def phase_command(args,phase,work,remaining=None):
    if phase not in PHASE_SECONDS:raise ValueError('Unknown frozen prefill phase')
    seconds=PHASE_SECONDS[phase]if remaining is None else min(PHASE_SECONDS[phase],max(1,int(remaining)-120))
    return [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(seconds+60),'--python',str(args.intake_python),
        '--','-B','-m',MODULE,'--worker-phase',phase,'--source-case',str(args.source_case),'--work-dir',str(work),
        '--codegen',str(args.out/'codegen'),'--memory-gib',str(args.memory_gib),'--case-gib',str(args.case_gib),
        '--phase-seconds',str(seconds)]+(
        ['--diagnostic-elide-final-ir']if args.diagnostic_elide_final_ir else [])


def budget(args,out):
    size=artifact_bytes(out);free=shutil.disk_usage(out).free
    if size+TERMINAL_RESERVE>args.case_gib*GIB or free<10*GIB:raise RuntimeError('Frozen prefill disk allowance exhausted')
    return dict(case_bytes=size,free_bytes=free)


def _terminal(args,out,result,started,revision,recorder):
    result['recorded_pre_persistence_seconds']=(time.perf_counter_ns()-started)/1e9
    result['timing']=summarize_events(read_events(out/'events.jsonl'))
    result['timing_scope']='Outer exclusive phase attribution after read-only source admission; source_metadata_seconds is separately observed. Completion includes source admission and terminal persistence.'
    result['metrics']={key:result.get(key)for key in ('status','attempted','completed','failed_phase','compiled_bootstraps','termination')}
    result['metrics'].update(dict.fromkeys(FORMAL));result['metrics'].update(dict.fromkeys(ZERO,0))
    result['artifact_bytes']=0
    for _ in range(20):
        write_json(out/'result.json',result);size=artifact_bytes(out)
        if size+TERMINAL_RESERVE>args.case_gib*GIB:
            result.update(status='error',terminal_budget_error='Completion marker reserve exhausted');result['metrics']['status']='error'
        if size==result['artifact_bytes']:break
        result['artifact_bytes']=size
    else:raise RuntimeError('Frozen prefill terminal size did not converge')
    row=dict(run_id=out.name,round=ROUND,model=MODEL,mode=MODE,git_rev=revision,metrics=result['metrics'],
        artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR/'registry.jsonl').open('a+')as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()):raise ValueError('Preserve the existing registry identity')
        stream.seek(0,os.SEEK_END);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],timing=result['timing'],required_artifact_bytes=artifact_bytes(out))


def execute(args):
    limits=resource_limits(args);started=time.perf_counter_ns();deadline=time.monotonic()+args.seconds
    previous=signal.getsignal(signal.SIGALRM)
    def expire(signum,frame):raise TimeoutError('Frozen source metadata allowance exhausted')
    signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,600)
    try:source=frozen_source(args.source_case)
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    source_seconds=(time.perf_counter_ns()-started)/1e9
    out=args.out.resolve();case=args.source_case.resolve()
    if (out.exists()or out.is_symlink()or out==paths.RESULTS_DIR.resolve()or not out.is_relative_to(paths.RESULTS_DIR.resolve())
            or out.is_relative_to(case)or case.is_relative_to(out)):
        raise ValueError('Use a fresh isolated configured results case')
    out.mkdir();recorder=StageRecorder(out/'events.jsonl')
    from ..registry import git_rev
    revision=git_rev();result=dict(round=ROUND,model=MODEL,mode=MODE,status='running',error=None,source=source,
        limits=limits,source_metadata_seconds=source_seconds,phase_results={},processes=[],attempted=dict.fromkeys(PHASE_SECONDS,0),
        completed=dict.fromkeys(PHASE_SECONDS,0),compiled_bootstraps=None,**dict.fromkeys(ZERO,0),**dict.fromkeys(FORMAL))
    write_json(out/'source-record.json',source);snapshot=out/'source';snapshot.mkdir()
    for module in (sys.modules[__name__],bounded,bridge,bridge.original):shutil.copyfile(module.__file__,snapshot/Path(module.__file__).name)
    active=[None];stop=threading.Event();failed=[];old={sig:signal.getsignal(sig)for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGUSR1)}
    def incoming(sig,frame):
        if not failed:result['termination']=dict(cause='incoming_signal',signal=sig)
        raise InterruptedError('Frozen prefill interrupted')
    for sig in old:signal.signal(sig,incoming)
    def monitor():
        from .r27_orbit_lazy_backend import _available_ram_bytes,_latest_rss_sample
        from .r31_lattigo_runtime_plan import physical_memory_observation
        while not stop.wait(.25):
            try:
                budget(args,out)
                if time.monotonic()>=deadline:raise TimeoutError('Complete post-freeze10800-second allowance exhausted')
                observation=physical_memory_observation(None if active[0]is None else _latest_rss_sample(active[0]),
                    _available_ram_bytes(),rss_gib=14,minimum_available_gib=1.25)
                if observation['stop']:raise RuntimeError(str(observation))
            except BaseException as exc:
                failed.append(str(exc));result['termination']=dict(cause='wall_timeout'if isinstance(exc,TimeoutError)else'physical_or_storage_guard',error=str(exc))
                try:write_json(out/'guard-failure.json',result['termination'])
                finally:os.kill(os.getpid(),signal.SIGUSR1)
                return
    thread=threading.Thread(target=monitor,daemon=True)
    try:
        thread.start()
        with recorder.span('frozen_full24_single_prefill_compile'):
            for phase in PHASE_SECONDS:
                remaining=deadline-time.monotonic()
                if remaining<=120:raise TimeoutError('No complete phase supervision allowance remains')
                result['failed_phase']=phase;work=out/'phases'/phase;command=phase_command(args,phase,work,remaining)
                phase_seconds=int(command[command.index('--phase-seconds')+1])
                result['attempted'][phase]+=1;write_json(out/'result.json',result)
                active[0]=out/(phase+'-resources.jsonl')
                with recorder.span('phase_'+phase):
                    observation=run_process(command,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=out/(phase+'.log'),
                        resources=active[0],timeout=min(phase_seconds+120,max(.01,deadline-time.monotonic())),mem_gib=args.memory_gib,fsize_gib=64,
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'))
                active[0]=None;result['processes'].append(dict(phase=phase,argv=command,observation=observation))
                target=work/'worker-result.json';child=read(target)if target.is_file()else None
                result['phase_results'][phase]=child;write_json(out/'result.json',result)
                if (failed or observation['exit_code']!=0 or observation['timed_out']or observation['signal']is not None
                        or observation['remaining_pids']or not child or child.get('status')!='ok'or child.get(phase+'_completed')is not True):
                    raise RuntimeError('First failed complete Qwen prefill phase: '+phase)
                result['completed'][phase]+=1
                if phase=='compile':result.update(compiled_bootstraps=child['compiled_bootstraps'],runtime_plan=child['runtime_plan'])
                if phase==args.through:break
            result.update(status='ok',failed_phase=None)
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        stop.set()
        if thread.is_alive():thread.join(timeout=2)
        for phase in PHASE_SECONDS:
            target=out/'phases'/phase/'worker-result.json'
            if target.is_file():result['phase_results'][phase]=read(target)
        _terminal(args,out,result,started,revision,recorder)
        for sig,handler in old.items():signal.signal(sig,handler)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case',type=Path,required=True);parser.add_argument('--out',type=Path)
    parser.add_argument('--through',choices=tuple(PHASE_SECONDS),default='compile')
    parser.add_argument('--worker-phase',choices=tuple(PHASE_SECONDS))
    for key in ('work-dir','codegen'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--intake-python',type=Path,default=paths.INTAKE_PYTHON)
    parser.add_argument('--memory-gib',type=float,default=32.);parser.add_argument('--case-gib',type=int,default=128)
    parser.add_argument('--seconds',type=int,default=10800);parser.add_argument('--phase-seconds',type=int)
    parser.add_argument('--diagnostic-elide-final-ir',action='store_true')
    args=parser.parse_args(argv);resource_limits(args)
    if args.worker_phase:
        if args.work_dir is None or args.codegen is None:parser.error('Every inner worker needs explicit work/codegen paths')
        if type(args.phase_seconds)is not int or not 1<=args.phase_seconds<=PHASE_SECONDS[args.worker_phase]:parser.error('Use the explicit finite original phase allowance')
        value=worker(args)
    else:
        if args.out is None:parser.error('Use a fresh explicit --out')
        value=execute(args)
    print(json.dumps(dict(status=value['status'],error=value.get('error'),compiled_bootstraps=value.get('compiled_bootstraps'))))
    return int(value['status']!='ok')


if __name__=='__main__':raise SystemExit(main())
