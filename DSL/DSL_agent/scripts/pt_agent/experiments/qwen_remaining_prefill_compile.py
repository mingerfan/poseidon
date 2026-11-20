"""Finite original23-layer/head continuation after a real layer0 ABI probe.

Every component uses its own original planner/verify, Count/Emit and native
processes. The master retains all25 source bindings and actual artifacts. It
never executes a model, reads an input sample or runs generated ciphertext code.
"""
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import gc
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import textwrap
import threading
import time
from types import FunctionType, SimpleNamespace

from . import qwen_prefill_compile as original
from . import qwen_layer_native_probe as probe
from . import qwen_layered_structure as layered
from .completion_metrics import artifact_bytes, persist_case_completion
from .heir_initializer_probe import durable_json, serial_lock, physical_observation
from .r10_paper_metrics import StageRecorder, run_process, read_events, summarize_events
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample

ROUND='QWEN-REMAINING-PREFILL-COMPILE'
MODEL=original.MODEL
MODE='source-bound23-original-layers-and-head-native-qwen48'
MODULE='pt_agent.experiments.qwen_remaining_prefill_compile'
COMPONENT_MODE='one-source-bound-original-layer-or-head-native-qwen48'
PREFILL_SCOPE='25 source-bound native component artifacts; typed I/O checked individually'
COMPONENTS=tuple('layer'+str(i)for i in range(1,24))+('head',)
PHASE_SECONDS=probe.PHASE_SECONDS.copy()
GIB=1024**3
RESERVE=original.TERMINAL_RESERVE


def spec(name):
    if name=='head':return dict(name=name,role='head',index=None,outputs=19,boundary_bootstraps=19)
    if name not in ('layer0',*COMPONENTS):raise ValueError('Use one original layer0..23 or head')
    index=int(name[5:])
    return dict(name=name,role='layer',index=index,outputs=32,boundary_bootstraps=32)


def _clean(observation):
    if (type(observation.get('exit_code'))is not int or observation['exit_code']!=0
            or observation.get('timed_out')is not False or observation.get('signal')is not None
            or observation.get('remaining_pids')!=[] or observation.get('signals_sent')!=[]):
        raise ValueError('Require a clean complete component process observation')


def _arity_clone(function,outputs,namespace):
    # The original own checker/marker has exactly these output-arity literals.
    # Only those constants change. Its arithmetic, native descriptors and SSA
    # checks, bundle closure and bounded lexical implementation are retained.
    expected={'_generic_header':1,'mark_segment_trace':3,'inspect_runtime_plan':4}
    if outputs not in (19,32)or function.__name__ not in expected:
        raise ValueError('Only the original32-output or19-output native ABI is supported')
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    found=sum(isinstance(node,ast.Constant)and type(node.value)is int and node.value==32 for node in ast.walk(tree))
    if found!=expected[function.__name__]:raise ValueError('Original own native ABI arity sites changed')
    constants=tuple(outputs if type(v)is int and v==32 else
        v.replace('32',str(outputs))if isinstance(v,str)else v for v in function.__code__.co_consts)
    cloned=FunctionType(function.__code__.replace(co_consts=constants),namespace,function.__name__,
        function.__defaults__,function.__closure__)
    cloned.__kwdefaults__=function.__kwdefaults__
    return cloned


def mark_trace(raw,marked,outputs):
    namespace=vars(probe).copy()
    namespace['_generic_header']=_arity_clone(probe._generic_header,outputs,namespace)
    return _arity_clone(probe.mark_segment_trace,outputs,namespace)(raw,marked)


def inspect_plan(path,expected_bootstraps,component):
    target=spec(component)
    checked=_arity_clone(probe.inspect_runtime_plan,target['outputs'],vars(probe).copy())(path,expected_bootstraps)
    checked.pop('probe_accepted',None);checked.pop('remaining23_layers_and_head',None)
    checked.update(component=target,component_native_accepted=True,full_prefill_verified=False,
        whole_generation_verified=False,linked_native_program_verified=False,prefill_compilation_scope=PREFILL_SCOPE)
    return checked


def _bindings(source,probe_case):
    """Parse frozen DSL; compare typed source records, without array payloads."""
    from ..dsl.parser import parse
    saved=original.read(probe_case/'codegen/source-bindings.json')
    rows=saved.get('layers',[]);head=saved.get('head')
    if (len(rows)!=24 or not isinstance(head,dict)or saved.get('planned_layers')!=[0]):
        raise ValueError('Require all original25 typed source bindings from the accepted layer0 probe')
    shapes={}
    for row in (*rows,head):
        for name,shape in row['weight_shapes']:
            if name in shapes and shapes[name]!=tuple(shape):raise ValueError('Original weight shape bindings conflict')
            shapes[name]=tuple(shape)
    partition=layered.split_program(parse(Path(source['program']).read_text()),shapes)
    actual=dict(layers=[layered.source_binding(s)for s in partition.layers],head=layered.source_binding(partition.head))
    if actual['layers']!=rows or actual['head']!=head:
        raise ValueError('Accepted probe source bindings differ from the frozen original DSL')
    return actual


def admission(source_case,probe_case):
    source_case,probe_case=Path(source_case).resolve(),Path(probe_case).resolve()
    source=original.frozen_source(source_case)
    record=original.read(probe_case/'result.json');complete=original.read(probe_case/'completion.json')
    if (record.get('round')!=probe.ROUND or record.get('mode')!=probe.MODE or record.get('model')!=MODEL
            or record.get('status')!='ok'or record.get('probe_accepted')is not True
            or record.get('full_model_compile_verified')is not False or record.get('probe_layer_index')!=0
            or record.get('termination')is not None or record.get('terminal_budget_error')is not None
            or any(record.get(k)!=0 for k in original.ZERO)or any(record.get(k)is not None for k in original.FORMAL)
            or complete.get('status')!='ok'or complete.get('registry_persisted')is not True
            or complete.get('required_artifacts_persisted')is not True
            or record.get('source')!=probe.admission(source_case)):
        raise ValueError('Require the complete same-source C2-frozen actual layer0 native ABI success')
    for phase in PHASE_SECONDS:
        if any(type(record.get(key,{}).get(phase))is not int or record[key][phase]!=1
               for key in ('attempted','completed')):
            raise ValueError('Layer0 must have one complete original placement/emit/trace/native chain')
        rows=[r for r in record.get('processes',[])if r.get('phase')==phase]
        if len(rows)!=1:raise ValueError('Require one actual layer0 process observation per phase')
        _clean(rows[0]['observation'])
        child=original.read(probe_case/'phases'/phase/'worker-result.json')
        if child!=record.get('phase_results',{}).get(phase)or child.get('status')!='ok'or child.get(phase+'_completed')is not True:
            raise ValueError('Layer0 phase result lineage differs')
    bindings=_bindings(source,probe_case)
    returned=original.read(probe_case/'codegen/planner-return.json')
    plan=original.read(probe_case/'codegen/refresh-plan.json')
    if (returned.get('source')!=bindings['layers'][0]or returned.get('planner_returned')is not True
            or returned.get('verify_completed')is not True or returned.get('planner')!='original.pt_agent.plan.plan_program'):
        raise ValueError('Accepted layer0 original planner source differs')
    path=Path(record['runtime_plan']).resolve();expected=plan['boots']+32
    if (path.parent!=(probe_case/'codegen/emit').resolve()or not path.name.startswith('lowered.')
            or not path.name.endswith('.runtime-plan.json')):
        raise ValueError('Layer0 actual native RuntimePlan path differs')
    native=probe.inspect_runtime_plan(path,expected)
    saved=record['phase_results']['compile']
    if any(saved.get(key)!=value for key,value in native.items()):
        raise ValueError('Layer0 saved actual native ABI differs from the complete metadata replay')
    return dict(frozen_source=source,source_case=str(source_case),probe_case=str(probe_case),
        first_layer_runtime_plan=str(path),first_layer_native=native,first_layer_expected_bootstraps=expected,
        first_layer_phase_clocks=record.get('phase_results'),first_layer_timing=record.get('timing'),
        bindings=bindings,parameter_context_changed=False,whole_generation_verified=False,
        full_prefill_verified=False,linked_native_program_verified=False,prefill_compilation_scope=PREFILL_SCOPE,
        source_binding_payloads='typed AST, original op IDs and weight shapes; no hashes')


def _component_source(args):
    return component_source(admission(args.source_case,args.probe_case),args.component)


def component_source(checked,name):
    target=spec(name)
    binding=checked['bindings']['head']if target['role']=='head'else checked['bindings']['layers'][target['index']]
    return dict(checked['frozen_source'],component=target,component_source=binding,
        first_layer_probe_case=checked['probe_case'],first_layer_native=checked['first_layer_native'],
        full_prefill_verified=False,whole_generation_verified=False)


def limits(args):
    cap=10800 if args.component else 43200
    if args.memory_gib!=24 or type(args.case_gib)is not int or not 1<=args.case_gib<=128:
        raise ValueError('Use finite AS24 and at most128GiB accumulated artifacts')
    if type(args.seconds)is not int or not 1<=args.seconds<=cap:raise ValueError('Use the finite explicit complete allowance')
    return dict(address_space_gib=24,subtree_rss_gib=14,minimum_available_gib=1.25,
        case_bytes=args.case_gib*GIB,minimum_free_bytes=10*GIB,file_gib=64,phases=PHASE_SECONDS.copy(),
        source_metadata_seconds=600,complete_batch_seconds=args.seconds,concurrency=1,
        previous_probe_artifacts_in_accumulated_guard=True)


def _budget(args,out):
    size=artifact_bytes(out);prior=artifact_bytes(args.probe_case)
    free=shutil.disk_usage(out).free
    if size+prior+RESERVE>args.case_gib*GIB or free<10*GIB:
        raise RuntimeError('Accumulated probe and continuation artifacts or free disk allowance exhausted')
    return dict(new_case_bytes=size,source_probe_bytes=prior,total_bytes=size+prior,free_bytes=free)


def _load(args,result):
    info,program=original._load_program(args,result)
    partition=layered.split_program(program,{name:array.shape for name,array in info.graph.weights.items()})
    layers,head=layered.callbacks(partition,info.graph.weights)
    callback=head if args.component=='head'else layers[spec(args.component)['index']]
    if layered.source_binding(callback.segment)!=result['source']['component_source']:
        raise ValueError('Actual loaded original component differs from the admitted frozen source')
    from .qwen_component_cost import already_scoped_lowering
    callback=already_scoped_lowering(callback)
    return callback


def export_component(backend,callback,bound):
    if callback.segment.role=='layer':return layered.export_layer(backend,callback=callback,bound_plan=bound)
    if (callback.segment.role!='head'or callback.segment.index is not None or backend.slots!=layered.SLOTS
            or bound.segment is not callback.segment or bound.source!=layered.source_binding(callback.segment)):
        raise ValueError('Retain the exact original typed head and its source-bound plan')
    state=layered.staged.Segments(backend)
    state.metadata.update(source_binding=bound.source,scope='original-head-native-export-preparation',
        full_prefill_verified=False,whole_generation_verified=False,native_metadata_verified=False)
    try:
        output=state.run('head',state.inputs,bound.plan,callback)
        if len(output)!=19:raise ValueError('Retain all19 original vocabulary head blocks')
        logits=state.normalize(output,'head-to-selector');state.finish(logits)
        if state.metadata['boundary_bootstraps']!=19:raise ValueError('Head requires19 actual boundary boot calls')
        state.metadata.update(status='ok',physical_inputs=34,logical_output_blocks=19,physical_output_blocks=19,
            symbolic_output_levels=[h.level for h in logits],mask_alias_outputs=[],
            explicit_boundary_bootstrap_calls=19,internal_plan_bootstrap_calls=state.metadata['segments'][0]['refresh']['boots'])
        return state
    except BaseException:
        state.metadata['status']='error';raise


def placement(args,result,recorder):
    from ..kernels.compact_linear import compact_transformer_projections
    from ..backends.streaming_count import StreamingCountBackend
    from . import qwen_component_plan_cache as cache
    callback=_load(args,result);target=spec(args.component);binding=layered.source_binding(callback.segment)
    cached=cache.worker_plan(args,result['source'],callback.segment,recorder)
    if cached is None:
        planner=original._planner_namespace()if target['role']=='head'else probe.planner_namespace()
        with recorder.span('original_'+args.component+'_plan'),original._projection_backend_scope(),compact_transformer_projections():
            plan,dag,placed=planner['plan_program'](callback.segment.program,callback.weights,profile='qwen48',frontend_normalize=True)
            bound=layered.bind_plan(callback.segment,plan,planned_source=binding)
            del dag,placed;gc.collect()
            verified=planner['verify_program'](callback.segment.program,callback.weights,plan,profile='qwen48')
    else:
        plan=cached['plan'];verified=cached['verification']
        bound=layered.bind_plan(callback.segment,plan,planned_source=binding)
    with recorder.span('independent_'+args.component+'_complete_count'):
        backend=StreamingCountBackend(layered.SLOTS,input_level=5,boot_out_level=5)
        state=export_component(backend,callback,bound);counted=dict(stats=backend.stats(),interface=state.metadata)
    if cached is not None:cache.compare_count(cached['count'],counted)
    if (verified['boots']!=plan['boots']or counted['stats']['counts']['boot']!=plan['boots']+target['boundary_bootstraps']
            or state.metadata['internal_plan_bootstrap_calls']!=plan['boots']):
        raise ValueError('Original component planner/verify and complete Count schedules differ')
    durable_json(args.codegen/'source-binding.json',binding);durable_json(args.codegen/'refresh-plan.json',plan)
    durable_json(args.codegen/'planner-return.json',dict(source=binding,planner='original.pt_agent.plan.plan_program',
        planner_returned=True,verify_completed=True,verification=verified,
        **(dict(plan_cache=cached['provenance']) if cached is not None else {})))
    durable_json(args.codegen/'count-replay.json',counted)
    result.update(placement_completed=True,component=target,planned_bootstraps=plan['boots'],
        boundary_bootstraps=target['boundary_bootstraps'],expected_compiled_bootstraps=plan['boots']+target['boundary_bootstraps'],
        count_stats=counted['stats'],plan_verification=verified,internal_planner_latency_us=plan.get('latency_us'),
        planner_cost_scope='Original internal plan; real boundary boots counted separately in complete Count/Emit')
    if cached is not None:result['plan_cache']=cached['provenance']


def emit(args,result,recorder):
    from ..backends.streaming_emit import StreamingEmitBackend
    callback=_load(args,result);returned=original.read(args.codegen/'planner-return.json')
    if (returned.get('planner_returned')is not True or returned.get('verify_completed')is not True
            or returned.get('planner')!='original.pt_agent.plan.plan_program'):
        raise ValueError('Require the actual original planner return and verify')
    bound=layered.bind_plan(callback.segment,original.read(args.codegen/'refresh-plan.json'),planned_source=returned['source'])
    counted=original.read(args.codegen/'count-replay.json')['stats']
    backend=StreamingEmitBackend(layered.SLOTS,directory=args.codegen/'emit',frontend=args.frontend,
        max_constant_bytes=int(args.case_gib*GIB-GIB))
    try:
        with recorder.span('original_'+args.component+'_complete_emit'):
            state=export_component(backend,callback,bound);stats=backend.stats()
            if any(stats[key]!=counted[key]for key in ('counts','ops_total','distinct_consts')):
                raise ValueError('Original complete component Count/Emit schedules differ')
            source=backend.write();durable_json(args.codegen/'component-interface.json',state.metadata)
            result.update(emit_completed=True,component=spec(args.component),trace_source=str(source),emit_stats=stats,
                physical_inputs=34,physical_outputs=spec(args.component)['outputs'],selector_input_retained=True)
    finally:
        if not backend.finished:backend.close()


def worker(args):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    args.work_dir.mkdir(parents=True,exist_ok=False);args.codegen.mkdir(exist_ok=True)
    recorder=StageRecorder(args.work_dir/'events.jsonl');started=time.perf_counter_ns()
    result=dict(round=ROUND,model=MODEL,mode=COMPONENT_MODE,status='running',error=None,phase=args.worker_phase,
        component=spec(args.component),component_native_accepted=False,full_prefill_verified=False,whole_generation_verified=False,
        linked_native_program_verified=False,prefill_compilation_scope=PREFILL_SCOPE,
        **dict.fromkeys(original.ZERO,0),**dict.fromkeys(original.FORMAL))
    durable_json(args.work_dir/'worker-result.json',result)
    try:
        result['source']=_component_source(args)
        for name in ('metadata_template','frontend','profile'):setattr(args,name,Path(result['source'][name]))
        from .r29_local_compile_matrix import native_metadata_commands
        with native_metadata_commands(original.read(args.metadata_template)):
            if args.worker_phase in ('placement','emit'):
                import torch
                torch.set_num_threads(1);torch.set_num_interop_threads(1)
                (placement if args.worker_phase=='placement'else emit)(args,result,recorder)
            else:
                from .. import toolchain
                host=toolchain.host_paths();command=probe.native_command(args,host)
                durable_json(args.work_dir/'native-argv.json',command)
                environment=dict(os.environ,HECATE=str(host['hecate']),HECATE_BUILD=str(host['hecate_build']),
                    LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'])
                observed=run_process(command,cwd=args.codegen/'emit',log=args.work_dir/'native.log',
                    resources=args.work_dir/'native-resources.jsonl',timeout=args.phase_seconds,
                    mem_gib=24,fsize_gib=64,env=environment)
                result['process']=observed;_clean(observed)
                if args.worker_phase=='trace':
                    result['trace_marker']=mark_trace(args.codegen/'emit/candidate_trace.mlir',
                        args.codegen/'emit/candidate_trace.segment.earth.mlir',spec(args.component)['outputs'])
                    result['trace_completed']=True
                else:
                    candidates=list((args.codegen/'emit').glob('lowered.*.runtime-plan.json'))
                    if len(candidates)!=1:raise ValueError('Require one unique actual component RuntimePlan')
                    expected=original.read(args.codegen/'refresh-plan.json')['boots']+spec(args.component)['boundary_bootstraps']
                    result.update(inspect_plan(candidates[0],expected,args.component),compile_completed=True,runtime_plan=str(candidates[0]))
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        result['elapsed_ns']=time.perf_counter_ns()-started;durable_json(args.work_dir/'worker-result.json',result)
    return result


def phase_command(args,phase,work,remaining=None):
    seconds=PHASE_SECONDS[phase]if remaining is None else min(PHASE_SECONDS[phase],max(1,int(remaining)-120))
    return [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(seconds+60),'--python',str(args.intake_python),
        '--','-B','-m',MODULE,'--worker-phase',phase,'--source-case',str(args.source_case),'--probe-case',str(args.probe_case),
        '--component',args.component,'--work-dir',str(work),'--codegen',str(args.out/'codegen'),
        '--memory-gib','24','--case-gib',str(args.case_gib),'--seconds',str(args.seconds),'--phase-seconds',str(seconds)]+(
        ['--diagnostic-elide-final-ir']if args.diagnostic_elide_final_ir else [])


def component_execute(args):
    ns=vars(original).copy();ns.update(ROUND=ROUND,MODEL=MODEL,MODE=COMPONENT_MODE,MODULE=MODULE,
        PHASE_SECONDS=PHASE_SECONDS,resource_limits=limits,frozen_source=lambda case:_component_source(args),
        phase_command=phase_command,budget=lambda a,out:_budget(a,out),__name__=__name__)
    def clone(fn):
        copied=FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
        copied.__kwdefaults__=fn.__kwdefaults__;return copied
    terminal=clone(original._terminal)
    def finish(a,out,result,started,revision,recorder):
        result.update(component=spec(a.component),full_prefill_verified=False,whole_generation_verified=False,
            linked_native_program_verified=False,prefill_compilation_scope=PREFILL_SCOPE,
            component_native_accepted=False,scope='One original component, independent real native ABI; no generation claim')
        child=result.get('phase_results',{}).get('compile')or{}
        result['component_native_accepted']=(result['status']=='ok'and result['completed'].get('compile')==1
            and child.get('component_native_accepted')is True and child.get('component')==spec(a.component))
        for module in (probe,layered):shutil.copyfile(module.__file__,out/'source'/Path(module.__file__).name)
        terminal(a,out,result,started,revision,recorder)
    saved_write=ns['write_json']
    from . import qwen_component_plan_cache as cache
    proof=[None]
    supervised=ns['run_process']
    def strict_process(command,**kwargs):
        observed=supervised(command,**kwargs)
        # Keep the real observation even when the original outer function has
        # not yet received it. A non-clean child never starts the next phase.
        phase=command[command.index('--worker-phase')+1]
        durable_json(args.out/(phase+'-supervision-observation.json'),observed)
        _clean(observed)
        return observed
    def save(path,value):
        if Path(path).name=='result.json'and isinstance(value,dict)and value.get('mode')==COMPONENT_MODE:
            value['component_native_accepted']=(value.get('status')=='ok'and value.get('completed',{}).get('compile')==1
                and (value.get('phase_results',{}).get('compile')or{}).get('component_native_accepted')is True)
        saved=saved_write(path,value)
        if Path(path).name=='source-record.json':cache.persist(Path(path).parent,proof[0])
        return saved
    ns.update(_terminal=finish,write_json=save,run_process=strict_process)
    deadline=time.monotonic()+args.seconds;gate=ns['frozen_source']
    def finite(case):
        signal.setitimer(signal.ITIMER_REAL,min(600.,max(.001,deadline-time.monotonic())))
        source=gate(case);proof[0]=cache.prepare_outer(case,source)
        return source
    ns['frozen_source']=finite
    return clone(original.execute)(args)


def accept_component(case,name,checked):
    record=original.read(case/'result.json');complete=original.read(case/'completion.json');target=spec(name)
    wanted=checked['bindings']['head']if name=='head'else checked['bindings']['layers'][target['index']]
    if (record.get('status')!='ok'or record.get('mode')!=COMPONENT_MODE or record.get('round')!=ROUND
            or record.get('model')!=MODEL or record.get('component')!=target
            or record.get('component_native_accepted')is not True or record.get('whole_generation_verified')is not False
            or record.get('linked_native_program_verified')is not False or record.get('prefill_compilation_scope')!=PREFILL_SCOPE
            or record.get('full_prefill_verified')is not False or record.get('terminal_budget_error')is not None
            or record.get('termination')is not None or record.get('failed_phase')is not None
            or record.get('source')!=component_source(checked,name)
            or any(record.get(k)!=0 for k in original.ZERO)or any(record.get(k)is not None for k in original.FORMAL)
            or complete.get('status')!='ok'or complete.get('registry_persisted')is not True
            or complete.get('required_artifacts_persisted')is not True):
        raise ValueError('Require this complete same-source component and its durable native ABI acceptance')
    for phase in PHASE_SECONDS:
        if any(type(record.get(key,{}).get(phase))is not int or record[key][phase]!=1 for key in ('attempted','completed')):
            raise ValueError('One complete original component phase chain required')
        observed=[r for r in record.get('processes',[])if r.get('phase')==phase]
        if len(observed)!=1:raise ValueError('Require one component phase observation')
        _clean(observed[0]['observation'])
        child=original.read(case/'phases'/phase/'worker-result.json')
        if (child!=record.get('phase_results',{}).get(phase)or child.get('status')!='ok'
                or child.get(phase+'_completed')is not True or child.get('component')!=target
                or child.get('source')!=component_source(checked,name)):
            raise ValueError('Every actual component phase must preserve this same frozen source')
    returned=original.read(case/'codegen/planner-return.json');plan=original.read(case/'codegen/refresh-plan.json')
    if (returned.get('source')!=wanted or returned.get('planner_returned')is not True
            or returned.get('verify_completed')is not True or returned.get('planner')!='original.pt_agent.plan.plan_program'):
        raise ValueError('Actual original component planner source binding differs')
    counted=original.read(case/'codegen/count-replay.json')['stats']
    emitted=record['phase_results']['emit'];interface=original.read(case/'codegen/component-interface.json')
    if (any(emitted.get('emit_stats',{}).get(key)!=counted.get(key)for key in ('counts','ops_total','distinct_consts'))
            or counted['counts']['boot']!=plan['boots']+target['boundary_bootstraps']
            or interface.get('physical_inputs')!=34 or interface.get('physical_output_blocks')!=target['outputs']
            or interface.get('explicit_boundary_bootstrap_calls')!=target['boundary_bootstraps']
            or interface.get('internal_plan_bootstrap_calls')!=plan['boots']):
        raise ValueError('Complete component Count/Emit/interface boundary contracts differ')
    path=Path(record['runtime_plan']).resolve()
    if path.parent!=(case/'codegen/emit').resolve():raise ValueError('Native artifact must belong to its component case')
    expected=plan['boots']+target['boundary_bootstraps'];native=inspect_plan(path,expected,name)
    if record.get('compiled_bootstraps')!=expected:raise ValueError('Outer actual compiled bootstrap count differs')
    child=original.read(case/'phases/compile/worker-result.json')
    if child!=record.get('phase_results',{}).get('compile')or any(child.get(k)!=v for k,v in native.items()):
        raise ValueError('Saved native component ABI differs from complete metadata replay')
    return dict(status='ok',component=target,source_binding=wanted,case=str(case),runtime_plan=str(path),
        native=native,expected_compiled_bootstraps=expected,phase_results=record['phase_results'],
        timing=record.get('timing'),completion=str(case/'completion.json'))


def _close(args,out,result,started,recorder):
    # Later terminal resource failures revoke aggregate acceptance while clean
    # per-component observations remain visible.
    result['timing']=summarize_events(read_events(out/'events.jsonl'))
    result['timing_scope']='Fresh remaining23-layer/head continuation; first-layer source clocks retained separately, never added'
    result.update(linked_native_program_verified=False,prefill_compilation_scope=PREFILL_SCOPE)
    for _ in range(20):
        result['full_prefill_verified']=(result.get('status')=='ok'and result.get('accepted_components')==25
            and all(row['status']=='ok'for row in result['components'].values()))
        result['whole_generation_verified']=False
        durable_json(out/'result.json',result);size=artifact_bytes(out)
        if size+artifact_bytes(args.probe_case)+RESERVE>args.case_gib*GIB:
            result.update(status='error',error='Accumulated artifact allowance including completion reserve exhausted',
                termination=dict(cause='accumulated_artifact_allowance',completion_reserve_bytes=RESERVE))
            result['full_prefill_verified']=False
            durable_json(out/'result.json',result);size=artifact_bytes(out)
        if result.get('artifact_bytes')==size:break
        result['artifact_bytes']=size
    else:raise RuntimeError('Continuation terminal metadata did not stabilize')
    row=dict(run_id=out.name,round=ROUND,model=MODEL,mode=MODE,git_rev=result.get('git_rev'),
        metrics={k:result[k]for k in ('status','accepted_components','full_prefill_verified','whole_generation_verified',
            'linked_native_program_verified','prefill_compilation_scope','compiled_bootstraps')},
        artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
    with (original.paths.RESULTS_DIR/'registry.jsonl').open('a+')as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()):raise ValueError('Preserve existing registry identity')
        stream.seek(0,os.SEEK_END);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],timing=result['timing'],required_artifact_bytes=artifact_bytes(out))


def execute(args):
    resource=limits(args);started=time.perf_counter_ns();deadline=time.monotonic()+args.seconds
    out=args.out.resolve();root=original.paths.RESULTS_DIR.resolve()
    if out.exists()or out.is_symlink()or out==root or not out.is_relative_to(root):raise ValueError('Use one new isolated continuation result case')
    if any(out.is_relative_to(Path(p).resolve())or Path(p).resolve().is_relative_to(out)for p in (args.source_case,args.probe_case)):
        raise ValueError('Retain original frozen and layer0 cases outside the new continuation')
    with serial_lock(root):
        out.mkdir();recorder=StageRecorder(out/'events.jsonl');active=[None];stop=threading.Event();failures=[]
        from ..registry import git_rev
        result=dict(round=ROUND,model=MODEL,mode=MODE,status='running',error=None,git_rev=git_rev(),limits=resource,
            components={name:dict(status='not_run',component=spec(name))for name in ('layer0',*COMPONENTS)},
            accepted_components=0,compiled_bootstraps=None,full_prefill_verified=False,whole_generation_verified=False,
            linked_native_program_verified=False,prefill_compilation_scope=PREFILL_SCOPE,
            processes=[],failed_component=None,**dict.fromkeys(original.ZERO,0),**dict.fromkeys(original.FORMAL))
        old={sig:signal.getsignal(sig)for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1,signal.SIGALRM)}
        def incoming(sig,frame):
            result['termination']=dict(signal=sig,reason=failures[-1]if failures else'incoming_signal')
            raise InterruptedError('Finite remaining Qwen compilation interrupted')
        for sig in old:signal.signal(sig,incoming)
        def check():
            if failures:raise RuntimeError(failures[-1])
            if time.monotonic()>=deadline:raise TimeoutError('Complete43200-second continuation allowance exhausted')
            _budget(args,out)
            observed=physical_observation(_latest_rss_sample(active[0])if active[0]else None,
                _available_ram_bytes(),shutil.disk_usage(out).free)
            if observed['stop']:
                result['resource_guard']=observed;raise RuntimeError(str(observed))
        def monitor():
            while not stop.wait(.25):
                try:check()
                except BaseException as exc:
                    failures.append(type(exc).__name__+': '+str(exc))
                    durable_json(out/'guard-failure.json',dict(reason=failures[-1]));os.kill(os.getpid(),signal.SIGUSR1);return
        watcher=threading.Thread(target=monitor,daemon=True)
        try:
            signal.setitimer(signal.ITIMER_REAL,min(600.,args.seconds))
            with recorder.span('C2-first-layer-native-source-admission'):checked=admission(args.source_case,args.probe_case)
            signal.setitimer(signal.ITIMER_REAL,0)
            result['source']=checked;durable_json(out/'source-bindings.json',checked['bindings'])
            result['components']['layer0']=dict(status='ok',component=spec('layer0'),case=checked['probe_case'],
                source_binding=checked['bindings']['layers'][0],runtime_plan=checked['first_layer_runtime_plan'],
                native=checked['first_layer_native'],phase_results=checked['first_layer_phase_clocks'],timing=checked['first_layer_timing'])
            result['accepted_components']=1
            snapshots=out/'source';snapshots.mkdir()
            for module in (sys.modules[__name__],probe,original,layered):shutil.copyfile(module.__file__,snapshots/Path(module.__file__).name)
            watcher.start();check()
            for name in COMPONENTS:
                check();remaining=deadline-time.monotonic()
                if remaining<=180:raise TimeoutError('Insufficient remaining component supervision allowance')
                seconds=min(10800,int(remaining)-120);case=out/name
                command=[sys.executable,'-B','-m',MODULE,'--source-case',str(args.source_case),'--probe-case',str(args.probe_case),
                    '--out',str(case),'--component',name,'--intake-python',str(args.intake_python),
                    '--memory-gib','24','--case-gib',str(args.case_gib),'--seconds',str(seconds)]+(
                    ['--diagnostic-elide-final-ir']if args.diagnostic_elide_final_ir else [])
                result['failed_component']=name;result['components'][name]['status']='running';durable_json(out/'result.json',result)
                active[0]=out/(name+'-resources.jsonl')
                with recorder.span('component-'+name,argv=command):
                    observed=run_process(command,cwd=original.paths.REPO/'DSL/DSL_agent/scripts',log=out/(name+'.log'),
                        resources=active[0],timeout=min(seconds+60,max(.01,deadline-time.monotonic())),mem_gib=24,fsize_gib=64,
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'))
                active[0]=None;result['processes'].append(dict(component=name,argv=command,observation=observed));_clean(observed)
                result['components'][name]=accept_component(case,name,checked)
                result['accepted_components']+=1;check();durable_json(out/'result.json',result)
            result.update(status='ok',failed_component=None,compiled_bootstraps=sum(row['native']['compiled_bootstraps']for row in result['components'].values()))
        except BaseException as exc:
            result.update(status='error',error=type(exc).__name__+': '+str(exc))
            if result['failed_component']:
                row=result['components'][result['failed_component']];row.update(status='error',error=result['error'])
                target=out/result['failed_component']/'result.json'
                if target.is_file():row['partial_result']=str(target)
        finally:
            active[0]=None;stop.set()
            if watcher.is_alive():watcher.join(timeout=3)
            signal.setitimer(signal.ITIMER_REAL,0)
            try:_close(args,out,result,started,recorder)
            finally:
                for sig,handler in old.items():signal.signal(sig,handler)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for key in ('source-case','probe-case'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--out',type=Path);parser.add_argument('--component',choices=COMPONENTS)
    parser.add_argument('--worker-phase',choices=tuple(PHASE_SECONDS))
    parser.add_argument('--through',choices=('compile',),default='compile')
    for key in ('work-dir','codegen'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--intake-python',type=Path,default=original.paths.INTAKE_PYTHON)
    parser.add_argument('--memory-gib',type=float,default=24.);parser.add_argument('--case-gib',type=int,default=128)
    parser.add_argument('--seconds',type=int);parser.add_argument('--phase-seconds',type=int)
    parser.add_argument('--diagnostic-elide-final-ir',action='store_true');args=parser.parse_args(argv)
    if args.seconds is None:args.seconds=10800 if args.component else 43200
    limits(args)
    if args.worker_phase:
        if not args.component or args.work_dir is None or args.codegen is None:parser.error('Require explicit component/work/codegen for pure worker')
        if type(args.phase_seconds)is not int or not 1<=args.phase_seconds<=PHASE_SECONDS[args.worker_phase]:parser.error('Use finite original phase allowance')
        result=worker(args)
    else:
        if args.out is None:parser.error('Require a fresh explicit result case')
        result=component_execute(args)if args.component else execute(args)
    print(json.dumps(dict(status=result['status'],error=result.get('error'),full_prefill_verified=result.get('full_prefill_verified',False),
        whole_generation_verified=False,linked_native_program_verified=False,prefill_compilation_scope=PREFILL_SCOPE)))
    return int(result['status']!='ok')


if __name__=='__main__':raise SystemExit(main())
