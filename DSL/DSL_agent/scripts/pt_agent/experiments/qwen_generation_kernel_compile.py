"""Finite C2-bound first/terminal generation-kernel native compilation only.

Two independent artifacts retain original selector/embedding arithmetic and
every output's effective level. Three external connectors belong to the client.
No model forward, text tokenization, dataset, API or ciphertext execution is exposed.
Original program loading retains its local tokenizer-object construction.
"""
import argparse
import ast
from collections import Counter
import copy
from datetime import datetime, timezone
import fcntl
import gc
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import textwrap
import threading
import time
from types import FunctionType

from . import qwen_prefill_compile as original
from . import qwen_generation_kernels as kernels
from .completion_metrics import artifact_bytes, persist_case_completion
from .heir_initializer_probe import durable_json, serial_lock, physical_observation
from .r10_paper_metrics import StageRecorder, run_process, read_events, summarize_events
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample

ROUND='QWEN-GENERATION-KERNEL-COMPILE'
MODEL=original.MODEL
MODE='same-c2-first20to22-terminal19to19-native-qwen48'
COMPONENT_MODE='one-source-bound-generation-kernel-native-qwen48'
MODULE='pt_agent.experiments.qwen_generation_kernel_compile'
PHASE_SECONDS=dict(placement=7200,emit=3600,trace=1800,compile=7200)
KINDS=('first','terminal')
GIB=1024**3
RESERVE=original.TERMINAL_RESERVE


def spec(kind):
    if kind not in KINDS:raise ValueError('Use the original first or terminal kernel')
    return dict(kind=kind,inputs=20 if kind=='first'else 19,outputs=22 if kind=='first'else 19,
        boundary_positions=[],boundary_bootstraps=0,future_boundary_bootstraps=3 if kind=='first'else 0,
        future_boundary_positions=[19,20,21]if kind=='first'else [])


def clean(observed):
    if (type(observed.get('exit_code'))is not int or observed['exit_code']!=0 or observed.get('timed_out')is not False
            or observed.get('signal')is not None or observed.get('remaining_pids')!=[] or observed.get('signals_sent')!=[]):
        raise ValueError('Require one clean complete phase with no residual process')


def _info_dir(path):
    path=Path(path).resolve();return path.parent if path.name=='model_info.json'else path


def admission(case):
    """Original C2 gate and saved public state metadata; never sample arrays."""
    case=Path(case).resolve();frozen=original.json_value(original.frozen_source(case))
    generation=original.read(case/'generation-input.json');info=original.read(_info_dir(frozen['model_info'])/'model_info.json')
    index=generation.get('index');split=info['split']
    if (type(index)is not int or index not in split['calibration'] or index<0 or index>=len(split['sample_ids'])
            or generation.get('sample_id')!=split['sample_ids'][index]):
        raise ValueError('Saved generation input must be this exact original calibration sample')
    linked_path=Path(frozen['original_source']['previous_linked_math']);linked=original.read(linked_path)
    own_path=Path(linked['source']['cached_sources']['own_output_result']);own=original.read(own_path)
    worker=own.get('worker')or{};source=worker.get('source')or{}
    if (linked.get('status')!='ok'or linked.get('linked_packed_math_verified')is not True
            or own.get('status')!='ok'or worker.get('status')!='ok'or worker.get('phase')!='own-output-tensor'
            or worker.get('own_output_generation_completed')is not True
            or _info_dir(source['model_info'])!=_info_dir(frozen['model_info'])):
        raise ValueError('Retain the original same-model saved step0 state lineage')
    reference=Path(source['original_hf_reference']);hf=original.read(reference)
    if (hf.get('status')!='ok'or hf.get('hf_reference_completed')is not True or hf.get('hf_forward_completed')!=6
            or Path(hf['original_hf_source']).resolve()!=Path(frozen['original_model']).resolve()
            or hf.get('original_model_identity')!=dict(layers=24,hidden_size=896,vocabulary=151936,
                use_cache=False,sequence_length=32,dtype='float64',attention='eager')):
        raise ValueError('Retain the completed original24-layer HF state metadata')
    suffix=generation['sample_id'].rsplit(':',1)[-1]
    if not suffix.isdecimal():raise ValueError('Original input row provenance must remain explicit')
    sample='train:'+suffix
    trajectories=[row for row in hf.get('trajectories',[])if row.get('sample_id')==sample]
    rows=[row for row in worker.get('sample_rows',[])if(row.get('sample_id'),row.get('step'))==(sample,0)]
    if len(trajectories)!=1 or len(rows)!=1:raise ValueError('One exact retained original step0 trajectory required')
    steps=[row for row in trajectories[0]['steps']if row.get('step')==0]
    if len(steps)!=1:raise ValueError('One original HF step0 is required')
    state={key:copy.deepcopy(steps[0][key])for key in ('step','prefix_length','prefix_token_ids','attention_mask','sel')}
    if any(rows[0]['input_state'].get(key)!=value for key,value in state.items()):
        raise ValueError('Original HF and own-output step0 input metadata differ')
    state['sample_id']=generation['sample_id'];binding=kernels.bind_source(case,state)
    if binding.snapshot()['frozen']!=frozen:raise ValueError('Original frozen source changed during state admission')
    return dict(frozen_source=frozen,binding=binding.snapshot(),initial_state=state,generation=generation,
        lineage=dict(linked=str(linked_path),own_output=str(own_path),reference=str(reference),original_sample_id=sample),
        original_token0_provenance='Original initial34[0]; first kernel logical input19, used in original token0*0 mask arithmetic',
        saved_state_metadata_verified=True,initial_ciphertext_payload_verified=False,
        linked_native_program_verified=False,whole_generation_verified=False,full_prefill_verified=False)


def component_source(checked,kind):return dict(checked,kernel=spec(kind))


def limits(args):
    if args.memory_gib!=24 or type(args.case_gib)is not int or not 1<=args.case_gib<=128:
        raise ValueError('Use finite AS24/case through128GiB, original physical guard unchanged')
    if type(args.seconds)is not int or not 1<=args.seconds<=21600:raise ValueError('Use finite total through21600seconds')
    return dict(address_space_gib=24,subtree_rss_gib=14,minimum_available_gib=1.25,file_gib=64,
        case_bytes=args.case_gib*GIB,minimum_free_bytes=10*GIB,complete_batch_seconds=args.seconds,
        source_metadata_seconds=600,phases=PHASE_SECONDS.copy(),concurrency=1)


def budget(args,out):
    root=args.batch_out if getattr(args,'batch_out',None)is not None else out
    size=artifact_bytes(root);free=shutil.disk_usage(root).free
    if size+RESERVE>args.case_gib*GIB or free<10*GIB:raise RuntimeError('Accumulated kernel case or completion/disk reserve exhausted')
    return dict(case_bytes=size,free_bytes=free)


def _load(args,result):
    info,program=original._load_program(args,result);weights=info.graph.weights
    if len(weights)!=559 or any(str(a.dtype)!='float64'or a.flags.writeable for a in weights.values()):
        raise ValueError('Retain all559 original readonly float64 weights')
    made=kernels.make_kernels(kernels.SourceBinding(result['source']['binding']),weights)
    kernel=made[0 if args.kind=='first'else 1];made[1 if args.kind=='first'else 0].close()
    metadata=[[name,list(a.shape),str(a.dtype),bool(a.flags.writeable)]for name,a in sorted(weights.items())]
    return info,kernel,metadata


def interface_record(record):
    """Label unchanged original output metadata; external client owns3 boots."""
    record=copy.deepcopy(record);kind=record['kind'];target=spec(kind)
    if (record.get('inputs')!=target['inputs']or record.get('outputs')!=target['outputs']
            or len(record.get('symbolic_output_levels',[]))!=target['outputs']
            or record.get('explicit_boundary_bootstraps_in_this_adapter')!=0
            or record.get('caller_required_boundary_positions')!=target['future_boundary_positions']):
        raise ValueError('Require the original complete raw kernel output and caller connector contract')
    record.update(external_boundary_bootstraps_in_component=0,future_boundary_bootstraps=target['future_boundary_bootstraps'],
        external_connector_owner='Original Go RunTwoTokens BootstrapBoundary calls',output_levels_relabelled=False)
    return record


def placement(args,result,recorder):
    owner=kernel=None
    try:
        owner,kernel,weights=_load(args,result)
        with recorder.span('original_'+args.kind+'_planner_verify'):
            planned,timing=kernels.plan_kernel(kernel,metadata_template=original.read(args.metadata_template))
        with recorder.span('independent_complete_kernel_count'):
            counted=kernels.count_kernel(kernel,planned,metadata_template=original.read(args.metadata_template))
            interface=interface_record(counted['interface']);stats=counted['stats']
        expected=planned.refresh_plan['boots']+spec(args.kind)['boundary_bootstraps']
        if stats['counts']['boot']!=expected:raise ValueError('Complete Count differs from actual original plan plus caller boots')
        durable_json(args.codegen/'planned-kernel.json',dict(source=planned.source,refresh_plan=planned.refresh_plan,replay=planned.replay))
        durable_json(args.codegen/'planner-return.json',dict(source=kernel.source(),weights=weights,planner_returned=True,
            verify_completed=True,verification=planned.replay,timing=timing))
        durable_json(args.codegen/'count-replay.json',dict(stats=stats,interface=interface))
        result.update(placement_completed=True,planned_bootstraps=planned.refresh_plan['boots'],
            boundary_bootstraps=spec(args.kind)['boundary_bootstraps'],expected_compiled_bootstraps=expected,
            count_stats=stats,interface=interface,plan_verification=planned.replay)
    finally:
        if kernel is not None:kernel.close()
        owner=kernel=None;gc.collect()


def emit(args,result,recorder):
    owner=kernel=physical=None
    try:
        owner,kernel,weights=_load(args,result);returned=original.read(args.codegen/'planner-return.json')
        if (returned.get('source')!=kernel.source()or returned.get('weights')!=weights
                or returned.get('planner_returned')is not True or returned.get('verify_completed')is not True):
            raise ValueError('Actual original planner, readonly weights and same bound kernel required')
        saved=original.read(args.codegen/'planned-kernel.json');planned=kernels.PlannedKernel(**saved)
        counted=original.read(args.codegen/'count-replay.json')
        with recorder.span('original_kernel_complete_streaming_emit'):
            emitted=kernels.emit_kernel(kernel,planned,metadata_template=original.read(args.metadata_template),
                directory=args.codegen/'emit',frontend=args.frontend,max_constant_bytes=int(args.case_gib*GIB-GIB),counted=counted)
            interface=interface_record(emitted['interface']);stats=emitted['stats']
            if (interface!=counted['interface']or any(stats[k]!=counted['stats'][k]for k in ('counts','ops_total','distinct_consts'))):
                raise ValueError('Complete kernel Count/Emit arithmetic, constants and per-output levels differ')
            durable_json(args.codegen/'kernel-interface.json',interface)
            result.update(emit_completed=True,trace_source=emitted['trace_source'],emit_stats=stats,interface=interface)
    finally:
        if physical is not None:
            try:
                if not physical.finished:physical.close()
            finally:kernels._release_physical(physical)
        if kernel is not None:kernel.close()
        owner=kernel=physical=None;gc.collect()


def mark_trace(raw,marked,kind):
    from . import qwen_layer_native_probe as probe
    target=spec(kind);namespace=vars(probe).copy()
    def clone(fn):
        tree=ast.parse(textwrap.dedent(inspect.getsource(fn)))
        expected={'_generic_header':{34:1,32:1},'mark_segment_trace':{34:1,32:3}}
        for value,count in expected[fn.__name__].items():
            if sum(isinstance(n,ast.Constant)and type(n.value)is int and n.value==value for n in ast.walk(tree))!=count:
                raise ValueError('Original generic native ABI arity seam changed')
        consts=tuple(target['inputs']if type(v)is int and v==34 else target['outputs']if type(v)is int and v==32 else v
            for v in fn.__code__.co_consts)
        return FunctionType(fn.__code__.replace(co_consts=consts),namespace,fn.__name__,fn.__defaults__,fn.__closure__)
    namespace['_generic_header']=clone(probe._generic_header)
    return clone(probe.mark_segment_trace)(raw,marked)


def inspect_plan(path,expected_bootstraps,interface,kind):
    """Variable per-output levels, complete native SSA and bundle stat closure."""
    from . import qwen_layer_native_probe as probe
    from .native_component_matrix import array_items
    path=Path(path);target=spec(kind);levels=interface.get('symbolic_output_levels')
    if (type(expected_bootstraps)is not int or expected_bootstraps<target['boundary_bootstraps']
            or not isinstance(levels,list)or len(levels)!=target['outputs']
            or any(type(x)is not int or not 1<=x<=5 for x in levels)
            or any(levels[i]!=5 for i in target['boundary_positions'])or probe._field(path,'format_version')!=1):
        raise ValueError('Require exact complete Count output levels and native format1')
    inputs=list(array_items(path,'external_inputs'));outputs=list(array_items(path,'final_outputs'))
    if (len(inputs)!=target['inputs']or len(set(inputs))!=len(inputs)or len(outputs)!=target['outputs']
            or len(set(outputs))!=len(outputs)or set(inputs)&set(outputs)):
        raise ValueError('Complete kernel actual native input/output ABI differs')
    def number(name):
        if not isinstance(name,str)or not name.isdecimal()or str(int(name))!=name:raise ValueError('Original consecutive numeric native SSA required')
        return int(name)
    wanted=dict(zip(inputs,[5]*len(inputs)));wanted.update(zip(outputs,levels));descriptions={};count=0
    for row in array_items(path,'values'):
        if (number(row['id'])!=count or row.get('context')!='qwen-bootstrap48'or row.get('ntt')is not True
                or row.get('place')!=dict(kind='host',rank=0)):
            raise ValueError('Native descriptor ID/context/NTT/place differs')
        if row['id']in wanted:
            expected=dict(probe.EXPECTED,level=wanted[row['id']])
            if any(type(row.get(k))is not type(v)or row[k]!=v for k,v in expected.items()):
                raise ValueError('Native actual per-output level/scale/components differs from Count')
            descriptions[row['id']]=row
        count+=1
    if set(descriptions)!=set(wanted):raise ValueError('Missing complete native I/O metadata')
    defined=bytearray(count)
    def define(name):
        i=number(name)
        if i>=count or defined[i]:raise ValueError('Native SSA definition is missing or duplicated')
        defined[i]=1
    for name in inputs:define(name)
    referenced=set()
    for row in array_items(path,'initialization'):
        if row.get('kind')!='encode':raise ValueError('Original encode initialization required')
        payload=row.get('payload',{})
        if payload.get('kind')=='bundle':referenced.add(payload['content'])
        elif payload.get('kind')!='inline':raise ValueError('Unsupported original plaintext encoding')
        define(row['output'])
    counts=Counter();boot_inputs=set();boot_outputs=set();ordinals=set()
    for row in array_items(path,'execution'):
        op=row.get('op')
        if row.get('kind')!='compute'or op not in probe.ARITIES or len(row.get('inputs',[]))!=probe.ARITIES[op]:
            raise ValueError('Original native instruction kind/arity differs')
        for name in row['inputs']:
            i=number(name)
            if i>=count or not defined[i]:raise ValueError('Native operand has no prior SSA producer')
        define(row['output']);counts[op]+=1;ordinal=row.get('ordinal')
        if type(ordinal)is not int or ordinal in ordinals:raise ValueError('Native execution ordinal duplicated')
        ordinals.add(ordinal)
        if op=='boot':
            expected=dict(target_level=5,target_scale_log2=43,target_components=2,
                operator_profile='cpu-bootstrap48-level1-to5',implementation='native')
            if any(type(row.get('attrs',{}).get(k))is not type(v)or row['attrs'][k]!=v for k,v in expected.items()):
                raise ValueError('Original native bootstrap implementation/profile/target differs')
            boot_inputs.update(row['inputs']);boot_outputs.add(row['output'])
    if (counts['boot']!=expected_bootstraps or any(outputs[i]not in boot_outputs for i in target['boundary_positions'])
            or any(not x for x in defined)or list(array_items(path,'finalization'))):
        raise ValueError('Complete native bootstrap inventory, boundary outputs or SSA closure differs')
    for row in array_items(path,'values'):
        if row['id']in boot_inputs|boot_outputs:
            level=1 if row['id']in boot_inputs else 5
            if any(row.get(k)!=v for k,v in dict(kind='ciphertext',level=level,scale_log2=43,components=2).items()):
                raise ValueError('Actual native bootstrap input1/output5 raw43 metadata differs')
    bundle_record=None
    if referenced:
        reference=probe._field(path,'plaintext_bundle');bundle=path.with_name(path.name.removesuffix('.runtime-plan.json')+'.bundle')
        manifest=original.read(bundle/'manifest.json');blobs={}
        if (manifest.get('bundle_format_version')!=1 or manifest.get('version')!=1
                or reference.get('version')!=1 or manifest.get('bundle_id')!=reference.get('id')):
            raise ValueError('Original native plaintext bundle metadata differs')
        for row in manifest['blobs']:
            content=row['content'];match=re.fullmatch(r'sha256:([a-f0-9]{64})',content)
            if not match or content in blobs or type(row.get('byte_length'))is not int or row['byte_length']<1:
                raise ValueError('Invalid opaque bundle locator or file size metadata')
            blob=bundle/'data'/(match.group(1)+'.bin')
            if blob.is_symlink()or not blob.is_file()or blob.stat().st_size!=row['byte_length']:
                raise ValueError('Original bundle file closure differs')
            blobs[content]=row['byte_length']
        if not referenced<=set(blobs):raise ValueError('Native plan references missing original bundle operands')
        bundle_record=dict(path=str(bundle),manifest_blobs=len(blobs),referenced_blobs=len(referenced),payload_reads=0,integrity_checks=0)
    return dict(kernel_native_accepted=True,native_ABI_verified=True,kernel=target,physical_inputs=len(inputs),
        physical_outputs=len(outputs),input_levels=[5]*len(inputs),output_levels=levels,nominal_scale_log2=43,
        context='qwen-bootstrap48',components=2,ntt=True,complete_native_SSA_verified=True,
        compiled_bootstraps=counts['boot'],compiled_operation_count=sum(counts.values()),compiled_op_counts=dict(counts),
        bundle_closure=bundle_record,linked_native_program_verified=False,whole_generation_verified=False)


def worker(args):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    args.work_dir.mkdir(parents=True,exist_ok=False);args.codegen.mkdir(exist_ok=True)
    recorder=StageRecorder(args.work_dir/'events.jsonl');started=time.perf_counter_ns()
    result=dict(round=ROUND,model=MODEL,mode=COMPONENT_MODE,status='running',error=None,phase=args.worker_phase,
        kernel=spec(args.kind),kernel_native_accepted=False,**dict.fromkeys(original.ZERO,0),**dict.fromkeys(original.FORMAL))
    durable_json(args.work_dir/'worker-result.json',result)
    try:
        result['source']=component_source(admission(args.source_case),args.kind)
        for name in ('metadata_template','frontend','profile'):setattr(args,name,Path(result['source']['frozen_source'][name]))
        with kernels._metadata(original.read(args.metadata_template)):
            if args.worker_phase in ('placement','emit'):
                import torch
                torch.set_num_threads(1);torch.set_num_interop_threads(1)
                (placement if args.worker_phase=='placement'else emit)(args,result,recorder)
            else:
                from .. import toolchain
                from .qwen_layer_native_probe import native_command
                host=toolchain.host_paths();command=native_command(args,host);durable_json(args.work_dir/'native-argv.json',command)
                environment=dict(os.environ,HECATE=str(host['hecate']),HECATE_BUILD=str(host['hecate_build']),
                    LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'])
                observed=run_process(command,cwd=args.codegen/'emit',log=args.work_dir/'native.log',
                    resources=args.work_dir/'native-resources.jsonl',timeout=args.phase_seconds,mem_gib=24,fsize_gib=64,env=environment)
                result['process']=observed;clean(observed)
                if args.worker_phase=='trace':
                    result['trace_marker']=mark_trace(args.codegen/'emit/candidate_trace.mlir',
                        args.codegen/'emit/candidate_trace.segment.earth.mlir',args.kind);result['trace_completed']=True
                else:
                    plans=list((args.codegen/'emit').glob('lowered.*.runtime-plan.json'))
                    if len(plans)!=1:raise ValueError('One complete native kernel RuntimePlan is required')
                    planned=original.read(args.codegen/'planned-kernel.json');interface=original.read(args.codegen/'kernel-interface.json')
                    expected=planned['refresh_plan']['boots']+spec(args.kind)['boundary_bootstraps']
                    result.update(inspect_plan(plans[0],expected,interface,args.kind),compile_completed=True,runtime_plan=str(plans[0]))
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        result['kernel_native_accepted']=(result['status']=='ok'and args.worker_phase=='compile'
            and result.get('compile_completed')is True and result.get('native_ABI_verified')is True)
        result.update(linked_native_program_verified=False,whole_generation_verified=False)
        result['elapsed_ns']=time.perf_counter_ns()-started;durable_json(args.work_dir/'worker-result.json',result)
    return result


def phase_command(args,phase,work,remaining=None):
    seconds=PHASE_SECONDS[phase]if remaining is None else min(PHASE_SECONDS[phase],max(1,int(remaining)-120))
    return [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(seconds+60),'--python',str(args.intake_python),
        '--','-B','-m',MODULE,'--worker-phase',phase,'--source-case',str(args.source_case),'--kind',args.kind,
        '--work-dir',str(work),'--codegen',str(args.out/'codegen'),'--memory-gib','24','--case-gib',str(args.case_gib),
        '--seconds',str(args.seconds),'--phase-seconds',str(seconds)]+(['--diagnostic-elide-final-ir']if args.diagnostic_elide_final_ir else [])


def component_execute(args):
    namespace=vars(original).copy();namespace.update(ROUND=ROUND,MODEL=MODEL,MODE=COMPONENT_MODE,MODULE=MODULE,
        PHASE_SECONDS=PHASE_SECONDS,resource_limits=limits,frozen_source=lambda case:component_source(admission(case),args.kind),
        phase_command=phase_command,budget=budget,__name__=__name__)
    def clone(fn):
        f=FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__);f.__kwdefaults__=fn.__kwdefaults__;return f
    terminal=clone(original._terminal);deadline=time.monotonic()+args.seconds;saved_write=namespace['write_json']
    def save(path,value):
        if Path(path).name=='result.json'and isinstance(value,dict)and value.get('mode')==COMPONENT_MODE:
            value.update(kernel=spec(args.kind),linked_native_program_verified=False,whole_generation_verified=False)
            value['kernel_native_accepted']=(value.get('status')=='ok'and value.get('completed',{}).get('compile')==1
                and (value.get('phase_results',{}).get('compile')or{}).get('kernel_native_accepted')is True)
            if isinstance(value.get('metrics'),dict):
                value['metrics'].update(kernel_native_accepted=value['kernel_native_accepted'],
                    linked_native_program_verified=False,whole_generation_verified=False)
        return saved_write(path,value)
    def finish(a,out,result,started,revision,recorder):
        try:
            budget(a,out)
            if time.monotonic()>=deadline:raise TimeoutError('Kernel component complete allowance exhausted')
            observed=physical_observation(None,_available_ram_bytes(),shutil.disk_usage(out).free)
            if observed['stop']:raise RuntimeError(str(observed))
        except BaseException as exc:result.update(status='error',error='Terminal kernel policy: '+str(exc))
        shutil.copyfile(kernels.__file__,out/'source'/Path(kernels.__file__).name)
        terminal(a,out,result,started,revision,recorder)
    def observed_process(command,**kwargs):
        observed=run_process(command,**kwargs);phase=command[command.index('--worker-phase')+1]
        durable_json(args.out/(phase+'-supervision-observation.json'),observed);clean(observed);return observed
    namespace.update(_terminal=finish,write_json=save,run_process=observed_process)
    source_gate=namespace['frozen_source']
    def finite(case):
        signal.setitimer(signal.ITIMER_REAL,min(600.,max(.001,deadline-time.monotonic())))
        return source_gate(case)
    namespace['frozen_source']=finite
    return clone(original.execute)(args)


def accept_component(case,kind,checked):
    record=original.read(case/'result.json');complete=original.read(case/'completion.json');source=component_source(checked,kind)
    if (record.get('status')!='ok'or record.get('round')!=ROUND or record.get('mode')!=COMPONENT_MODE
            or record.get('model')!=MODEL or record.get('source')!=source or record.get('kernel')!=spec(kind)
            or record.get('kernel_native_accepted')is not True or record.get('termination')is not None
            or record.get('terminal_budget_error')is not None or record.get('failed_phase')is not None
            or any(record.get(k)!=0 for k in original.ZERO)or any(record.get(k)is not None for k in original.FORMAL)
            or complete.get('status')!='ok'or complete.get('registry_persisted')is not True
            or complete.get('required_artifacts_persisted')is not True):raise ValueError('Complete same-C2 durable native kernel required')
    for phase in PHASE_SECONDS:
        if any(type(record.get(key,{}).get(phase))is not int or record[key][phase]!=1
               for key in ('attempted','completed')):raise ValueError('Each kernel phase must complete once')
        rows=[r for r in record.get('processes',[])if r.get('phase')==phase]
        if len(rows)!=1:raise ValueError('One actual observation per kernel phase required')
        clean(rows[0]['observation']);child=original.read(case/'phases'/phase/'worker-result.json')
        if (child!=record.get('phase_results',{}).get(phase)or child.get('source')!=source
                or child.get('status')!='ok'or child.get(phase+'_completed')is not True
                or child.get('round')!=ROUND or child.get('model')!=MODEL or child.get('mode')!=COMPONENT_MODE
                or child.get('kernel')!=spec(kind)or any(child.get(k)!=0 for k in original.ZERO)
                or any(child.get(k)is not None for k in original.FORMAL)):raise ValueError('Kernel worker lineage differs')
    planned=original.read(case/'codegen/planned-kernel.json');returned=original.read(case/'codegen/planner-return.json')
    counted=original.read(case/'codegen/count-replay.json');interface=original.read(case/'codegen/kernel-interface.json')
    plan=planned['refresh_plan'];replay=planned['replay'];kernel_source=planned['source']
    from .qwen_monotone_selector import config
    geometry=kernels._geometry(kernels.SLOTS,kernels.VOCABULARY,kernels.HIDDEN,kernels.CAPACITY)
    roles=['one_hot:'+str(i)for i in range(19)]+(['mixed_embedding','attention_mask','sel']if kind=='first'else [])
    weights=returned.get('weights')or[]
    if (kernel_source.get('binding')!=checked['binding']or kernel_source.get('kind')!=kind
            or kernel_source.get('geometry')!=geometry or kernel_source.get('selector')!=original.json_value(config())
            or kernel_source.get('output_roles')!=roles
            or kernel_source.get('embedding_weight')!=(kernels.WEIGHT if kind=='first'else None)
            or kernel_source.get('original_token0_input')!=(19 if kind=='first'else None)
            or kernel_source.get('future_boundary_bootstraps')!=spec(kind)['future_boundary_bootstraps']
            or plan.get('format')!='pt-agent-refresh-plan-v1'or plan.get('levels')!=kernels.LEVELS
            or type(plan.get('boots'))is not int or plan['boots']<0
            or replay.get('calls')!=plan.get('calls')or replay.get('boots')!=plan['boots']or replay.get('inner_boots')!=plan['boots']
            or returned.get('verification')!=replay or len(weights)!=559
            or len({row[0]for row in weights})!=559 or any(row[2:]!=['float64',False]for row in weights)
            or [row[1]for row in weights if row[0]==kernels.WEIGHT]!=[[151936,896]]
            or planned['source']!=counted['interface']['source']or returned.get('source')!=planned['source']
            or returned.get('planner_returned')is not True or returned.get('verify_completed')is not True
            or interface!=counted['interface']or any(record['phase_results']['emit']['emit_stats'][k]!=counted['stats'][k]
                for k in ('counts','ops_total','distinct_consts'))):raise ValueError('Actual original plan/verify and Count/Emit binding differs')
    interface_record(interface)
    expected=planned['refresh_plan']['boots']+spec(kind)['boundary_bootstraps']
    if counted['stats']['counts']['boot']!=expected or record.get('compiled_bootstraps')!=expected:
        raise ValueError('Complete kernel boundary bootstrap count differs')
    path=Path(record['runtime_plan']).resolve()
    if path.parent!=(case/'codegen/emit').resolve()or not path.name.startswith('lowered.'):raise ValueError('Kernel native artifact path differs')
    native=inspect_plan(path,expected,interface,kind);child=record['phase_results']['compile']
    if any(child.get(k)!=v for k,v in native.items()):raise ValueError('Actual saved native metadata replay differs')
    return dict(status='ok',kernel=spec(kind),case=str(case),runtime_plan=str(path),native=native,
        source=source,interface=interface,phase_results=record['phase_results'],timing=record.get('timing'))


def close(args,result,started,recorder,deadline,failures):
    try:
        if failures:raise RuntimeError(failures[-1])
        if time.monotonic()>=deadline:raise TimeoutError('Kernel total21600-second allowance exhausted')
        budget(args,args.out);observed=physical_observation(None,_available_ram_bytes(),shutil.disk_usage(args.out).free)
        if observed['stop']:raise RuntimeError(str(observed))
    except BaseException as exc:result.update(status='error',error='Terminal kernel batch policy: '+str(exc))
    result['timing']=summarize_events(read_events(recorder.path));result['artifact_bytes']=0
    for _ in range(20):
        result['two_kernel_native_components_verified']=(result['status']=='ok'and result.get('accepted_components')==2
            and all(row['status']=='ok'for row in result['components'].values()))
        result.update(linked_native_program_verified=False,whole_generation_verified=False,full_prefill_verified=False)
        durable_json(args.out/'result.json',result);size=artifact_bytes(args.out)
        if size+RESERVE>args.case_gib*GIB:
            result.update(status='error',error='Kernel completion reserve exhausted',two_kernel_native_components_verified=False)
            durable_json(args.out/'result.json',result);size=artifact_bytes(args.out)
        if size==result['artifact_bytes']:break
        result['artifact_bytes']=size
    else:raise RuntimeError('Kernel batch terminal inventory did not stabilize')
    row=dict(run_id=args.out.name,round=ROUND,model=MODEL,mode=MODE,git_rev=result.get('git_rev'),
        metrics={k:result[k]for k in ('status','accepted_components','two_kernel_native_components_verified',
            'linked_native_program_verified','whole_generation_verified','full_prefill_verified','compiled_bootstraps')},
        artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat())
    with (original.paths.RESULTS_DIR/'registry.jsonl').open('a+')as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id')==args.out.name for line in stream if line.strip()):raise ValueError('Preserve existing kernel registry identity')
        stream.seek(0,os.SEEK_END);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    persist_case_completion(args.out,round_label=ROUND,started_ns=started,status=result['status'],timing=result['timing'],required_artifact_bytes=artifact_bytes(args.out))


def execute(args):
    resource=limits(args);started=time.perf_counter_ns();deadline=time.monotonic()+args.seconds
    out=args.out.resolve();root=original.paths.RESULTS_DIR.resolve();source_case=args.source_case.resolve()
    if (out.exists()or out.is_symlink()or out==root or not out.is_relative_to(root)
            or out.is_relative_to(source_case)or source_case.is_relative_to(out)):raise ValueError('Use one new isolated kernel batch case')
    with serial_lock(root):
        out.mkdir();recorder=StageRecorder(out/'events.jsonl');active=[None];stop=threading.Event();failures=[]
        from ..registry import git_rev
        result=dict(round=ROUND,model=MODEL,mode=MODE,status='running',error=None,git_rev=git_rev(),limits=resource,
            components={k:dict(status='not_run',kernel=spec(k))for k in KINDS},accepted_components=0,compiled_bootstraps=None,
            two_kernel_native_components_verified=False,linked_native_program_verified=False,whole_generation_verified=False,
            full_prefill_verified=False,processes=[],failed_component=None,**dict.fromkeys(original.ZERO,0),**dict.fromkeys(original.FORMAL))
        old={sig:signal.getsignal(sig)for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1,signal.SIGALRM)}
        def incoming(sig,frame):
            result['termination']=dict(signal=sig,cause=failures[-1]if failures else'incoming_signal')
            raise InterruptedError('Kernel compile batch interrupted')
        for sig in old:signal.signal(sig,incoming)
        def check():
            if failures:raise RuntimeError(failures[-1])
            if time.monotonic()>=deadline:raise TimeoutError('Complete kernel batch allowance exhausted')
            budget(args,out);observed=physical_observation(_latest_rss_sample(active[0])if active[0]else None,
                _available_ram_bytes(),shutil.disk_usage(out).free)
            if observed['stop']:raise RuntimeError(str(observed))
        def monitor():
            while not stop.wait(.25):
                try:check()
                except BaseException as exc:
                    failures.append(type(exc).__name__+': '+str(exc))
                    try:durable_json(out/'guard-failure.json',dict(reason=failures[-1]))
                    finally:os.kill(os.getpid(),signal.SIGUSR1)
                    return
        watcher=threading.Thread(target=monitor,daemon=True)
        try:
            signal.setitimer(signal.ITIMER_REAL,min(600.,args.seconds))
            with recorder.span('same-C2-saved-state-admission'):checked=admission(args.source_case)
            signal.setitimer(signal.ITIMER_REAL,0);result['source']=checked
            snapshots=out/'source';snapshots.mkdir()
            for module in (sys.modules[__name__],original,kernels):shutil.copyfile(module.__file__,snapshots/Path(module.__file__).name)
            durable_json(out/'source-record.json',checked);watcher.start();check()
            for kind in KINDS:
                check();remaining=deadline-time.monotonic()
                if remaining<=180:raise TimeoutError('Insufficient kernel supervision time remains')
                seconds=min(21600,int(remaining)-120);case=out/kind
                command=[sys.executable,'-B','-m',MODULE,'--source-case',str(args.source_case),'--out',str(case),
                    '--batch-out',str(out),'--kind',kind,'--intake-python',str(args.intake_python),'--memory-gib','24',
                    '--case-gib',str(args.case_gib),'--seconds',str(seconds)]+(['--diagnostic-elide-final-ir']if args.diagnostic_elide_final_ir else [])
                result['failed_component']=kind;result['components'][kind]['status']='running';durable_json(out/'result.json',result)
                active[0]=out/(kind+'-resources.jsonl')
                with recorder.span('component-'+kind,argv=command):
                    observed=run_process(command,cwd=original.paths.REPO/'DSL/DSL_agent/scripts',log=out/(kind+'.log'),
                        resources=active[0],timeout=min(seconds+60,max(.01,deadline-time.monotonic())),mem_gib=24,fsize_gib=64,
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'))
                active[0]=None;result['processes'].append(dict(kind=kind,argv=command,observation=observed));clean(observed)
                result['components'][kind]=accept_component(case,kind,checked);result['accepted_components']+=1;check()
            result.update(status='ok',failed_component=None,compiled_bootstraps=sum(r['native']['compiled_bootstraps']for r in result['components'].values()))
        except BaseException as exc:
            result.update(status='error',error=type(exc).__name__+': '+str(exc))
            if result['failed_component']:result['components'][result['failed_component']].update(status='error',error=result['error'])
        finally:
            active[0]=None;stop.set()
            if watcher.is_alive():watcher.join(timeout=3)
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig in old:signal.signal(sig,signal.SIG_IGN)
            try:close(args,result,started,recorder,deadline,failures)
            finally:
                for sig,handler in old.items():signal.signal(sig,handler)
        return result


def parser():
    cli=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    cli.add_argument('--source-case',type=Path,required=True);cli.add_argument('--out',type=Path)
    cli.add_argument('--kind',choices=KINDS);cli.add_argument('--worker-phase',choices=tuple(PHASE_SECONDS))
    for key in ('work-dir','codegen','batch-out'):cli.add_argument('--'+key,type=Path)
    cli.add_argument('--through',choices=('compile',),default='compile')
    cli.add_argument('--intake-python',type=Path,default=original.paths.INTAKE_PYTHON)
    cli.add_argument('--memory-gib',type=float,default=24.);cli.add_argument('--case-gib',type=int,default=128)
    cli.add_argument('--seconds',type=int,default=21600);cli.add_argument('--phase-seconds',type=int)
    cli.add_argument('--diagnostic-elide-final-ir',action='store_true');return cli


def main(argv=None):
    cli=parser();args=cli.parse_args(argv);limits(args)
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().absolute()if key=='intake_python'else value.expanduser().resolve())
    if args.worker_phase:
        if not args.kind or args.work_dir is None or args.codegen is None:cli.error('Explicit kernel/work/codegen required')
        if type(args.phase_seconds)is not int or not 1<=args.phase_seconds<=PHASE_SECONDS[args.worker_phase]:cli.error('Use finite phase allowance')
        result=worker(args)
    else:
        if args.out is None:cli.error('Use a fresh explicit output case')
        if args.batch_out is not None and (not args.kind or args.out.parent!=args.batch_out or args.out.name!=args.kind):
            cli.error('Only the matching fresh component may use its original batch directory')
        result=component_execute(args)if args.kind else execute(args)
    print(json.dumps({k:result.get(k)for k in ('status','error','kernel_native_accepted','two_kernel_native_components_verified')}))
    return int(result['status']!='ok')


if __name__=='__main__':raise SystemExit(main())
