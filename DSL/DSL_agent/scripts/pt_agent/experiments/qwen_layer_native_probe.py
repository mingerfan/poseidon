"""Finite post-C2 layer0 native probe; complete-model deployment stays pending.

Only own orchestration and existing interfaces are adapted. The original
program, weights, planner candidates, qwen48 profile and native boot contract
remain. Raw native trace is retained; a new copy adds existing segment return
attributes. No API, model forward, data read or CKKS client lives here.
"""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
from types import FunctionType

from . import qwen_prefill_compile as original
from . import qwen_layered_structure as layered
from .native_component_matrix import array_items
from .r10_paper_metrics import StageRecorder, run_process, write_json

ROUND='QWEN-FROZEN-LAYER0-NATIVE-PROBE'
MODEL=original.MODEL
MODE='full24-source-layer0-only-native-abi-probe-qwen48'
MODULE='pt_agent.experiments.qwen_layer_native_probe'
PHASE_SECONDS=dict(placement=3600,emit=1800,trace=1800,compile=7200)
GIB=1024**3
EXPECTED=dict(kind='ciphertext',level=5,scale_log2=43,components=2,context='qwen-bootstrap48',ntt=True)
ARITIES=dict(mod_switch=1,rotate=1,mul_cp=2,rescale=1,add_cc=2,add_cp=2,boot=1,
    mul_cc=2,relinearize=1,negate=1)


def admission(case):
    source=original.frozen_source(case)
    return dict(source,probe_layer_index=0,probe_scope='One original decoder layer only; remaining23 layers and head not compiled',
        full_model_compile_verified=False,parameter_context_changed=False)


def resource_limits(args):
    result=original.resource_limits(args)
    result.update(phases=PHASE_SECONDS.copy(),probe_layer_index=0,full_model_compile_verified=False,
        global_deadline_applies_to_source=True)
    return result


def logical_mask_backend(base):
    """Planner-only34-output ABI; original arithmetic and computed checks stay."""
    class LogicalMaskOutputs(base):
        def output(self,values):
            values=list(values)
            if (len(values)!=34 or len(self._inputs)!=34
                    or any(values[i] is not self._inputs[i] for i in (32,33))):
                raise ValueError('Planner logical outputs require32 computed values and exact original mask aliases')
            super().output(values[:32])
            self._outputs.extend(values[32:])
    return LogicalMaskOutputs


def planner_namespace():
    from ..backends.cost_dag import CostDag
    from ..backends.streaming_count import StreamingCountBackend
    return original._planner_namespace(dag_class=logical_mask_backend(CostDag),
        count_class=logical_mask_backend(StreamingCountBackend))


def _load(args,result):
    info,program=original._load_program(args,result)
    partition=layered.split_program(program,{name:array.shape for name,array in info.graph.weights.items()})
    callback=layered.callbacks(partition,info.graph.weights)[0][0]
    from .qwen_component_cost import already_scoped_lowering
    callback=already_scoped_lowering(callback)
    return partition,callback


def _count(callback,bound):
    from ..backends.streaming_count import StreamingCountBackend
    backend=StreamingCountBackend(layered.SLOTS,input_level=5,boot_out_level=5)
    state=layered.export_layer(backend,callback=callback,bound_plan=bound)
    return dict(stats=backend.stats(),interface=state.metadata)


def placement(args,result,recorder):
    from ..kernels.compact_linear import compact_transformer_projections
    from . import qwen_component_plan_cache as cache
    partition,callback=_load(args,result)
    source=layered.source_binding(callback.segment)
    write_json(args.codegen/'source-bindings.json',dict(layers=[layered.source_binding(s) for s in partition.layers],
        head=layered.source_binding(partition.head),planned_layers=[0],remaining_plans='not_run'))
    cached=cache.worker_plan(args,result['source'],callback.segment,recorder)
    if cached is None:
        planner=planner_namespace()
        with recorder.span('original_layer0_program_plan'),original._projection_backend_scope(),compact_transformer_projections():
            plan,dag,placed=planner['plan_program'](callback.segment.program,callback.weights,
                profile='qwen48',frontend_normalize=True)
            # Bind at this actual trusted planner return, before any independent replay.
            bound=layered.bind_plan(callback.segment,plan,planned_source=source)
            del dag,placed;gc.collect()
            verified=planner['verify_program'](callback.segment.program,callback.weights,plan,profile='qwen48')
    else:
        plan=cached['plan'];verified=cached['verification']
        bound=layered.bind_plan(callback.segment,plan,planned_source=source)
    with recorder.span('independent_layer0_count_with32_boundary_boots'):
        counted=_count(callback,bound)
    if cached is not None:cache.compare_count(cached['count'],counted)
    if (verified['boots']!=plan['boots'] or counted['stats']['counts']['boot']!=plan['boots']+32
            or counted['interface']['internal_plan_bootstrap_calls']!=plan['boots']):
        raise ValueError('Original layer0 plan, verify and complete boundary Count differ')
    write_json(args.codegen/'refresh-plan.json',plan)
    write_json(args.codegen/'planner-return.json',dict(source=source,planner='original.pt_agent.plan.plan_program',
        planner_returned=True,verify_completed=True,verification=verified,source_bindings='source-bindings.json',
        **(dict(plan_cache=cached['provenance']) if cached is not None else {})))
    write_json(args.codegen/'count-replay.json',counted)
    result.update(placement_completed=True,planned_bootstraps=plan['boots'],boundary_bootstraps=32,
        internal_planner_latency_us=plan.get('latency_us'),
        planner_cost_scope='Original internal arithmetic; fixed32 boundary boots separately counted in complete Count/Emit',
        expected_compiled_bootstraps=plan['boots']+32,plan_verification=verified,count_stats=counted['stats'])
    if cached is not None:result['plan_cache']=cached['provenance']


def emit(args,result,recorder):
    from ..backends.streaming_emit import StreamingEmitBackend
    partition,callback=_load(args,result)
    returned=original.read(args.codegen/'planner-return.json')
    if (returned.get('planner_returned')is not True or returned.get('verify_completed')is not True
            or returned.get('planner')!='original.pt_agent.plan.plan_program'):
        raise ValueError('Require the actual original layer0 planner return and independent verify')
    bound=layered.bind_plan(callback.segment,original.read(args.codegen/'refresh-plan.json'),
        planned_source=returned['source'])
    counted=original.read(args.codegen/'count-replay.json')['stats']
    backend=StreamingEmitBackend(layered.SLOTS,directory=args.codegen/'emit',frontend=args.frontend,
        max_constant_bytes=int(args.case_gib*GIB-GIB))
    try:
        with recorder.span('same_layer0_trusted_emit_with32_boundary_boots'):
            state=layered.export_layer(backend,callback=callback,bound_plan=bound);stats=backend.stats()
            if any(stats[key]!=counted[key]for key in ('counts','ops_total','distinct_consts')):
                raise ValueError('Complete original layer0 Count/Emit boundary schedules differ')
            source=backend.write()
            write_json(args.codegen/'layer-interface.json',state.metadata)
            result.update(emit_completed=True,trace_source=str(source),emit_stats=stats,
                physical_inputs=34,physical_outputs=32,selector_input_retained=True)
    finally:
        if not backend.finished:backend.close()


def _balanced(text,start):
    pairs={'(':')','[':']','<':'>'};stack=[];quoted=False;escaped=False
    for index in range(start,len(text)):
        char=text[index]
        if quoted:
            if escaped:escaped=False
            elif char=='\\':escaped=True
            elif char=='"':quoted=False
            continue
        if char=='"':quoted=True
        elif char in pairs:stack.append(pairs[char])
        elif char in ')]>':
            if not stack or stack.pop()!=char:raise ValueError('Malformed trace function header delimiters')
            if not stack:return index
    raise ValueError('Trace function header exceeds bounded complete delimiter scan')


def _cells(text):
    result=[];start=0;index=0
    while index<len(text):
        if text[index]in '([<':index=_balanced(text,index)
        elif text[index]=='"':
            raise ValueError('Unexpected quoted cell in the strict trace ABI header')
        elif text[index]==',':result.append(text[start:index].strip());start=index+1
        index+=1
    result.append(text[start:].strip())
    return result


def _generic_header(prefix):
    match=re.search(r'(?m)^[ \t]*"func\.func"\(\)\s*<\{',prefix)
    if not match:raise ValueError('Require the original generic-form golden function header')
    position=match.end()
    field=re.match(r'\s*function_type\s*=\s*',prefix[position:])
    if not field:raise ValueError('Retain original generic function_type property')
    position+=field.end()
    if prefix[position:position+1]!='(':raise ValueError('Generic function_type requires34 input types')
    end=_balanced(prefix,position);args=_cells(prefix[position+1:end]);position=end+1
    arrow=re.match(r'\s*->\s*',prefix[position:])
    if not arrow:raise ValueError('Generic function_type requires original output tuple')
    position+=arrow.end()
    if prefix[position:position+1]!='(':raise ValueError('Generic function_type requires32 explicit output types')
    end=_balanced(prefix,position);outputs=_cells(prefix[position+1:end]);position=end+1
    suffix=re.match(r',\s*sym_name\s*=\s*"_hecate_golden"\s*\}\s*>\s*\(\s*\{',prefix[position:])
    if not suffix:raise ValueError('Conflicting properties or unexpected original generic function symbol')
    tensor=re.compile(r'tensor<1x!earth\.ci<\s*0\s*\*\s*0\s*>>$')
    if len(args)!=34 or len(outputs)!=32 or any(not tensor.fullmatch(x)for x in args+outputs):
        raise ValueError('Generic trace must retain34 original input types and32 cipher result types')
    return position+suffix.end()-1


def _function_end(path,region_open):
    # A fixed-size lexical pass skips quoted text and line comments, rejects
    # nested regions, and identifies the function region boundary. Dense numeric
    # operands are never decoded or retained as a full line/string.
    token=re.compile(r'\\.|"|//|\n|[{}()]')
    declarations=re.compile(r'(?m)^[ \t]*(?P<operation>"func\.func"\(\))')
    quoted=comment=False;depth=1;paren=False;closing=None;after=''
    seen=[];decl_tail='';carry='';offset=0
    with Path(path).open()as stream:
        while True:
            chunk=stream.read(65536);eof=not chunk
            if chunk:
                combined=decl_tail+chunk
                for found in declarations.finditer(combined):
                    absolute=offset-len(decl_tail)+found.start('operation')
                    if not seen or absolute>seen[-1]:seen.append(absolute)
                decl_tail=combined[-256:]
            buffer=carry+chunk;begin=offset-len(carry);limit=len(buffer)if eof else max(0,len(buffer)-1)
            if closing is None:
                consumed=limit
                for found in token.finditer(buffer):
                    if found.end()>limit:consumed=found.start();break
                    absolute=begin+found.start()
                    if absolute<=region_open:continue
                    word=found.group()
                    if comment:
                        if word=='\n':comment=False
                        continue
                    if word.startswith('\\'):continue
                    if word=='"':quoted=not quoted;continue
                    if quoted:continue
                    if word=='//':comment=True;continue
                    if word=='{':
                        if paren:raise ValueError('Nested native regions are outside the single-layer trace contract')
                        depth+=1
                    elif word=='}':
                        depth-=1
                        if depth==0:
                            closing=absolute;after=buffer[found.start():];break
                    if word in ('(',')','{','}'):
                        paren=word=='('
                carry=buffer[consumed:]
            elif len(after)<4096:after+=chunk[:4096-len(after)]
            offset+=len(chunk)
            if eof:break
    if closing is None or len(seen)!=1:raise ValueError('Require exactly one complete original generic function')
    suffix=re.match(r'}\s*\)\s*:',after)
    if not suffix:raise ValueError('Existing function attributes or malformed generic function closure')
    close=re.match(r'}\s*\)',after)
    return closing+close.end()


def mark_segment_trace(source,destination):
    """Add only discardable attributes to an own generic trace copy."""
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve()or destination.exists()or destination.is_symlink():
        raise ValueError('Retain raw trace and use one fresh marked trace copy')
    with source.open()as stream:region_open=_generic_header(stream.read(2*1024**2))
    insertion=_function_end(source,region_open)
    attrs=' {is_mid_segment = true, segment_returnBypasses = ['+', '.join(['true']*32)+']}'
    temporary=destination.with_name(destination.name+'.tmp')
    if temporary.exists()or temporary.is_symlink():raise ValueError('Preserve an existing trace marking attempt')
    offset=0;inserted=False
    with source.open()as inp,temporary.open('x')as out:
        while True:
            chunk=inp.read(65536)
            if not chunk:break
            if not inserted and offset<=insertion<=offset+len(chunk):
                at=insertion-offset;out.write(chunk[:at]+attrs+chunk[at:]);inserted=True
            else:out.write(chunk)
            offset+=len(chunk)
    if not inserted:raise ValueError('The complete original function attribute position is missing')
    temporary.replace(destination)
    return dict(raw_trace=str(source),marked_trace=str(destination),physical_inputs=34,physical_outputs=32,
        is_mid_segment=True,return_bypasses=[True]*32,segment_inputType_added=False,
        source_form='original-generic-form',attribute_position='unique-function-discardable-attribute-dictionary',
        Earth_consumed_input_level=0,expected_native_remaining_input_level=5,
        arithmetic_rewritten=False,raw_trace_retained=True,native_ABI_verified=False)


def native_command(args,host):
    directory=args.codegen/'emit'
    if args.worker_phase=='trace':return [sys.executable,'-B',str(directory/'candidate_trace.py')]
    from .r29_local_compile_matrix import compiler_argv
    from .r30_common_model_e2e import diagnostic_compile_argv
    template=original.read(args.metadata_template)
    if (any(arg.startswith(('--runtime-plan-physical-level-','--runtime-plan-levels-per-logical-level'))for arg in template)
            or '--runtime-plan-ntt=true'not in template or '--runtime-plan-boot-implementation=native'not in template):
        raise ValueError('Retain native qwen48 levels/NTT/bootstrap export without physical remapping')
    argv=compiler_argv(template,'pars',binary=host['hecate_opt'],source=directory/'candidate_trace.segment.earth.mlir',
        profile=args.profile,output=directory/'lowered.mlir')
    return diagnostic_compile_argv(argv,elide_final_ir=args.diagnostic_elide_final_ir)


def _field(path,key):
    decoder=json.JSONDecoder();pattern=re.compile(r'"'+re.escape(key)+r'"\s*:\s*')
    with Path(path).open()as stream:
        buffer='';found=None
        while True:
            chunk=stream.read(65536)
            if not chunk:raise ValueError('Missing or truncated native field: '+key)
            buffer+=chunk
            if found is None:
                found=pattern.search(buffer)
                if found is None:buffer=buffer[-128:];continue
                buffer=buffer[found.end():];found=True
            try:return decoder.raw_decode(buffer.lstrip())[0]
            except json.JSONDecodeError:
                if len(buffer)>2*1024**2:raise ValueError('Native metadata field exceeds bounded small-object allowance')


def inspect_runtime_plan(path,expected_bootstraps):
    """Stream complete metadata and file stat; never read a blob or compute a digest."""
    path=Path(path)
    if type(expected_bootstraps)is not int or expected_bootstraps<32 or _field(path,'format_version')!=1:
        raise ValueError('Require original format1 layer plan and its actual planned boot inventory')
    inputs=list(array_items(path,'external_inputs'));outputs=list(array_items(path,'final_outputs'))
    if len(inputs)!=34 or len(set(inputs))!=34 or len(outputs)!=32 or len(set(outputs))!=32:
        raise ValueError('Native layer ABI requires34 independent inputs and32 independent outputs')
    def number(name):
        if not isinstance(name,str)or not name.isdecimal()or str(int(name))!=name:raise ValueError('Require original numeric native SSA IDs')
        return int(name)
    selected=set(inputs+outputs);descriptions={};count=0
    for item in array_items(path,'values'):
        if number(item['id'])!=count:raise ValueError('Native descriptor IDs must be complete consecutive original IDs')
        if item.get('context')!='qwen-bootstrap48'or item.get('ntt')is not True:
            raise ValueError('Native value context/NTT differs')
        if item['id']in selected:
            if any(type(item.get(k))is not type(v)or item[k]!=v for k,v in EXPECTED.items()):
                raise ValueError('Actual native I/O level/scale/context/NTT/components differs')
            descriptions[item['id']]=item
        count+=1
    if set(descriptions)!=selected:raise ValueError('Native I/O metadata is missing')
    defined=bytearray(count)
    def define(name):
        i=number(name)
        if i>=count or defined[i]:raise ValueError('Native SSA output is absent or redefined')
        defined[i]=1
    for name in inputs:define(name)
    referenced=set()
    for item in array_items(path,'initialization'):
        if item.get('kind')!='encode':raise ValueError('Native initialization must be original encodes')
        payload=item.get('payload',{})
        if payload.get('kind')=='bundle':referenced.add(payload['content'])
        elif payload.get('kind')!='inline':raise ValueError('Unsupported original encode payload')
        define(item['output'])
    counts=Counter();boot_ids=set();boot_inputs=set();final_boots=set();ordinals=set()
    for item in array_items(path,'execution'):
        if (item.get('kind')!='compute'or item.get('op')not in ARITIES
                or len(item.get('inputs',[]))!=ARITIES[item['op']]):raise ValueError('Require supported native compute execution and arity')
        for name in item['inputs']:
            i=number(name)
            if i>=count or not defined[i]:raise ValueError('Native SSA operand has no prior definition')
        define(item['output']);counts[item['op']]+=1
        ordinal=item.get('ordinal')
        if type(ordinal)is not int or ordinal in ordinals:raise ValueError('Native instruction ordinal missing or duplicated')
        ordinals.add(ordinal)
        if item['op']=='boot':
            expected=dict(target_level=5,target_scale_log2=43,target_components=2,
                operator_profile='cpu-bootstrap48-level1-to5',implementation='native')
            if (len(item['inputs'])!=1 or any(type(item.get('attrs',{}).get(k))is not type(v)
                    or item['attrs'][k]!=v for k,v in expected.items())):
                raise ValueError('Actual native bootstrap profile/target metadata differs')
            boot_ids.add(item['output']);boot_inputs.update(item['inputs'])
            if item['output']in outputs:final_boots.add(item['output'])
    if (counts['boot']!=expected_bootstraps or final_boots!=set(outputs)
            or any(not defined[number(name)]for name in outputs)or any(not value for value in defined)
            or list(array_items(path,'finalization'))):
        raise ValueError('Native boot inventory, final boot outputs or complete SSA closure differs')
    wanted=boot_ids|boot_inputs
    for item in array_items(path,'values'):
        if item['id']in wanted:
            level=1 if item['id']in boot_inputs else 5
            if (item.get('kind')!='ciphertext'or item.get('level')!=level or item.get('scale_log2')!=43
                    or item.get('components')!=2):raise ValueError('Actual native bootstrap input/output type differs')
    bundle_record=None
    if referenced:
        reference=_field(path,'plaintext_bundle');bundle=path.with_name(path.name.removesuffix('.runtime-plan.json')+'.bundle')
        manifest=original.read(bundle/'manifest.json')
        if (manifest.get('bundle_format_version')!=1 or manifest.get('version')!=1
                or reference.get('version')!=1 or manifest.get('bundle_id')!=reference.get('id')):
            raise ValueError('Native bundle manifest contract differs')
        blobs={}
        for item in manifest['blobs']:
            content=item['content'];match=re.fullmatch(r'sha256:([a-f0-9]{64})',content)
            if not match or content in blobs or type(item.get('byte_length'))is not int or item['byte_length']<1:
                raise ValueError('Invalid opaque native bundle locator or size metadata')
            target=bundle/'data'/(match.group(1)+'.bin')
            if target.is_symlink()or not target.is_file()or target.stat().st_size!=item['byte_length']:
                raise ValueError('Native bundle file closure differs')
            blobs[content]=item['byte_length']
        if not referenced<=set(blobs):raise ValueError('Native initialization references missing manifest blobs')
        bundle_record=dict(path=str(bundle),manifest_blobs=len(blobs),referenced_blobs=len(referenced),
            blob_payload_reads=0,extra_integrity_checks=0)
    return dict(probe_accepted=True,native_ABI_verified=True,full_model_compile_verified=False,
        physical_inputs=34,physical_outputs=32,selector_input_retained=True,input_output_metadata=EXPECTED.copy(),
        compiled_bootstraps=counts['boot'],compiled_operation_count=sum(counts.values()),compiled_op_counts=dict(counts),
        complete_native_SSA_verified=True,bundle_closure=bundle_record,remaining23_layers_and_head='not_run')


def worker(args):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    args.work_dir.mkdir(parents=True,exist_ok=False);args.codegen.mkdir(exist_ok=True)
    recorder=StageRecorder(args.work_dir/'events.jsonl');started=time.perf_counter_ns()
    result=dict(round=ROUND,model=MODEL,mode=MODE,status='running',error=None,phase=args.worker_phase,
        probe_layer_index=0,probe_accepted=False,full_model_compile_verified=False,
        **dict.fromkeys(original.ZERO,0),**dict.fromkeys(original.FORMAL))
    write_json(args.work_dir/'worker-result.json',result)
    try:
        result['source']=admission(args.source_case)
        for name in ('metadata_template','frontend','profile'):setattr(args,name,Path(result['source'][name]))
        from .r29_local_compile_matrix import native_metadata_commands
        with native_metadata_commands(original.read(args.metadata_template)):
            if args.worker_phase in ('placement','emit'):
                import torch
                torch.set_num_threads(1);torch.set_num_interop_threads(1)
                (placement if args.worker_phase=='placement'else emit)(args,result,recorder)
            else:
                from .. import toolchain
                host=toolchain.host_paths();command=native_command(args,host)
                write_json(args.work_dir/'native-argv.json',command)
                environment=dict(os.environ,HECATE=str(host['hecate']),HECATE_BUILD=str(host['hecate_build']),
                    LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'])
                process=run_process(command,cwd=args.codegen/'emit',log=args.work_dir/'native.log',
                    resources=args.work_dir/'native-resources.jsonl',timeout=args.phase_seconds,
                    mem_gib=args.memory_gib,fsize_gib=64,env=environment)
                result['process']=process
                if process['exit_code']!=0 or process['timed_out']or process['signal']is not None or process['remaining_pids']:
                    raise RuntimeError('First native layer0 '+args.worker_phase+' attempt failed')
                if args.worker_phase=='trace':
                    result['trace_marker']=mark_segment_trace(args.codegen/'emit/candidate_trace.mlir',
                        args.codegen/'emit/candidate_trace.segment.earth.mlir')
                    result['trace_completed']=True
                else:
                    plans=list((args.codegen/'emit').glob('lowered.*.runtime-plan.json'))
                    if len(plans)!=1:raise ValueError('Require one unique original layer0 RuntimePlan')
                    expected=original.read(args.codegen/'refresh-plan.json')['boots']+32
                    result.update(inspect_runtime_plan(plans[0],expected),compile_completed=True,runtime_plan=str(plans[0]))
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        result['elapsed_ns']=time.perf_counter_ns()-started;write_json(args.work_dir/'worker-result.json',result)
    return result


def _namespace():
    """Reuse ordinary supervision functions; retain all contextmanager globals."""
    ns=vars(original).copy();ns.update(ROUND=ROUND,MODEL=MODEL,MODE=MODE,MODULE=MODULE,
        PHASE_SECONDS=PHASE_SECONDS,resource_limits=resource_limits,frozen_source=admission,__name__=__name__)
    save_json=ns['write_json']
    def persist_current_acceptance(path,value):
        # The original terminal writer may change status when its completion
        # reserve is exhausted. Derive acceptance at every result persistence,
        # including its subsequent error write, before registry/closure.
        if (Path(path).name=='result.json'and isinstance(value,dict)
                and value.get('round')==ROUND and value.get('mode')==MODE):
            value['probe_accepted']=(value.get('status')=='ok'
                and value.get('completed',{}).get('compile')==1
                and (value.get('phase_results',{}).get('compile')or{}).get('native_ABI_verified')is True)
        return save_json(path,value)
    ns['write_json']=persist_current_acceptance
    def clone(function):return FunctionType(function.__code__,ns,function.__name__,function.__defaults__,function.__closure__)
    terminal=clone(original._terminal)
    def finish(args,out,result,started,revision,recorder):
        result.update(probe_layer_index=0,full_model_compile_verified=False,parameter_context_changed=False,
            probe_accepted=False,
            scope='Single original layer0 native probe; full24 source binding does not establish deployment',
            inherited_supervision_span='frozen_full24_single_prefill_compile')
        for module in (original,layered):shutil.copyfile(module.__file__,out/'source'/Path(module.__file__).name)
        terminal(args,out,result,started,revision,recorder)
    ns['_terminal']=finish
    ns['phase_command']=clone(original.phase_command)
    ns['execute']=clone(original.execute)
    return ns


def execute(args):
    namespace=_namespace();source_gate=namespace['frozen_source'];deadline=time.monotonic()+args.seconds
    from . import qwen_component_plan_cache as cache
    proof=[None];save=namespace['write_json']
    def save_with_cache(path,value):
        saved=save(path,value)
        if Path(path).name=='source-record.json':cache.persist(Path(path).parent,proof[0])
        return saved
    namespace['write_json']=save_with_cache
    def finite_source(case):
        # The original execute installs its source handler before this call.
        # Keep a shorter explicit complete budget active during source admission.
        signal.setitimer(signal.ITIMER_REAL,min(600.,max(.001,deadline-time.monotonic())))
        source=source_gate(case);proof[0]=cache.prepare_outer(case,source)
        return source
    namespace['frozen_source']=finite_source
    return namespace['execute'](args)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case',type=Path,required=True);parser.add_argument('--out',type=Path)
    parser.add_argument('--through',choices=tuple(PHASE_SECONDS),default='compile')
    parser.add_argument('--worker-phase',choices=tuple(PHASE_SECONDS))
    for key in ('work-dir','codegen'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--intake-python',type=Path,default=original.paths.INTAKE_PYTHON)
    parser.add_argument('--memory-gib',type=float,default=24.);parser.add_argument('--case-gib',type=int,default=128)
    parser.add_argument('--seconds',type=int,default=10800);parser.add_argument('--phase-seconds',type=int)
    parser.add_argument('--diagnostic-elide-final-ir',action='store_true');args=parser.parse_args(argv)
    resource_limits(args)
    if args.worker_phase:
        if args.work_dir is None or args.codegen is None:parser.error('Inner worker requires explicit new work/codegen paths')
        if type(args.phase_seconds)is not int or not 1<=args.phase_seconds<=PHASE_SECONDS[args.worker_phase]:
            parser.error('Use the finite phase allowance')
        result=worker(args)
    else:
        if args.out is None:parser.error('Use a fresh explicit results case')
        result=execute(args)
    print(json.dumps(dict(status=result['status'],error=result.get('error'),probe_accepted=result.get('probe_accepted',False),
        full_model_compile_verified=False)))
    return int(result['status']!='ok')


if __name__=='__main__':raise SystemExit(main())
