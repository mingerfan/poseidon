"""Independent Lattigo RuntimePlan adapter: static inventory, prepare and finite probes.

The original compiler target stays in source metadata. This adapter selects a
separate N16/scale43 parameter variant; concrete primes and security are never
inferred from the Poseidon descriptor. Bundle content strings locate files only.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import threading
import time

from .. import paths
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .r27_orbit_native_backend import _environment, _tree_bytes
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample
from .completion_metrics import persist_case_completion
from .r31_lattigo_binding import NativeBinding,map_precision_plan,dft_static_inventory

ROUND = 'R31'
MODE = 'independent-lattigo-runtime-plan-adapter'
VARIANT = 'lattigo-n16-q55-43x8-p61x3-scale43-logicalslots14-lazy1'
OPS = {'mod_switch': 1, 'rotate': 1, 'mul_cp': 2, 'rescale': 1, 'add_cc': 2,
       'add_cp': 2, 'boot': 1, 'mul_cc': 2, 'relinearize': 1, 'negate': 1}
GIB = 1024**3
GO_SOURCE = Path(__file__).with_suffix('.go')


def runtime_variant(capacity,binding=None):
    return (binding or NativeBinding()).variant(capacity)


def checkpoint_arguments(prepared, interval=1):
    """Select sparse snapshots only when the actual binary proved support."""
    if type(interval) is not int or not 1 <= interval <= 1000000:
        raise ValueError('Checkpoint interval must be an integer from 1 through 1000000')
    if interval == 1:
        return []
    policy = prepared.get('cache_selftest', {}).get('checkpoint_policy', {})
    if (policy.get('unit') != 'existing_save_request' or policy.get('interval') != 1
            or policy.get('forced_requests', 0) < 1
            or policy.get('writes_completed', 0) < 1 or policy.get('write_errors') != 0):
        raise ValueError('Prepared binary lacks an actual checkpoint-policy self-test')
    return ['-checkpoint-interval', str(interval)]


def auxiliary_arguments(prepared, bits):
    if type(bits) is not int or bits not in (57, 61):
        raise ValueError('Explicit auxiliary prime size is57 or61')
    supported = prepared.get('supported_auxiliary_prime_bits', [])
    if bits != 61 and bits not in supported:
        raise ValueError('Prepared binary lacks the explicit auxiliary-prime binding')
    return ['-auxiliary-prime-bits', str(bits)] if supported else []


def c2s_arguments(prepared, bits):
    if bits is None:
        return []
    if type(bits) is not int or bits!=52:
        raise ValueError('Explicit C2S precision uses52 bits in four layers')
    if (prepared.get('client_cache_protocol',0)<10
            or bits not in prepared.get('supported_c2s_bits',[])):
        raise ValueError('Prepared binary lacks the explicit C2S52x4 binding')
    return ['-c2s-bits',str(bits)]


def check_bound_target(plan, binding, capacity):
    target = plan.get('target', {})
    if target.get('target_id') == 'pt-agent-lattigo-native-adapter':
        if (target.get('adapter_variant') != binding.variant(capacity)
                or target.get('precision_bits') != binding.precision_bits):
            raise ValueError('Mapped plan provenance differs from the explicit native binding')


def content_filename(content):
    if not isinstance(content, str) or ':' not in content:
        raise ValueError('Bundle content must be an existing namespaced identifier')
    leaf = content.split(':', 1)[1]
    if not re.fullmatch(r'[A-Za-z0-9_-]+', leaf):
        raise ValueError('Bundle content locator has an unsafe path component')
    return leaf + '.bin'


def physical_memory_observation(sample, available, *, rss_gib=14, minimum_available_gib=1):
    """Address-space limits remain separate from observed physical memory."""
    subtree=None if sample is None else sample['subtree_rss_bytes']
    reasons=[]
    if subtree is not None and subtree>rss_gib*GIB:reasons.append('child subtree RSS allowance exceeded')
    if available<minimum_available_gib*GIB:reasons.append('host available RAM reserve exhausted')
    return dict(stop=bool(reasons),reasons=reasons,subtree_rss_bytes=subtree,
        available_ram_bytes=available,rss_allowance_bytes=int(rss_gib*GIB),
        minimum_available_bytes=int(minimum_available_gib*GIB),
        sample_ts_ns=None if sample is None else sample.get('sample_ts_ns'))


def plan_inventory(plan, manifest, *, log_n=16, logical_slots=14,binding=None):
    """Validate symbolic plan semantics and count live coefficient payload only."""
    binding=binding or NativeBinding();unit=binding.precision_bits
    if plan.get('format_version') != 1 or plan.get('finalization') != []:
        raise ValueError('Only format-1 straight-line plans with empty finalization are supported')
    values = {}
    for value in plan['values']:
        name = value['id']
        if name in values or not isinstance(name, str):
            raise ValueError('Duplicate or non-string value identifier')
        if (value.get('kind') not in ('ciphertext', 'plaintext') or value.get('place') != {'kind':'host','rank':0}
                or value.get('ntt') is not True or value.get('context') != 'qwen-bootstrap48'
                or type(value.get('level')) is not int or not 0 <= value['level'] <= 8
                or type(value.get('scale_log2')) is not int or not 0 <= value['scale_log2'] <= 1023
                or value.get('components') not in ((2,3) if value['kind']=='ciphertext' else (1,))):
            raise ValueError('Unsupported value metadata: ' + name)
        values[name] = value
    if len(plan['external_inputs']) != 1 or len(plan['final_outputs']) != 1:
        raise ValueError('The adapter requires one ciphertext input and one output')
    if any(values[x]['kind'] != 'ciphertext' or values[x]['components'] != 2
           for x in plan['external_inputs'] + plan['final_outputs']):
        raise ValueError('Input/output must be degree-one ciphertexts')
    if (manifest.get('bundle_format_version'),manifest.get('version')) != (1,1):
        raise ValueError('Unsupported plaintext bundle version')
    bundle = plan.get('plaintext_bundle') or {}
    if bundle.get('id') != manifest.get('bundle_id') or bundle.get('version') != 1:
        raise ValueError('Plan and manifest bundle identifiers differ')
    blobs = {}
    for blob in manifest['blobs']:
        name = blob['content']; content_filename(name)
        if name in blobs or blob['byte_length'] != (1 << logical_slots) * 8:
            raise ValueError('Duplicate or incorrectly sized raw float64 bundle blob')
        blobs[name] = blob
    defined = set(plan['external_inputs']); encodes = {}
    for item in plan['initialization']:
        name = item['output']; payload = item.get('payload', {})
        if (item.get('kind') != 'encode' or name in defined or values[name]['kind'] != 'plaintext'
                or payload.get('kind') != 'bundle' or payload.get('content') not in blobs):
            raise ValueError('Unsupported or duplicate initialization')
        defined.add(name); encodes[name] = payload['content']
    counts=Counter(); rotations=set(); last={}; ordinals=set()
    for item in plan['initialization']+plan['execution']:
        ordinal=item['ordinal']
        if type(ordinal) is not int or ordinal in ordinals:
            raise ValueError('Instruction ordinals must be unique integers')
        ordinals.add(ordinal)
    for index, item in enumerate(plan['execution']):
        op=item.get('op'); inputs=item.get('inputs',[]); name=item['output']; attrs=item.get('attrs',{})
        if (item.get('kind')!='compute' or op not in OPS or len(inputs)!=OPS[op]
                or item.get('place')!={'kind':'host','rank':0} or name in defined
                or any(x not in defined for x in inputs)):
            raise ValueError('Unsupported/non-topological execution instruction')
        a=values[inputs[0]]; out=values[name]
        if a['kind']!='ciphertext' or out['kind']!='ciphertext':
            raise ValueError('Every operation starts with and produces a ciphertext')
        level,scale,components=a['level'],a['scale_log2'],a['components']
        if len(inputs)==2:
            b=values[inputs[1]]
            if b['level']!=level or b['context']!=a['context']:
                raise ValueError('Binary operand context/level mismatch')
            required='plaintext' if op.endswith('_cp') else 'ciphertext'
            if b['kind']!=required:
                raise ValueError('Binary operand kind mismatch')
            if op.startswith('add'):
                if b['scale_log2']!=scale:raise ValueError('Nominal addition scales differ')
                components=max(components,b['components'])
            else:
                scale+=b['scale_log2']; components+=b['components']-1
        if op in ('mod_switch','rescale'):
            target=attrs.get('target_level')
            if type(target) is not int or not 0<=target<level:raise ValueError('Invalid target level')
            if op=='rescale':
                if target!=level-1:raise ValueError('Adapter rescale consumes one native prime')
                scale=attrs['target_scale_log2']
                if a['scale_log2']-scale!=unit:raise ValueError('Rescale nominal prime differs from the explicit native binding')
            level=target
        elif op=='relinearize':
            if components!=3:raise ValueError('Relinearize requires degree two')
            components=2
        elif op=='boot':
            if attrs.get('implementation')!='native':raise ValueError('Only native bootstrap is supported')
            level,scale,components=(attrs[k] for k in ('target_level','target_scale_log2','target_components'))
            if not 0<=level<=8 or scale!=unit or components!=2:raise ValueError('Unsupported native bootstrap output')
        elif op=='rotate':
            if type(attrs.get('steps')) is not int or components!=2:raise ValueError('Invalid rotation')
            rotations.add(attrs['steps']%(1<<logical_slots))
        if (out['level'],out['scale_log2'],out['components'])!=(level,scale,components):
            raise ValueError('Output metadata disagrees with symbolic operation: '+str(item['ordinal']))
        defined.add(name);counts[op]+=1
        for operand in inputs:last[operand]=index
    if set(values)!=defined or any(x not in defined for x in plan['final_outputs']):
        raise ValueError('Plan contains missing/unproduced values')
    live=set(plan['external_inputs']); peak_count=len(live)
    payload=lambda name:values[name]['components']*(1<<log_n)*(values[name]['level']+1)*8
    peak_bytes=sum(map(payload,live));eager=sum((1<<log_n)*(values[x]['level']+1)*8 for x in encodes)
    final=set(plan['final_outputs'])
    for i,item in enumerate(plan['execution']):
        live.add(item['output']);peak_count=max(peak_count,len(live));peak_bytes=max(peak_bytes,sum(map(payload,live)))
        for name in set(item['inputs']):
            if last[name]==i and name not in final:live.discard(name)
        if item['output'] not in last and item['output'] not in final:live.discard(item['output'])
    return dict(value_count=len(values),execution_count=len(plan['execution']),encode_count=len(encodes),
        unique_blob_count=len(blobs),op_counts=dict(sorted(counts.items())),bootstrap_count=counts['boot'],
        rotation_steps=sorted(rotations-{0}),logical_slots=logical_slots,slot_count=1<<logical_slots,
        last_use_ciphertext_peak_count=peak_count,last_use_ciphertext_peak_payload_bytes=peak_bytes,
        eager_plaintext_payload_bytes=eager,lazy_plaintext_peak_payload_bytes=max(
            [(1<<log_n)*(values[x]['level']+1)*8 for x in encodes] or [0]),
        terminal_ciphertext_ids=sorted(live),coefficient_payload_scope='Ring coefficients only; evaluator, keys, encoder/DFT scratch and allocator overhead excluded',
        concrete_prime_binding_verified=False,security_acceptance=None,extra_integrity_checks=0,
        content_identifiers='Existing strings used only as data/<suffix>.bin locators; no digest computation or comparison')


def make_regression(work):
    """Deterministic public fixture exercising the same N16 residual variant."""
    bundle=work/'fixture.bundle';(bundle/'data').mkdir(parents=True)
    def desc(name,level,scale=43,components=2,kind='ciphertext'):
        return dict(id=name,level=level,scale_log2=scale,components=components,kind=kind,
                    context='qwen-bootstrap48',ntt=True,place=dict(kind='host',rank=0))
    values=[desc('x',5),desc('quarter',4,kind='plaintext',components=1),desc('half',3,kind='plaintext',components=1)]
    instructions=[]
    def op(name,code,inputs,level,scale=43,components=2,**attrs):
        values.append(desc(name,level,scale,components));item=dict(kind='compute',ordinal=len(instructions)+2,
            op=code,inputs=inputs,output=name,place=dict(kind='host',rank=0))
        if attrs:item['attrs']=attrs
        instructions.append(item)
    op('r','rotate',['x'],5,steps=1);op('square','mul_cc',['r','r'],5,86,3)
    op('relin','relinearize',['square'],5,86);op('scaled','rescale',['relin'],4,target_level=4,target_scale_log2=43)
    op('shifted','add_cp',['scaled','quarter'],4);op('negative','negate',['shifted'],4)
    op('back','rotate',['negative'],4,steps=-1);op('down','mod_switch',['back'],3,target_level=3)
    op('product','mul_cp',['down','half'],3,86);op('lastscale','rescale',['product'],2,target_level=2,target_scale_log2=43)
    op('y','add_cc',['lastscale','lastscale'],2)
    init=[];blobs=[]
    for ordinal,(name,number) in enumerate((('quarter',.25),('half',.5))):
        content='fixture:'+name;raw=struct.pack('<d',number)*(1<<14)
        (bundle/'data'/content_filename(content)).write_bytes(raw)
        blobs.append(dict(content=content,byte_length=len(raw)))
        init.append(dict(kind='encode',ordinal=ordinal,output=name,payload=dict(kind='bundle',content=content)))
    plan=dict(format_version=1,plan_id=1,target=dict(target_id='poseidon-ckks-cpu',fixture=True),
        values=values,external_inputs=['x'],initialization=init,execution=instructions,finalization=[],final_outputs=['y'],
        plaintext_bundle=dict(id='r31-public-unit-fixture',version=1))
    manifest=dict(bundle_format_version=1,bundle_id='r31-public-unit-fixture',version=1,blobs=blobs)
    write_json(work/'fixture-plan.json',plan);write_json(bundle/'manifest.json',manifest)
    numbers=[(i%17-8)/32 for i in range(1<<14)]
    (work/'input.bin').write_bytes(struct.pack('<'+'d'*len(numbers),*numbers))
    write_json(work/'expected.json',dict(indices=list(range(len(numbers))),values=[-x*x-.25 for x in numbers],
        scope='Deterministic unit fixture only; no model/calibration/validation sample'))
    return work/'fixture-plan.json',bundle,work/'input.bin',work/'expected.json'


def execute(args):
    binding=NativeBinding(getattr(args,'precision_bits',43),getattr(args,'s2c_bits',None),getattr(args,'mod1_inv_degree',None),getattr(args,'mod1_degree',None),getattr(args,'log_message_ratio',None),getattr(args,'bootstrap_input_divisor',None),getattr(args,'bootstrap_normalization_log2',0),getattr(args,'auxiliary_prime_bits',61),getattr(args,'c2s_bits',None))
    started=time.perf_counter_ns();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    work=out/'work';work.mkdir();recorder=StageRecorder(out/'events.jsonl')
    result=dict(round=ROUND,mode=MODE,variant=runtime_variant(args.bootstrap_cache,binding),precision_bits=binding.precision_bits,phase=args.phase,status='running',error=None,
        original_libraries_modified=False,extra_integrity_checks=0,llm_calls=0,downloads=0,installations=0,
        security_acceptance=None,accuracy_acceptance=None,CKKS_latency=None,compiled_bootstraps=None,
        formal_matrix_complete=False,processes=[],artifact_bytes=0,source_plan=str(args.plan),source_bundle=str(args.bundle))
    result['requested_native_parameters']=dict(log_n=16,log_q=[55]+[binding.precision_bits]*8,
        log_p=[binding.auxiliary_prime_bits]*3,log_default_scale=binding.precision_bits,logical_log_slots=14,
        bootstrap_log_p=[binding.auxiliary_prime_bits]*4,actual_constructor_primes_pending=True)
    if binding.s2c_bits is not None:
        result['requested_native_parameters']['bootstrap_slots_to_coeffs_factorization_depth_and_log_scales']=[[binding.s2c_bits] for _ in range(3)]
    if binding.c2s_bits is not None:
        result['requested_native_parameters']['bootstrap_coeffs_to_slots_factorization_depth_and_log_scales']=[[binding.c2s_bits] for _ in range(4)]
    if binding.mod1_inv_degree is not None:result['requested_native_parameters']['bootstrap_mod1_inv_degree']=binding.mod1_inv_degree
    if binding.mod1_degree is not None:result['requested_native_parameters']['bootstrap_mod1_degree']=binding.mod1_degree
    if binding.log_message_ratio is not None:result['requested_native_parameters']['bootstrap_log_message_ratio']=binding.log_message_ratio
    if binding.value_divisor:result['bootstrap_wrapper_policy']=binding.metadata()['bootstrap_wrapper_contract']
    result['source_model_result']=None
    physical_phase=args.phase in ('execute','bootstrap-regression')
    result['memory_policy']=dict(address_space_gib=args.native_address_space_gib if physical_phase else args.memory_gib,
        subtree_rss_gib=args.native_rss_gib if physical_phase else None,
        minimum_available_ram_gib=args.minimum_available_gib if physical_phase else None,
        go_soft_memory_limit=args.native_go_memory_limit if physical_phase else '1500MiB',
        physical_memory_expansion=False)
    deadline=time.monotonic()+args.seconds;stop=threading.Event();failures=[]
    active_resources=[None]
    def check():
        if failures:raise RuntimeError(failures[0])
        if time.monotonic()>deadline:raise TimeoutError('R31 phase wall allowance exhausted')
        if _tree_bytes(out)>args.case_gib*GIB or shutil.disk_usage(out).free<10*GIB:
            raise RuntimeError('R31 disk allowance exhausted')
    old={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    def interrupted(sig,frame):raise RuntimeError('R31 interrupted: '+str(sig))
    for sig in old:signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,args.seconds)
    def monitor():
        next_disk=0
        while not stop.wait(.1):
            try:
                if time.monotonic()>=next_disk:
                    check();next_disk=time.monotonic()+1
                if physical_phase and active_resources[0] is not None:
                    observed=physical_memory_observation(_latest_rss_sample(active_resources[0]),
                        _available_ram_bytes(),rss_gib=args.native_rss_gib,
                        minimum_available_gib=args.minimum_available_gib)
                    if observed['stop']:
                        result['memory_guard_observation']=observed
                        write_json(out/'memory-guard-observation.json',observed)
                        raise RuntimeError('R31 physical RSS/available-RAM guard stopped the child tree')
            except BaseException as exc:
                failures.append(type(exc).__name__+': '+str(exc));os.kill(os.getpid(),signal.SIGTERM);return
    watcher=threading.Thread(target=monitor,daemon=True)
    env=None
    def phase(name,argv,timeout=600,mem=2,cwd=None):
        check();row=dict(stage=name,argv=[str(x) for x in argv]);result['processes'].append(row)
        with recorder.span(name):
            active_resources[0]=out/(name+'-resources.jsonl')
            try:
                row['observation']=run_process(row['argv'],cwd=cwd or work,log=out/(name+'.log'),
                    resources=active_resources[0],timeout=min(timeout,max(.01,deadline-time.monotonic())),
                    mem_gib=mem,fsize_gib=8,env=env)
            finally:active_resources[0]=None
        obs=row['observation'];check()
        if obs['exit_code']!=0 or obs['timed_out'] or obs['signal'] is not None or obs['remaining_pids']:
            raise RuntimeError('R31 child failed: '+name)
    try:
        watcher.start()
        shutil.copyfile(Path(__file__),out/'driver.py');shutil.copyfile(GO_SOURCE,out/'client.go')
        shutil.copyfile(Path(__file__).with_name('r31_lattigo_binding.py'),out/'native-binding.py')
        with recorder.span('source_plan_inventory'):
            plan=json.loads(args.plan.read_text());manifest=json.loads((args.bundle/'manifest.json').read_text())
            result['inventory']=plan_inventory(plan,manifest,binding=binding);result['original_target']=plan['target']
            source_result=getattr(args,'source_result',None)
            if source_result is not None:
                original=json.loads(source_result.read_text())
                result['source_model_result']=dict(path=str(source_result),round=original.get('round'),
                    status=original.get('status'),model=original.get('model'),system=original.get('system'),
                    source_model_info=original.get('source_model_info'),source=original.get('source'))
            write_json(out/'inventory.json',result['inventory'])
        if args.phase!='inventory':
            for name in ('tmp','telemetry','gopath'):(work/name).mkdir()
            (work/'telemetry/mode').write_text('off\n');env=_environment(work,args.go_sdk)
            env['GOMEMLIMIT']=args.native_go_memory_limit if physical_phase else '1500MiB'
            if args.phase=='prepare':
                prior=json.loads((args.prepared/'result.json').read_text())
                if prior.get('status')!='ok':raise ValueError('Successful isolated R27 preparation required')
                root=args.prepared/'work/lattigo'
                key_source=(root/'core/rlwe/keys.go').read_text()
                if 'GetGaloisKeyFunc func(uint64)' not in key_source:raise ValueError('Isolated lazy-key callbacks missing')
                module=work/'module';module.mkdir();shutil.copyfile(GO_SOURCE,module/'main.go')
                for name in ('module-cache','build-cache'):shutil.copytree(args.prepared/'work'/name,work/name)
                text=(args.prepared/'work/module/go.mod').read_text().replace('module pt-agent-r27-lazy','module pt-agent-r31-plan')
                (module/'go.mod').write_text(text);shutil.copyfile(args.prepared/'work/module/go.sum',module/'go.sum')
                phase('build',[args.go_sdk/'bin/go','build','-p=1','-buildvcs=false','-mod=readonly','-o',work/'r31-client','.'],cwd=module)
                result['ready_binary']=str(work/'r31-client');result['lattigo_source']=str(root)
                result['client_cache_protocol']=10
                result['supports_input_ssa_check']=True
                result['supported_bootstrap_cache_capacities']=[1,16,32,40,44,48]
                result['supported_precision_bits']=[43,50]
                result['supported_s2c_bits']=[50]
                result['supported_c2s_bits']=[52]
                result['supported_mod1_inv_degrees']=[7,15]
                result['supported_mod1_degrees']=[63]
                result['supported_log_message_ratios']=[2]
                result['supported_bootstrap_input_divisors']=[2]
                result['supported_bootstrap_normalizations_log2']=[0,1,2,3]
                result['supported_auxiliary_prime_bits']=[57,61]
                result['supports_key_request_trace']=True
                binary=work/'r31-client'
                phase('cache-regression',[binary,'-mode','cache-selftest','-bootstrap-cache',str(args.bootstrap_cache),'-precision-bits',str(binding.precision_bits),'-s2c-bits',str(binding.s2c_bits or 0),'-c2s-bits',str(binding.c2s_bits or 0),'-mod1-inv-degree',str(binding.mod1_inv_degree or 0),'-mod1-degree',str(binding.mod1_degree or 0),'-log-message-ratio',str(binding.log_message_ratio or 0),'-bootstrap-input-divisor',str(binding.bootstrap_input_divisor or 0),'-bootstrap-normalization-log2',str(binding.bootstrap_normalization_log2),
                    '-auxiliary-prime-bits',str(binding.auxiliary_prime_bits),
                    '-output',out/'cache-selftest.json','-events',out/'cache-selftest-events.jsonl'],timeout=30,mem=2)
                cache_test=json.loads((out/'cache-selftest.json').read_text())
                if cache_test.get('status')!='ok' or cache_test.get('cache_selftest_passed') is not True or cache_test.get('keygen_api_attempts')!=0:
                    raise RuntimeError('Key-free cache regression failed')
                result['cache_selftest']=cache_test
                command=[binary,'-mode','parameters','-plan',args.plan,'-bundle',args.bundle]
            else:
                prior=json.loads((args.prepared/'result.json').read_text())
                if (prior.get('status')!='ok' or prior.get('phase')!='prepare'
                        or (prior.get('variant')!=result['variant'] and not(
                            prior.get('client_cache_protocol',0)>=1 and args.bootstrap_cache in prior.get('supported_bootstrap_cache_capacities',[])))):
                    raise ValueError('A successful R31 prepare result is required')
                if binding.precision_bits!=prior.get('precision_bits',43) and not(
                        prior.get('client_cache_protocol',0)>=2 and binding.precision_bits in prior.get('supported_precision_bits',[])):
                    raise ValueError('Prepared binary does not support the requested explicit precision binding')
                if binding.s2c_bits is not None and not(
                        prior.get('client_cache_protocol',0)>=3 and binding.s2c_bits in prior.get('supported_s2c_bits',[])):
                    raise ValueError('Prepared binary does not support the requested explicit S2C precision binding')
                if binding.mod1_inv_degree is not None and not(
                        prior.get('client_cache_protocol',0)>=4 and binding.mod1_inv_degree in prior.get('supported_mod1_inv_degrees',[])):
                    raise ValueError('Prepared binary does not support the requested explicit inverse-polynomial binding')
                if binding.mod1_degree is not None and not(
                        prior.get('client_cache_protocol',0)>=5 and binding.mod1_degree in prior.get('supported_mod1_degrees',[])):
                    raise ValueError('Prepared binary does not support the requested explicit Mod1-degree binding')
                if getattr(args,'check_input_ssa',False) and prior.get('supports_input_ssa_check') is not True:
                    raise ValueError('Prepared binary does not support input SSA metadata observations')
                if binding.log_message_ratio is not None and not(
                        prior.get('client_cache_protocol',0)>=7 and binding.log_message_ratio in prior.get('supported_log_message_ratios',[])):
                    raise ValueError('Prepared binary does not support the requested explicit message ratio')
                if binding.bootstrap_normalization_log2 and not(prior.get('client_cache_protocol',0)>=9 and binding.bootstrap_normalization_log2 in prior.get('supported_bootstrap_normalizations_log2',[])):
                    raise ValueError('Prepared client lacks explicit log2 bootstrap normalization')
                if binding.bootstrap_input_divisor is not None and not(
                        prior.get('client_cache_protocol',0)>=8 and binding.bootstrap_input_divisor in prior.get('supported_bootstrap_input_divisors',[])):
                    raise ValueError('Prepared binary does not support the explicit value-normalizing bootstrap wrapper')
                if getattr(args,'trace_key_requests',False) and prior.get('supports_key_request_trace') is not True:
                    raise ValueError('Prepared binary does not support public key-request observations')
                selected_auxiliary = auxiliary_arguments(prior, binding.auxiliary_prime_bits)
                selected_c2s = c2s_arguments(prior, binding.c2s_bits)
                binary=Path(prior['ready_binary'])
                result['prepared_result']=str(args.prepared/'result.json')
                result['prepared_binary']=str(binary)
                # Preserve the source snapshot from the actual binary's build.
                shutil.copyfile(args.prepared/'client.go',out/'client.go')
                result['prepared_client_source']=str(args.prepared/'client.go')
                interval = getattr(args, 'checkpoint_interval', 1)
                selected_checkpoints = checkpoint_arguments(prior, interval)
                result['checkpoint_interval'] = interval
                result['checkpoint_unit'] = 'existing_save_request'
                if args.phase=='regression':
                    with recorder.span('deterministic_fixture'):
                        selected,bundle,input_path,expected=make_regression(work)
                    if binding.precision_bits!=43:
                        mapped,report=map_precision_plan(json.loads(selected.read_text()),binding,bootstrap_cache=args.bootstrap_cache)
                        selected=work/'precision-bound-fixture-plan.json';write_json(selected,mapped);result['fixture_precision_mapping']=report
                    result['fixture_inventory']=plan_inventory(json.loads(selected.read_text()),json.loads((bundle/'manifest.json').read_text()),binding=binding)
                elif args.phase=='bootstrap-regression':
                    if any(getattr(args,k) is None for k in ('program','r13_result','probes','probe_source','source_result')):
                        raise ValueError('The native bootstrap probe requires explicit original training/source paths')
                    input_work=work/'input-prep'
                    source_worker=Path(__file__).with_name('r31_lattigo_input_worker.py')
                    shutil.copyfile(source_worker,out/'input-worker.py')
                    command=[args.pack_python,'-B','-m','pt_agent.experiments.r31_lattigo_input_worker',
                        '--mode','bootstrap-identity','--program',args.program,'--r13-result',args.r13_result,
                        '--probes',args.probes,'--probe-source',args.probe_source,
                        '--bootstrap-count',str(args.bootstrap_count),
                        '--compilation-result',args.source_result,'--work-dir',input_work]
                    if args.branch:command+=['--branch']
                    phase('prepare-training-input',command,timeout=120,mem=2,
                        cwd=paths.REPO/'DSL/DSL_agent/scripts')
                    source=json.loads((input_work/'input-result.json').read_text())
                    if source.get('status')!='ok':raise RuntimeError('Saved training input preparation failed')
                    selected,bundle,input_path,expected=(Path(source[k]) for k in ('plan','bundle','input','expected'))
                    result['training_input_source']=source
                    if binding.precision_bits!=43:
                        mapped,report=map_precision_plan(json.loads(selected.read_text()),binding,bootstrap_cache=args.bootstrap_cache)
                        selected=work/'precision-bound-fixture-plan.json';write_json(selected,mapped);result['fixture_precision_mapping']=report
                    result['fixture_inventory']=plan_inventory(json.loads(selected.read_text()),json.loads((bundle/'manifest.json').read_text()),binding=binding)
                else:
                    selected,bundle,input_path,expected=args.plan,args.bundle,args.input,args.expected
                    if input_path is None or expected is None:raise ValueError('Execution requires explicit packed input and final reference')
                    if result['inventory']['bootstrap_count'] and not args.allow_bootstrap:raise ValueError('Bootstrap execution must be explicitly selected')
                check_bound_target(json.loads(Path(selected).read_text()), binding, args.bootstrap_cache)
                command=[binary,'-mode','execute','-plan',selected,'-bundle',bundle,'-input',input_path,'-expected',expected]
                command += selected_checkpoints
                if args.allow_bootstrap or args.phase=='bootstrap-regression':command+=['-allow-bootstrap']
            if args.phase=='prepare' or prior.get('client_cache_protocol',0)>=1:
                command+=['-bootstrap-cache',str(args.bootstrap_cache)]
            if args.phase=='prepare' or prior.get('client_cache_protocol',0)>=2:
                command+=['-precision-bits',str(binding.precision_bits)]
            if binding.s2c_bits is not None:
                command+=['-s2c-bits',str(binding.s2c_bits)]
            if binding.c2s_bits is not None:
                command+=(['-c2s-bits',str(binding.c2s_bits)] if args.phase=='prepare' else selected_c2s)
            if binding.mod1_inv_degree is not None:
                command+=['-mod1-inv-degree',str(binding.mod1_inv_degree)]
            if binding.mod1_degree is not None:
                command+=['-mod1-degree',str(binding.mod1_degree)]
            if binding.log_message_ratio is not None:command+=['-log-message-ratio',str(binding.log_message_ratio)]
            if binding.bootstrap_input_divisor is not None:command+=['-bootstrap-input-divisor',str(binding.bootstrap_input_divisor)]
            if binding.bootstrap_normalization_log2:command+=['-bootstrap-normalization-log2',str(binding.bootstrap_normalization_log2)]
            command += (['-auxiliary-prime-bits',str(binding.auxiliary_prime_bits)]
                        if args.phase=='prepare' else selected_auxiliary)
            if getattr(args,'trace_key_requests',False):command+=['-trace-key-requests']
            if getattr(args,'check_input_ssa',False):command+=['-check-input-ssa']
            command+=['-output',out/'client.json','-events',out/'go-events.jsonl']
            phase('parameters' if args.phase=='prepare' else args.phase,command,timeout=args.seconds,
                mem=args.native_address_space_gib if physical_phase else args.memory_gib)
            result['client']=json.loads((out/'client.json').read_text())
            result['bootstrap_static_inventory']=dft_static_inventory(result['client']['bootstrap_parameters'])
            result['requested_native_parameters']['actual_constructor_primes_pending']=False
            if result['client'].get('status')!='ok':raise RuntimeError('R31 client did not complete')
            if args.phase=='prepare' and result['client'].get('keygen_api_attempts')!=0:raise RuntimeError('Parameter preparation generated keys')
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        stop.set()
        if watcher.is_alive():watcher.join(timeout=3)
        if (out/'client.json').is_file():result['client']=json.loads((out/'client.json').read_text())
        result['timing']=summarize_events(read_events(out/'events.jsonl'))
        result['recorded_elapsed_ns']=time.perf_counter_ns()-started
        result['metrics']={k:result.get(k) for k in ('status','phase','variant','inventory','security_acceptance','accuracy_acceptance','CKKS_latency','llm_calls','extra_integrity_checks','recorded_elapsed_ns')}
        result['metrics']['native_counts']={k:result.get('client',{}).get(k) for k in ('keygen_api_attempts','keygen_api_completed','compute_attempts','compute_completed','encryption_attempts','final_decryption_attempts','bootstrap_attempts','bootstrap_completed')}
        result['metrics']['memory_policy']=result['memory_policy']
        result['metrics']['memory_guard_observation']=result.get('memory_guard_observation')
        for _ in range(20):
            write_json(out/'result.json',result);size=_tree_bytes(out)
            if size>args.case_gib*GIB and result['status']!='error':
                result.update(status='error',error='R31 terminal artifacts exceeded the case allowance')
                result['metrics']['status']='error'
            if size==result['artifact_bytes']:break
            result['artifact_bytes']=size
        else:raise RuntimeError('R31 terminal artifact size did not converge')
        from ..registry import git_rev
        row=dict(run_id=out.name,round=ROUND,model='runtime-plan',mode=MODE,git_rev=git_rev(),
            metrics=result['metrics'],artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
        with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],
            timing=result['timing'],required_artifact_bytes=result['artifact_bytes'])
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in old.items():signal.signal(sig,handler)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inventory','prepare','regression','bootstrap-regression','execute'))
    for name in ('out','plan','bundle'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-result',type=Path,help='Optional original compilation/source result metadata')
    p.add_argument('--prepared',type=Path,default=paths.RESULTS_DIR/'orbit-lazy-lifetime-prepare-r27-20261116')
    p.add_argument('--go-sdk',type=Path,default=paths.ORBIT_ROOT.parent/'toolchains/go125/go')
    p.add_argument('--input',type=Path);p.add_argument('--expected',type=Path);p.add_argument('--allow-bootstrap',action='store_true')
    p.add_argument('--bootstrap-cache',type=int,choices=(1,16,32,40,44,48),default=1)
    p.add_argument('--precision-bits',type=int,choices=(43,50),default=43)
    p.add_argument('--s2c-bits',type=int,choices=(50,))
    p.add_argument('--c2s-bits',type=int,choices=(52,))
    p.add_argument('--mod1-inv-degree',type=int,choices=(7,15))
    p.add_argument('--mod1-degree',type=int,choices=(63,))
    p.add_argument('--log-message-ratio',type=int,choices=(2,))
    p.add_argument('--bootstrap-input-divisor',type=int,choices=(2,))
    p.add_argument('--bootstrap-normalization-log2',type=int,choices=(0,1,2,3),default=0)
    p.add_argument('--auxiliary-prime-bits', type=int, choices=(57,61), default=61)
    p.add_argument('--trace-key-requests',action='store_true')
    p.add_argument('--bootstrap-count',type=int,choices=(1,2,3),default=1)
    p.add_argument('--branch',action='store_true')
    p.add_argument('--check-input-ssa',action='store_true')
    p.add_argument('--checkpoint-interval', type=int, default=1,
                   help='Full-record save-request interval; every event and final observation is retained')
    for name in ('program','r13-result','probes','probe-source'):p.add_argument('--'+name,type=Path)
    p.add_argument('--pack-python',type=Path,default=paths.ORBIT_ROOT.parent/'orbit-venv/bin/python')
    p.add_argument('--seconds',type=float,default=900);p.add_argument('--memory-gib',type=float,default=4)
    p.add_argument('--native-address-space-gib',type=float,default=24)
    p.add_argument('--native-rss-gib',type=float,default=14)
    p.add_argument('--minimum-available-gib',type=float,default=1)
    p.add_argument('--native-go-memory-limit',default='11GiB')
    p.add_argument('--case-gib',type=float,default=4);args=p.parse_args(argv)
    if min(args.seconds,args.memory_gib,args.case_gib,args.native_address_space_gib,
           args.native_rss_gib,args.minimum_available_gib)<=0:
        p.error('Positive resource allowances required')
    if not 1 <= args.checkpoint_interval <= 1000000:
        p.error('Checkpoint interval must be from 1 through 1000000')
    result=execute(args);print(json.dumps(dict(status=result['status'],error=result['error'],out=str(args.out))))
    return 0 if result['status']=='ok' else 1

if __name__=='__main__':raise SystemExit(main())
