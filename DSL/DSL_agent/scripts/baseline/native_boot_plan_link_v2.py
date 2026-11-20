"""Prototype linker for the B-certified public native ABI; never runs ciphertexts.

This creates a NEW derived artifact. It leaves compiler artifacts untouched,
keeps every compiled arithmetic instruction and SSA edge, and preserves
source physical levels exactly and adds explicit original Runtime rescale/Boot ops.
Actual metadata, numerical validation, key handoff and SDK acceptance remain
required. This module does not open any production backend capability gate.
"""
import copy
import hashlib
import json
import re

CAPABILITY_BINDING='e4e8fe348b7f875d7af1283ca4c51da23090c17f591f42f677a5b4d56d914c9e'
POLICY='poseidon-native-certified-fragment-link-no-translation-prototype-v2'
MAX_VALUES=16384
MAX_OPS=8192
MAX_IO=16
MAX_BYTES=2*1024**2
HOST=dict(kind='host',rank=0)

class LinkError(ValueError):
    pass

def require(ok,reason):
    if not ok:raise LinkError(reason)

def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()

def same(left,right):
    return encode(left)==encode(right)

def _pairs(items):
    result={}
    for key,value in items:
        require(key not in result,'duplicate_json_key');result[key]=value
    return result

def parse(raw):
    require(type(raw)is bytes and 0<len(raw)<=MAX_BYTES,'bounded_json_input')
    try:
        return json.loads(raw.decode('utf-8'),object_pairs_hook=_pairs,
            parse_constant=lambda _:(_ for _ in()).throw(LinkError('nonfinite_json')))
    except (UnicodeError,json.JSONDecodeError)as error:
        raise LinkError('invalid_json')from error

def _id(value):
    return type(value)is str and re.fullmatch(r'0|[1-9][0-9]{0,8}',value)is not None

def authority(capability_raw,spec_raw):
    cap=parse(capability_raw);spec=parse(spec_raw)
    require(type(cap)is dict and cap.get('binding')==CAPABILITY_BINDING and
        digest({k:v for k,v in cap.items()if k!='binding'})==CAPABILITY_BINDING,'capability_identity')
    require(hashlib.sha256(spec_raw).hexdigest()==cap['kernel_source_sha256']['runtime-input/operator-spec.json'],'operator_spec_identity')
    require(cap['context']['security']=='tc128'and cap['context']['poly_degree']==32768 and cap['context']['q_p_bits']==880 and cap['context']['original_tc128_max_bits']==881,'certified_security_context')
    require(cap['evaluator_policy']==dict(home=False,intermediate_decrypt_reencrypt=False,network=False,new_keygen=False,reference=False,secret_key=False),'certified_evaluator_policy')
    require(spec['context']['poly_degree']==32768 and spec['context']['rns_moduli_log2']==[48]*16 and spec['rescale_mode']=='eager','certified_modulus_model')
    return cap,spec

def _metadata(value,context):
    require(type(value)is dict and set(value)=={'id','kind','context','level','scale_log2','components','ntt','place'},'value_fields')
    require(_id(value['id'])and value['kind']=='ciphertext'and value['context']==context and value['ntt']is True and same(value['place'],HOST),'ciphertext_context_or_place')
    require(type(value['level'])is int and 0<=value['level']<=15 and
        type(value['scale_log2'])is int and 1<=value['scale_log2']<=(value['level']+1)*48 and
        type(value['components'])is int and value['components']in(2,3),'ciphertext_type_bounds')
    return value['level'],value['scale_log2'],value['components']

def check_plan(plan,cap,spec,*,allow_boot):
    require(type(plan)is dict and set(plan)=={'format_version','plan_id','target','values','external_inputs','initialization','execution','finalization','final_outputs'},'plan_fields')
    expected_target=dict(capability_version=1,device_counts=[0],operator_spec=dict(id=spec['spec_id'],source_sha256='sha256:'+cap['kernel_source_sha256']['runtime-input/operator-spec.json'],version=1),target_id='poseidon-ckks-cpu',world_size=1)
    require(type(plan['format_version'])is int and plan['format_version']==1 and
        type(plan['plan_id'])is str and re.fullmatch(r'0|[1-9][0-9]{0,19}',plan['plan_id'])is not None and int(plan['plan_id'])<2**64 and same(plan['target'],expected_target),'target_identity')
    require(plan['initialization']==[]and plan['finalization']==[],'no_host_secret_or_transfer_initializers')
    require(type(plan['values'])is list and 1<=len(plan['values'])<=MAX_VALUES and
        type(plan['execution'])is list and len(plan['execution'])<=MAX_OPS,'plan_capacity')
    context=spec['context']['context_id'];values={}
    for value in plan['values']:
        _metadata(value,context);require(value['id']not in values,'duplicate_value');values[value['id']]=value
    for key in ['external_inputs','final_outputs']:
        ids=plan[key];require(type(ids)is list and 1<=len(ids)<=MAX_IO and all(_id(v)and v in values for v in ids)and len(ids)==len(set(ids)),'plan_io')
    available=set(plan['external_inputs'])
    for index,op in enumerate(plan['execution']):
        require(type(op)is dict and {'kind','op','ordinal','inputs','output','place'}<=set(op)<= {'kind','op','ordinal','inputs','output','place','attrs'},'instruction_fields')
        require(op['kind']=='compute'and same(op['place'],HOST)and type(op['ordinal'])is int and op['ordinal']==index,'host_compute_order')
        name=op['op'];require(name in ('mul_cc','relinearize','rescale','boot')and spec['operators'][name]['supported']is True,'uncertified_operation')
        require(type(op['inputs'])is list and all(_id(v)and v in available for v in op['inputs'])and _id(op['output'])and op['output']in values and op['output']not in available,'ssa_order_or_alias')
        operands=[_metadata(values[v],context)for v in op['inputs']];result=_metadata(values[op['output']],context);attrs=op.get('attrs',{})
        if name=='mul_cc':
            require(attrs=={}and len(operands)==2,'multiply_arity_or_attrs')
            a,b=operands;require(a[0]==b[0]and a[2]==b[2]==2 and result==(a[0],a[1]+b[1],3),'multiply_types')
        elif name=='relinearize':
            require(attrs=={}and len(operands)==1,'relinearize_arity_or_attrs');a=operands[0]
            require(a[2]==3 and result==(a[0],a[1],2),'relinearize_types')
        elif name=='rescale':
            require(type(attrs)is dict and set(attrs)=={'target_level','target_scale_log2'}and all(type(v)is int for v in attrs.values())and len(operands)==1,'rescale_attrs')
            a=operands[0];require(a[0]>0 and a[2]==2 and result==(a[0]-1,a[1]-48,2)and result[:2]==(attrs['target_level'],attrs['target_scale_log2']),'original_rescale_types')
        else:
            require(allow_boot and len(operands)==1,'source_fragment_boot_not_admitted');a=operands[0]
            profile={v['level']:v['operator_profile']for v in cap['certified_boot_inputs']}
            require(a[2]==2 and a[:2]in((13,38),(0,40)),'certified_boot_input_nominal_type')
            require(same(attrs,dict(implementation='native',operator_profile=profile[a[0]],target_components=2,target_level=1,target_scale_log2=44))and result==(1,44,2),'certified_native_boot_attributes')
        available.add(op['output'])
    require(available==set(values)and all(v in available for v in plan['final_outputs']),'complete_value_inventory')
    return values

def semantic_dag(plan):
    """Symbolic operator/SSA projection only; no tensor or ciphertext evaluation."""
    references={v:['input',i]for i,v in enumerate(plan['external_inputs'])};nodes=[]
    for op in plan['execution']:
        if op['op']=='mul_cc':
            nodes.append(dict(op='multiply',inputs=[references[v]for v in op['inputs']]))
            references[op['output']]=['node',len(nodes)-1]
        else:
            require(op['op']in('relinearize','rescale','boot'),'nontransparent_projection_operation')
            references[op['output']]=references[op['inputs'][0]]
    return dict(inputs=len(plan['external_inputs']),nodes=nodes,outputs=[references[v]for v in plan['final_outputs']])

def link_fragment(source_raw,capability_raw,spec_raw):
    cap,spec=authority(capability_raw,spec_raw);source=parse(source_raw)
    source_values=check_plan(source,cap,spec,allow_boot=False)
    inputs=[source_values[v]for v in source['external_inputs']]
    require(all(v['scale_log2']==44 and v['components']==2 for v in inputs),'fragment_input_precision_profile')
    source_level=inputs[0]['level'];require(source_level==1 and all(v['level']==1 for v in inputs),'source_fragment_must_already_match_certified_post_Boot_level1')
    delta=0
    require(all(0<=v['level']+delta<=15 and v['scale_log2']<=(v['level']+delta+1)*48 for v in source_values.values()),'insufficient_certified_post_boot_levels')
    linked=dict(format_version=1,plan_id=str(int(hashlib.sha256(source_raw).hexdigest()[:15],16)),target=copy.deepcopy(source['target']),values=[],external_inputs=[],initialization=[],execution=[],finalization=[],final_outputs=[])
    values={};mapping={};pairs=[];prefix=[];suffix=[];normalization=[]
    context=spec['context']['context_id']
    def value(level,scale,components):
        identifier=str(len(linked['values'])+1)
        row=dict(id=identifier,kind='ciphertext',context=context,level=level,scale_log2=scale,components=components,ntt=True,place=copy.deepcopy(HOST))
        _metadata(row,context);linked['values'].append(row);values[identifier]=row;return identifier
    def emit(name,args,out,attrs=None):
        op=dict(kind='compute',op=name,ordinal=len(linked['execution']),inputs=args,output=out,place=copy.deepcopy(HOST))
        if attrs is not None:op['attrs']=attrs
        linked['execution'].append(op);return op['ordinal']
    def boot(identifier,level):
        out=value(1,44,2);profile=next(p for p in cap['certified_boot_inputs']if p['level']==level)
        ordinal=emit('boot',[identifier],out,dict(implementation='native',operator_profile=profile['operator_profile'],target_components=2,target_level=1,target_scale_log2=44))
        return out,ordinal
    for identifier in source['external_inputs']:
        incoming=value(13,38,2);linked['external_inputs'].append(incoming)
        mapping[identifier],ordinal=boot(incoming,13);prefix.append(ordinal)
    for op in source['execution']:
        descriptor=source_values[op['output']];out=value(descriptor['level']+delta,descriptor['scale_log2'],descriptor['components'])
        attrs=copy.deepcopy(op.get('attrs'))
        if op['op']=='rescale':attrs['target_level']+=delta
        ordinal=emit(op['op'],[mapping[v]for v in op['inputs']],out,attrs)
        mapping[op['output']]=out;pairs.append(dict(source_ordinal=op['ordinal'],derived_ordinal=ordinal))
    for identifier in source['final_outputs']:
        out=mapping[identifier];descriptor=values[out]
        if (descriptor['level'],descriptor['scale_log2'],descriptor['components'])==(1,88,2):
            normalized=value(0,40,2);normalization.append(emit('rescale',[out],normalized,dict(target_level=0,target_scale_log2=40)));out=normalized;descriptor=values[out]
        require(descriptor['components']==2,'fragment_output_requires_relinearize')
        if (descriptor['level'],descriptor['scale_log2'])==(0,40):
            out,ordinal=boot(out,0);suffix.append(ordinal)
        else:require((descriptor['level'],descriptor['scale_log2'])==(1,44),'fragment_output_not_certified_next_profile')
        linked['final_outputs'].append(out)
    check_plan(linked,cap,spec,allow_boot=True)
    original_semantics=semantic_dag(source);require(semantic_dag(linked)==original_semantics,'arithmetic_dag_or_output_order_changed')
    for a in source['execution']:
        b=linked['execution'][pairs[a['ordinal']]['derived_ordinal']]
        require(b['op']==a['op']and b['inputs']==[mapping[v]for v in a['inputs']]and b['output']==mapping[a['output']],'instruction_edges_changed')
    encoded=encode(linked)
    require(len(encoded)<=MAX_BYTES,'derived_plan_capacity')
    record=dict(format=POLICY,source_plan_sha256=hashlib.sha256(source_raw).hexdigest(),derived_plan_sha256=hashlib.sha256(encoded).hexdigest(),capability_binding=cap['binding'],operator_spec_sha256=hashlib.sha256(spec_raw).hexdigest(),key_epoch=cap['key_epoch'],uniform_physical_level_delta=delta,source_to_derived_values=mapping,source_instruction_pairs=pairs,prefix_native_boot_ordinals=prefix,output_original_runtime_rescale_ordinals=normalization,suffix_native_boot_ordinals=suffix,semantic_dag_sha256=digest(original_semantics),compiled_arithmetic_instructions_preserved=True,source_artifact_not_modified=True,all_source_physical_levels_and_scales_preserved=True,actual_ciphertext_scales_modified=False,required_original_input_actual_scale=next(p['actual_scale']for p in cap['certified_boot_inputs']if p['level']==13),required_original_output_actual_scale=cap['boot_output']['actual_scale'],runtime_actual_metadata_verification_required=True,original_runtime_verifier_required=True,encrypted_numerical_acceptance_required=True,encrypted_execution_performed=False,Agent_generation_claim=False,SDK_gate_enabled=False)
    record['binding']=digest(record)
    return linked,record
