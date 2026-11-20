"""Versioned private mathematical evaluation for retained dual references.

No generation, compilation, CKKS or candidate Python execution. All three input
identities and the source/declared function identities freeze before generation.
"""
import copy
import math
from pathlib import Path
import sys

import numpy as np
from benchmark_graph import validate
from delivery_component import bound,canonical,checked_bound,read_bytes,require,sha,write_once
from first_output_assessment import evaluation_policy,MATH,CKKS,compare
from first_output_dsl_math import evaluate as dsl_mathematics
import first_output_math_producer as original
import retained_dual_reference_rows as references
import retained_math_domains as domains

SPEC='poseidon-retained-mathematical-producer-v1'
RECORD='poseidon-retained-three-layer-mathematical-evidence-v1'
DOMAIN_RECORD='poseidon-retained-effective-domain-evidence-v1'


def implementation_bundle(extra_sources=()):
    sources=[Path(__file__).absolute(),Path(domains.__file__).absolute(),Path(references.__file__).absolute(),
        Path(__file__).with_name('retained_math_evidence.py'),*map(Path,extra_sources)]
    return original.implementation_bundle(sources)


def _normalized(request,samples):
    require(type(samples)is list and 1<=len(samples)<=1024,'Finite retained mathematical case count')
    logical=[];physical=[];wire=[]
    # Preserve the original 16-input normalization limit for each chunk. This
    # does not increase any backend process/input limit or execute a model.
    for start in range(0,len(samples),16):
        a,b,c=original._normalized(request,samples[start:start+16]);logical.extend(a);physical.extend(b);wire.extend(c)
    return logical,physical,wire


def _state(request,selection,samples,raw_cases):
    provider=references.Provider(selection);target=selection['target']
    require(target['kind']=='frozen_generation_request'and target['request_id']==request['request_id'],'Retained rows must freeze for this generation request')
    specs=request['model']['inputs'];require(len(specs)==1 and specs[0]['shape']==target['input_shape'],'Explicit retained single-tensor input ABI')
    relation=references._json(provider.files.data[provider.contract['input_relation']['path']])
    provenance=relation['source_graph_weights_and_approximation'];declared=references._json(provider.files.data[provenance['full_ModelSpec']['path']])
    checked=validate(declared);request_check=validate(request['model'])
    require(declared['inputs']==specs,'Declared reference model input signature differs')
    arity=lambda model,check:sum(math.prod(check['shapes'][o['value']])for o in model['outputs'])
    require(arity(declared,checked)==arity(request['model'],request_check)==target['output_values'],'Complete declared/request output signature')
    require(len(request['layout']['output_selectors'])==target['output_values'],'Complete physical output selector arity')
    logical,physical,wire=_normalized(request,samples)
    require(type(raw_cases)is list and len(raw_cases)==len(logical),'Private raw case inputs required')
    bindings=[];seen=set()
    for index,(entry,inputs,packed,named)in enumerate(zip(raw_cases,logical,physical,wire)):
        require(type(entry)is dict and set(entry)=={'case_id','raw_input'}and type(entry['raw_input'])is bytes,'Typed raw input identity')
        key=entry['case_id'];require(type(key)is str and key in provider.rows and key not in seen,'Unique declared retained case');seen.add(key)
        case,receipt=provider.rows[key];raw=entry['raw_input'];declared_bytes=inputs[specs[0]['name']].astype('<f8',copy=False).tobytes(order='C')
        require(len(raw)==math.prod(target['input_shape'])and sha(raw)==receipt['raw_input_sha256'],'Raw input case changed')
        require(sha(declared_bytes)==receipt['declared_input_sha256'],'Logical declared bytes differ from retained case')
        if case['label']is not None:
            require(sha(bytes([case['label']])+raw)==case['input']['record_sha256'],'Original labelled raw record changed')
        bindings.append(dict(index=index,case_id=key,raw_input_sha256=sha(raw),declared_input_sha256=sha(declared_bytes),
            logical_input_sha256=sha(canonical(named)),physical_input_sha256=sha(canonical(packed)),
            original_source_input_sha256=receipt['original_source_input_sha256']))
    require(len({r['physical_input_sha256']for r in bindings})==len(bindings),'Distinct physical evaluation inputs required')
    source_identity=bound(dict(format='poseidon-retained-original-function-identity-v1',model_source=provenance['model_source'],
        original_weights=provenance['original_weights'],input_relation=provider.contract['input_relation'],
        original_input_dtype='<f4',historical_input_tensor_reobserved=False))
    return provider,declared,provenance,source_identity,logical,physical,wire,bindings


def make_policy(request,selection,samples,raw_cases,*,domain,approximation_requirements,application_requirements,
                whole_model_required=False,application_contract=None,required_application_samples=None,domain_contract=None,
                maximum_scalar_operations=2**28,test_only=False,extra_sources=()):
    require(MATH==dict(atol=1e-12,rtol=1e-12)and CKKS==dict(atol=1e-5,rtol=1e-4),'Original numerical thresholds')
    require(type(test_only)is bool and type(whole_model_required)is bool,'Explicit mathematical scope')
    require(type(maximum_scalar_operations)is int and 1<=maximum_scalar_operations<=2**28,'Original independent mathematical work budget')
    provider,declared,provenance,identity,logical,physical,wire,bindings=_state(request,selection,samples,raw_cases)
    require(type(domain)is dict and set(domain)=={s['name']for s in declared['inputs']},'Explicit frozen input domain')
    for interval in domain.values():
        require(type(interval)is list and len(interval)==2 and all(type(x)in(int,float)and math.isfinite(x)for x in interval)and interval[0]<=interval[1],'Finite original input interval')
    if domain_contract is not None:
        domains.verify_contract(domain_contract,declared)
        definition=domain_contract['definition']
        if definition is not None:
            origin=provenance['approximation_definition']
            require(set(definition)<= {'path','sha256','bytes'} and
                    all(definition.get(k)==origin.get(k) for k in ('path','sha256')) and
                    ('bytes' not in definition or definition['bytes']==origin.get('bytes')),
                    'Use the original approximation-definition domain source')
    require(not any(n['op']=='polynomial'for n in declared['nodes'])or domain_contract is not None,'Declared polynomial validity contract required')
    if whole_model_required:
        require([r['case_id']for r in bindings]==selection['case_ids'],'Complete original case set and order required')
        require(domain_contract is not None and bool(application_requirements),'Complete-model domain and application requirements required')
        require(type(required_application_samples)is int and required_application_samples>0,'Frozen full application sample count required')
        parent=references._json(provider.files.data[provider.contract['original_contract']['path']])
        original_images=parent.get('original_full_test',{}).get('images')
        if original_images is not None:
            require(type(original_images)is int and required_application_samples==original_images,'Retained full application scope cannot be reduced')
    require(not application_requirements or type(application_contract)is dict,'Frozen independent application contract required')
    if application_requirements:
        require(type(application_contract.get('samples'))is int and application_contract['samples']>0,'Finite application dataset count')
        if required_application_samples is not None:require(application_contract['samples']==required_application_samples,'Original full application count cannot be shortened')
    else:require(application_contract is None and required_application_samples is None,'Unexpected application scope')
    bundle=implementation_bundle(extra_sources);sources=bundle['sources']
    if application_contract is not None:
        require(type(application_contract.get('producer_sources'))is dict and bool(application_contract['producer_sources'])
            and all(sources.get(p)==h for p,h in application_contract['producer_sources'].items()),'Application producer must freeze before generation')
    frozen_domain=dict(input_intervals=copy.deepcopy(domain),effective_declared_domain=copy.deepcopy(domain_contract))
    spec=bound(dict(format=SPEC,entry_mode='external_assessment_only',reference_kind='retained_source_and_declared_rows',
        request_id=request['request_id'],request_model_sha256=sha(canonical(request['model'])),request_sha256=sha(canonical(request)),
        original_function_identity=identity,original_model_sha256=identity['binding'],declared_reference_model_sha256=sha(canonical(declared)),
        declared_model_file=provenance['full_ModelSpec'],reference_selection=copy.deepcopy(selection),input_bindings=bindings,
        domain=copy.deepcopy(frozen_domain),application_contract=copy.deepcopy(application_contract),required_application_samples=required_application_samples,
        maximum_scalar_operations=maximum_scalar_operations,producer_sources=sources,producer_assets=bundle['assets'],
        python=sys.version,numpy=np.__version__,test_only=test_only,whole_model_required=whole_model_required))
    policy=evaluation_policy([r['physical_input_sha256']for r in bindings],domain_sha256=sha(canonical(frozen_domain)),
        approximation_requirements=approximation_requirements,application_requirements=application_requirements,whole_model_required=whole_model_required,
        independent_math_sources=sources,original_model_sha256=identity['binding'])
    return bound(dict({k:v for k,v in policy.items()if k!='binding'},retained_mathematical_producer=spec))


def check_policy(policy,request,selection,samples,raw_cases):
    checked_bound(policy);spec=checked_bound(policy['retained_mathematical_producer'])
    require(spec['format']==SPEC and spec['reference_kind']=='retained_source_and_declared_rows','Distinct retained mathematical policy')
    require(type(spec['test_only'])is bool and spec['whole_model_required']is policy['whole_model_required'],'Frozen mathematical test/whole-model scope')
    require(spec['request_id']==request['request_id']and spec['request_sha256']==sha(canonical(request))
        and spec['request_model_sha256']==sha(canonical(request['model']))and spec['reference_selection']==selection,'Frozen target/selection changed')
    require(policy['independent_math']==MATH and policy['CKKS']==CKKS and spec['producer_sources']==policy['independent_math_sources']
        and all(sha(read_bytes(p))==h for p,h in spec['producer_sources'].items())
        and all(Path(p).name!='.env'and sha(read_bytes(p))==h for p,h in spec['producer_assets'].items())
        and spec['python']==sys.version and spec['numpy']==np.__version__,'Frozen mathematical implementation/environment changed')
    state=_state(request,selection,samples,raw_cases);_,declared,_,identity,_,_,_,bindings=state
    require(bindings==spec['input_bindings']and [r['physical_input_sha256']for r in bindings]==policy['inputs']
        and identity==spec['original_function_identity']and identity['binding']==policy['original_model_sha256']==spec['original_model_sha256']
        and sha(canonical(declared))==spec['declared_reference_model_sha256']and sha(canonical(spec['domain']))==policy['frozen_domain_sha256'],'Frozen raw/logical/physical/function/domain identities changed')
    return spec,state


def _indices(spec,indices):
    count=len(spec['input_bindings']);indices=list(range(count))if indices is None else indices
    require(type(indices)is list and 1<=len(indices)<=16 and all(type(i)is int and 0<=i<count for i in indices)
        and indices==sorted(set(indices)),'Finite unique original input subset, at most 16 per record')
    return indices


def produce_domain_record(request,selection,samples,raw_cases,policy,output,*,indices=None):
    spec,state=check_policy(policy,request,selection,samples,raw_cases);provider,declared,_,_,logical,_,_,bindings=state
    contract=spec['domain']['effective_declared_domain'];require(contract is not None,'No effective domain contract selected')
    chosen=_indices(spec,indices);rows=[]
    path=Path(output).absolute();require(path.suffix=='.json'and path.parent.resolve()==path.parent and not path.exists(),'Fresh private domain record path')
    write_once(path.with_name(path.name+'.attempt.json'),canonical(bound(dict(format=DOMAIN_RECORD+'-attempt',policy_sha256=sha(canonical(policy)),input_indices=chosen))))
    for i in chosen:
        observation=domains.evaluate(declared,logical[i],contract,maximum_scalar_operations=spec['maximum_scalar_operations'])
        rows.append(dict(bindings[i],domain_observation=observation))
    check_policy(policy,request,selection,samples,raw_cases);provider.verify_sources()
    record=bound(dict(format=DOMAIN_RECORD,mode='declared_function_domain_evaluation',request_id=request['request_id'],policy_sha256=sha(canonical(policy)),
        producer_spec_binding=spec['binding'],domain_contract_binding=contract['binding'],producer_sources=spec['producer_sources'],
        declared_reference_model_sha256=spec['declared_reference_model_sha256'],input_indices=chosen,inputs=rows,
        candidate_python_executed=False,CKKS_outputs_used=False,encrypted_execution=False,runtime_or_refresh_domain_certified=False,test_only=spec['test_only']))
    write_once(path,canonical(record));return dict(path=str(path),sha256=sha(read_bytes(path)))


def _domain_rows(descriptor,policy,spec,indices):
    contract=spec['domain']['effective_declared_domain']
    if contract is None:
        require(descriptor is None,'Unexpected effective-domain report');return {},'not_required'
    if descriptor is None:return {},'missing'
    from retained_math_evidence import verify_domain_record
    report=verify_domain_record(descriptor,policy,spec)
    by_index={r['index']:r for r in report['inputs']};require(all(i in by_index for i in indices),'Effective-domain evidence incomplete for this mathematical subset')
    return by_index,'verified'


def produce_record(request,source,selection,samples,raw_cases,policy,output,*,indices=None,domain_evidence=None,application_evidence=None):
    require(type(source)is bytes,'Untouched original DSL bytes required')
    spec,state=check_policy(policy,request,selection,samples,raw_cases);provider,_,_,_,logical,physical,_,bindings=state
    chosen=_indices(spec,indices);effective,domain_status=_domain_rows(domain_evidence,policy,spec,chosen)
    path=Path(output).absolute();require(path.suffix=='.json'and path.parent.resolve()==path.parent and not path.exists(),'Fresh private mathematical record path')
    write_once(path.with_name(path.name+'.attempt.json'),canonical(bound(dict(format=RECORD+'-attempt',source_sha256=sha(source),policy_sha256=sha(canonical(policy)),input_indices=chosen))))
    if policy['application_requirements']and application_evidence is None:application={};application_status='missing'
    else:
        application=original._application_values(spec['application_contract'],application_evidence,policy,request,source)
        application_status='verified'if policy['application_requirements']else'not_required'
    rows=[];reference_records=[]
    input_name=request['model']['inputs'][0]['name']
    for i in chosen:
        raw=raw_cases[i]['raw_input'];declared_bytes=logical[i][input_name].astype('<f8',copy=False).tobytes(order='C')
        reference=provider.resolve(bindings[i]['case_id'],raw,declared_bytes,target=selection['target'])
        mathematical=dsl_mathematics(source,request,physical[i],maximum_scalar_operations=spec['maximum_scalar_operations'])
        require(len(mathematical['values'])==len(reference['declared_reference'])==len(reference['original_reference']),'Full mathematical output arity')
        inside=all(np.all(x>=spec['domain']['input_intervals'][name][0])and np.all(x<=spec['domain']['input_intervals'][name][1])for name,x in logical[i].items())
        effective_ok=domain_status=='not_required'or domain_status=='verified'and effective[i]['domain_observation']['passed']is True
        rows.append(dict(input_sha256=bindings[i]['physical_input_sha256'],source_sha256=sha(source),case_id=bindings[i]['case_id'],
            raw_input_sha256=bindings[i]['raw_input_sha256'],declared_input_sha256=bindings[i]['declared_input_sha256'],
            original_reference=list(reference['original_reference']),declared_reference=list(reference['declared_reference']),
            dsl_mathematical_output=mathematical['values'],inside_frozen_domain=bool(inside and effective_ok),
            input_domain_passed=bool(inside),effective_domain_status=domain_status,domain_sha256=policy['frozen_domain_sha256'],
            application_metrics=copy.deepcopy(application),application_evidence_status=application_status,whole_model_output=True,
            mathematical_scalar_operations=mathematical['scalar_operations'],retained_reference_record_binding=reference['binding']))
        rows[-1]['original_to_declared_approximation']=compare(reference['original_reference'],reference['declared_reference'],atol=0.,rtol=0.)
        rows[-1]['independent_math_comparison']=compare(reference['declared_reference'],mathematical['values'],**MATH)
        reference_records.append(reference)
    check_policy(policy,request,selection,samples,raw_cases);provider.verify_sources()
    record=bound(dict(format=RECORD,mode='external_mathematical_evaluation',request_id=request['request_id'],source_sha256=sha(source),
        policy_sha256=sha(canonical(policy)),producer_spec_binding=spec['binding'],producer_sources=spec['producer_sources'],
        original_model_sha256=spec['original_model_sha256'],declared_model_sha256=spec['request_model_sha256'],
        declared_reference_model_sha256=spec['declared_reference_model_sha256'],reference_origin='retained_source_and_declared_rows',
        reference_provider_selection_binding=selection['binding'],reference_records=reference_records,input_indices=chosen,inputs=rows,
        domain_evidence=copy.deepcopy(domain_evidence),application_evidence=copy.deepcopy(application_evidence),
        domain_evidence_status=domain_status,application_evidence_status=application_status,candidate_python_executed=False,
        CKKS_outputs_used_as_reference=False,encrypted_execution=False,native_source_worker_report=False,
        first_output_success_certified=False,test_only=spec['test_only']))
    write_once(path,canonical(record));return dict(path=str(path),sha256=sha(read_bytes(path)))
