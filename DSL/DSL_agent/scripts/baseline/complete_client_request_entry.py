"""Frozen public request -> durable admission outcome before any Provider exists.

This optional source-capability entry records blocked requests. It cannot start
generation with a source-only backend descriptor. Compilation and numerical
assessment remain separate external services; old delivery entry points remain.
"""
import copy
from pathlib import Path
import time

import delivery_component as delivery
from delivery_component import bound,canonical,checked_bound,read_bytes,require,sha,write_once
import complete_client_capability as capability

VERSION='poseidon-frozen-complete-client-request-entry-v1'
BLOCKED={'unsupported','resource_blocked','backend_blocked'}


def source_manifest():
    import component_contract,compiler_configuration,native_execution_profile,benchmark_graph,model_resources,first_output_assessment
    modules=[capability,component_contract,compiler_configuration,native_execution_profile,benchmark_graph,model_resources,first_output_assessment]
    paths=[Path(__file__).absolute(),*[Path(module.__file__).absolute()for module in modules]]
    return {str(path):sha(read_bytes(path))for path in paths}


def _source_closure(route):
    """Only frozen public code/metadata inputs; never credential or weight files."""
    result=route['capability']['source_sha256']
    for name,expected in result.items():
        path=Path(name)
        require(path.is_absolute()and path.suffix in('.py','.json','.md','.cpp','.h'),
                'Only explicitly frozen public capability code and metadata')
        require(sha(read_bytes(path))==expected,'Public capability source changed')
    return result


def freeze(mechanism,plan,routes,*,test_only=False):
    """Add a distinct mode to a new mechanism, never re-sign an old delivery."""
    delivery.verify_generator(mechanism);checked_bound(plan)
    require(type(test_only)is bool and type(routes)is list and routes,'Explicit entry role and public routes')
    require('complete_client_request_entry'not in mechanism,'Entry selected once before generation')
    require(sha(canonical(plan))==mechanism['evaluation_policy_sha256'],'Exact original frozen delivery slots')
    sources=source_manifest()
    require(all(mechanism['source_sha256'].get(path)==value for path,value in sources.items()),
            'New entry and capability checker must be in the new source freeze')
    frozen={}
    for value in routes:
        require(type(value)is dict and set(value)=={'request','requirements','capability'},'Closed public route inputs')
        request=delivery.public_request(copy.deepcopy(value['request']));key=request['request_id']
        require(key not in frozen and key in plan['slots']and plan['slots'][key]['request_sha256']==sha(canonical(request)),
                'One exact frozen request and output location')
        require(test_only or value['capability']['test_only']is False,'Fixture cannot authorize production admission')
        expected=dict(request_id=key,requirements_binding=value['requirements']['binding'],capability_binding=value['capability']['binding'])
        _source_closure(value)
        decision=capability.assess(request,value['requirements'],value['capability'],expected,
                                   test_only=value['capability']['test_only'])
        require(decision['status']in BLOCKED and decision['Provider_or_execution_authority']is False,
                'This source-only version cannot construct a Provider or grant native readiness')
        frozen[key]=dict(request_sha256=sha(canonical(request)),requirements=copy.deepcopy(value['requirements']),
            capability=copy.deepcopy(value['capability']),frozen_admission=decision)
    require(set(frozen)==set(plan['slots']),'No frozen request omitted from admission inventory')
    return bound(dict({key:value for key,value in mechanism.items()if key!='binding'},
        complete_client_request_entry=VERSION,complete_client_entry_sources=sources,
        complete_client_entry_test_only=test_only,complete_client_routes=frozen))


def _selection(mechanism,plan):
    delivery.verify_generator(mechanism);checked_bound(plan)
    require(mechanism.get('complete_client_request_entry')==VERSION and
            mechanism.get('complete_client_entry_sources')==source_manifest(),'Frozen entry mode/source changed')
    require(sha(canonical(plan))==mechanism['evaluation_policy_sha256']and
            set(mechanism['complete_client_routes'])==set(plan['slots']),'Original request inventory and plan')


def record_once(request,output,*,mechanism,plan,batch=None,checkpoint=None,test_only=False):
    """Preserve a standard delivery receipt with no source and zero API budget.

No Provider, evaluator, repair or native callback is accepted by this API.
"""
    _selection(mechanism,plan)
    require(type(test_only)is bool and mechanism['complete_client_entry_test_only']is test_only,
            'Frozen development/production entry role')
    request=delivery.public_request(copy.deepcopy(request));key=request['request_id']
    require(key in mechanism['complete_client_routes'],'Request absent from frozen admission inventory')
    route=mechanism['complete_client_routes'][key];slot=plan['slots'][key]
    require(route['request_sha256']==slot['request_sha256']==sha(canonical(request))and
            str(Path(output).absolute())==slot['delivery_path'],'No alternate request or second output location')
    require(slot['split']!='acceptance'or plan['independent_seal_verified'],'Unsealed independent acceptance rejected')
    _source_closure(route);decision=checked_bound(route['frozen_admission'])
    require(decision['request_id']==key and decision['requirements_binding']==route['requirements']['binding']and
            decision['capability_binding']==route['capability']['binding']and decision['status']in BLOCKED and
            decision['Provider_or_execution_authority']is False,'Frozen source-only admission cannot become delivery success')
    current=capability.assess(request,route['requirements'],route['capability'],dict(request_id=key,
        requirements_binding=route['requirements']['binding'],capability_binding=route['capability']['binding']),
        test_only=route['capability']['test_only'])
    require(current==decision,'Rebound or stale admission classification rejected before any Provider')
    if not test_only:
        from shared_execution import Batch
        require(isinstance(batch,Batch)and callable(checkpoint),'Original accounted host batch required')
        row=batch.record();target=Path(output).absolute()
        require(row['plan']['source_binding']==mechanism['binding']and
                all(value==0 for value in row['plan']['stages'].values())and
                any(target!=Path(item['path'])and target.is_relative_to(item['path'])for item in row['artifacts']),
                'Admission-only zero-API batch with original source/accounting identity')
        checkpoint()
    folder=delivery._folder(output);started=time.time_ns()
    for name,value in [('request.json',request),('generator.json',mechanism),('delivery-plan.json',plan),
            ('capability.json',route['capability']),('requirements.json',route['requirements']),('admission.json',decision)]:
        write_once(folder/name,canonical(value))
    attempt=bound(dict(format=VERSION,request_id=key,mechanism_binding=mechanism['binding'],started_at_ns=started,
        test_only=test_only,Provider_constructions=0,generations=0,HTTP_attempts=0,compilation=0,
        encrypted_execution=0,post_output_feedback=0,repairs=0,retries=0))
    write_once(folder/'attempt.json',canonical(attempt))
    receipt=bound(dict(format=delivery.VERSION,mode='development_admission_probe'if test_only else'delivery_generation',
        origin='pre_generation_admission',request_id=key,mechanism_binding=mechanism['binding'],first_generation_index=None,
        post_output_repairs=0,real_Agent=False,external_evaluation_performed=False,
        request_sha256=sha(read_bytes(folder/'request.json')),generator_sha256=sha(read_bytes(folder/'generator.json')),
        delivery_plan_sha256=sha(read_bytes(folder/'delivery-plan.json')),response_sha256=None,source_sha256=None,source_bytes=0,
        status=decision['status'],first_output_eligible=False,provider_metrics=None,
        complete_client_entry=VERSION,capability_sha256=sha(read_bytes(folder/'capability.json')),
        requirements_sha256=sha(read_bytes(folder/'requirements.json')),admission_sha256=sha(read_bytes(folder/'admission.json')),
        attempt_sha256=sha(read_bytes(folder/'attempt.json')),test_only=test_only))
    write_once(folder/'delivery.json',canonical(receipt))
    verify_record(folder,receipt['binding'])
    if not test_only:checkpoint()
    return receipt


def verify_record(output,expected_binding):
    """Saved-byte consistency only; no post-output backend/host reconstruction."""
    receipt,request,source,mechanism=delivery.verify_delivery(output,expected_binding)
    require(receipt['complete_client_entry']==mechanism['complete_client_request_entry']==VERSION and source is None,
            'Admission-only entry cannot contain a replacement DSL')
    folder=Path(output).absolute();route=mechanism['complete_client_routes'][request['request_id']]
    import json
    values={}
    for name,key in [('capability','capability_sha256'),('requirements','requirements_sha256'),('admission','admission_sha256'),('attempt','attempt_sha256')]:
        raw=read_bytes(folder/(name+'.json'));require(sha(raw)==receipt[key],'Frozen admission artifact changed')
        values[name]=checked_bound(json.loads(raw))
    require(values['capability']==route['capability']and values['requirements']==route['requirements']and
            values['admission']==route['frozen_admission'],'Saved exact pre-generation route and decision')
    attempt=values['attempt'];decision=values['admission']
    require(attempt['format']==VERSION and attempt['request_id']==receipt['request_id']and
            attempt['mechanism_binding']==mechanism['binding']and
            all(attempt[key]==0 for key in ['Provider_constructions','generations','HTTP_attempts','compilation',
                'encrypted_execution','post_output_feedback','repairs','retries']),'No Provider, compiler, numerical work or repair inside admission')
    require(receipt['status']==decision['status']and decision['status']in BLOCKED and
            receipt['first_output_eligible']is False and receipt['real_Agent']is False and
            receipt['post_output_repairs']==0 and receipt['first_generation_index']is None and
            receipt['source_sha256']is receipt['response_sha256']is None and
            not(folder/'response.txt').exists()and
            receipt['test_only']==attempt['test_only']==mechanism['complete_client_entry_test_only'],
            'Blocked request cannot be promoted to original Agent output')
    return receipt,request,mechanism


def assess_blocked_once(output,expected_binding,plan,assessment_output):
    """External standard assessment; source-less records never call an evaluator."""
    receipt,request,mechanism=verify_record(output,expected_binding)
    from first_output_assessment import assess_once
    result=assess_once(output,expected_binding,plan,assessment_output,None)
    verify_record(output,expected_binding)
    require(result['status']==receipt['status']and result['input_groups_evaluated']==0 and
            all(result[key]is False for key in ['raw_compile_passed','raw_numeric_passed','raw_joint_passed',
                'first_compile_passed','first_numeric_passed','first_joint_passed']),
            'Standard external assessment must retain the blocked outcome and zero numerical claims')
    return result
