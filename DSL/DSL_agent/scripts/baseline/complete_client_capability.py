"""Optional public route admission and separate external program-data audit.

No Provider, compiler, native process or numerical evaluation is invoked here.
Interface consistency cannot grant backend qualification or execution authority.
"""
import copy
import math
import re

from delivery_component import bound, canonical, checked_bound, require, sha

FORMAT='poseidon-complete-client-source-capability-v1'
REQUIREMENTS='poseidon-complete-client-public-route-requirements-v1'
ADMISSION='poseidon-complete-client-pre-generation-admission-v1'
MAX_BYTES=1024**2


def _bounded(value):
    require(len(canonical(value))<=MAX_BYTES,'Complete client metadata1MiB bound')


def _integer(value,low,high):
    return type(value)is int and low<=value<=high


def _capability(value,expected,test_only):
    checked_bound(value,expected); _bounded(value)
    require(value['format']==FORMAT and type(test_only)is bool and value['test_only']is test_only,
            'Exact externally pinned capability role; fixture is never real backend evidence')
    require(value['source_only']is True and value['execution_qualified']is False,
            'This source-consistency interface cannot mint an execution certificate')
    context=value['context']; profile=value['compiler_configuration']; limits=value['limits']
    require(context['scheme']=='CKKS' and context['security']=='tc128' and
            _integer(context['degree'],2,2**20) and context['degree']&(context['degree']-1)==0 and
            context['slots']==context['degree']//2,'Original CKKS context and security')
    for key in ['q','p']:
        require(type(context[key])is list and 1<=len(context[key])<=64 and
                all(_integer(n,3,2**63-1)for n in context[key]),'Exact public modulus values, not only bit counts')
    require(len(set(context['q']+context['p']))==len(context['q'])+len(context['p']),
            'Distinct original q/p moduli')
    require(context['compiler_profile_sha256']==profile['ckks_config_sha256'] and
            context['parameter_identity']==profile['parameter_identity'],'One declared compiler/context identity')
    require(type(value['rotations'])is list and len(set(value['rotations']))==len(value['rotations']) and
            all(_integer(step,-context['slots']+1,context['slots']-1) and step!=0 for step in value['rotations']) and
            type(value['conjugation'])is bool,'Explicit supported key steps; conjugation is a separate capability')
    require(_integer(limits['units'],1,16) and _integer(limits['physical_inputs'],1,16) and
            _integer(limits['physical_outputs'],1,16) and _integer(limits['instructions'],1,4096) and
            _integer(limits['values'],1,8192) and _integer(limits['work_memory_bytes'],1,4*1024**3),
            'Original final-client aggregate bounds and unchanged native memory limit')
    require(value['output_ABI']==dict(slots=context['slots'],bytes_per_physical_final=context['slots']*16,
            real_imag_interleaved_binary64=True),'All physical slots, both complex components')
    require(type(value['source_sha256'])is dict and value['source_sha256'] and
            all(type(k)is str and re.fullmatch('[0-9a-f]{64}',v) for k,v in value['source_sha256'].items()),
            'Explicit public capability source identities')
    require(all(type(value[k])is int for k in ['level_min','level_max','scale_log2_min','scale_log2_max']) and
            0<=value['level_min']<=value['level_max']<len(context['q']) and
            0<=value['scale_log2_min']<=value['scale_log2_max']<=1023,'Finite structural descriptor bounds')
    return value


def freeze_requirements(request,*,rotations,input_level,input_scale_log2,maximum_units,
                        maximum_instructions,maximum_values,conjugation=False,required_refreshes=0,
                        maximum_work_memory_bytes=4*1024**3):
    """Trusted planner claims from public constraints, frozen before generation.

Actual compiler operations must be checked separately after immutable DSL output.
"""
    from component_contract import reconstruct_request
    reconstruct_request(copy.deepcopy(request))
    require(type(rotations)is list and len(set(rotations))==len(rotations) and
            all(type(n)is int and n!=0 for n in rotations) and type(conjugation)is bool,
            'Public planned rotations and separate conjugation')
    require(all(type(n)is int for n in [input_level,input_scale_log2,maximum_units,
            maximum_instructions,maximum_values,required_refreshes,maximum_work_memory_bytes]),
            'Integer public planner resource/state claims')
    return bound(dict(format=REQUIREMENTS,request_id=request['request_id'],
        request_sha256=sha(canonical(request)),compiler_configuration_sha256=sha(canonical(request['compiler_configuration'])),
        compiler_profile_sha256=request['compiler_profile_sha256'],rotations=copy.deepcopy(rotations),conjugation=conjugation,
        input_level=input_level,input_scale_log2=input_scale_log2,maximum_units=maximum_units,
        maximum_instructions=maximum_instructions,maximum_values=maximum_values,required_refreshes=required_refreshes,
        maximum_work_memory_bytes=maximum_work_memory_bytes,
        evidence_stage='public_planner_before_generation',compiler_or_numeric_feedback_used=False))


def assess(request,requirements,capability,expected,*,test_only=False):
    """Return unsupported or backend_blocked without invoking the Provider."""
    require(type(expected)is dict and set(expected)=={'request_id','requirements_binding','capability_binding'},
            'Independently frozen request/requirements/capability')
    from component_contract import reconstruct_request
    reconstruct_request(copy.deepcopy(request)); _bounded(requirements)
    checked_bound(requirements,expected['requirements_binding'])
    cap=_capability(capability,expected['capability_binding'],test_only)
    require(requirements['format']==REQUIREMENTS and requirements['request_id']==request['request_id']==expected['request_id'] and
            requirements['request_sha256']==sha(canonical(request)) and
            requirements['compiler_configuration_sha256']==sha(canonical(request['compiler_configuration'])) and
            requirements['compiler_profile_sha256']==request['compiler_profile_sha256'] and
            requirements['evidence_stage']=='public_planner_before_generation' and
            requirements['compiler_or_numeric_feedback_used']is False,'Frozen public request and pre-generation planner claims')
    require(type(requirements['rotations'])is list and len(set(requirements['rotations']))==len(requirements['rotations']) and
            all(type(n)is int and n!=0 for n in requirements['rotations']) and type(requirements['conjugation'])is bool,
            'Typed planned key steps')
    require(type(requirements['input_level'])is int and type(requirements['input_scale_log2'])is int,
            'Typed original input level and scale')
    require(_integer(requirements['required_refreshes'],0,4096) and type(cap['bootstrap_available'])is bool,
            'Frozen public refresh demand and separately certified capability')
    reasons=[]; layout=request['layout']; context=cap['context']; limits=cap['limits']
    if request['compiler_configuration']!=cap['compiler_configuration']:
        reasons.append('exact_compiler_configuration_or_profile_mismatch')
    if request['compiler_profile_sha256']!=context['compiler_profile_sha256']:
        reasons.append('exact_compiler_profile_bytes_mismatch')
    if not set(requirements['rotations'])<=set(cap['rotations']) or requirements['conjugation'] and not cap['conjugation']:
        reasons.append('required_rotation_or_conjugation_not_available')
    if requirements['input_level']!=cap['input_state']['level'] or requirements['input_scale_log2']!=cap['input_state']['scale_log2']:
        reasons.append('original_input_level_or_scale_mismatch')
    if requirements['required_refreshes'] and not cap['bootstrap_available']:
        reasons.append('required_bootstrap_capability_not_certified')
    for name,limit in [('maximum_units','units'),('maximum_instructions','instructions'),('maximum_values','values'),('maximum_work_memory_bytes','work_memory_bytes')]:
        if not _integer(requirements[name],1,limits[limit]): reasons.append(name+'_outside_capability')
    period=layout['input_slot_period']
    if not _integer(period,1,context['slots']) or context['slots']%period:
        reasons.append('declared_slot_period_outside_capability')
    if not 1<=len(layout['inputs'])+len(layout.get('auxiliary_ciphertexts',[]))<=limits['physical_inputs']:
        reasons.append('physical_input_count_outside_capability')
    if not _integer(layout['output_ciphertexts'],1,limits['physical_outputs']):
        reasons.append('physical_output_count_outside_capability')
    compatible=not reasons
    resource_reasons={name+'_outside_capability'for name in ['maximum_units','maximum_instructions','maximum_values','maximum_work_memory_bytes','physical_input_count','physical_output_count']}
    status='backend_blocked'if compatible or reasons==['required_bootstrap_capability_not_certified']else (
        'resource_blocked'if set(reasons)<=resource_reasons else'unsupported')
    return bound(dict(format=ADMISSION,stage='before_Provider_before_DSL_output',request_id=request['request_id'],
        requirements_binding=requirements['binding'],capability_binding=cap['binding'],test_only=test_only,
        status=status,interface_compatible=compatible,
        reasons=['source_only_capability_not_complete_native_backend_qualification']if compatible else reasons,
        public_planner_claims_are_not_actual_compiler_operation_evidence=True,complete_logical_aliases_verified=False,
        Provider_calls=0,compiler_calls=0,native_calls=0,numeric_calls=0,Provider_or_execution_authority=False,
        first_or_full_model_or_SDK_or_Bootstrap_qualification=False))


def audit_external(request,requirements,capability,expected,compiler_inputs,runtime_inputs,*,test_only=False):
    """External post-output data check; never feeds a repair back to delivery."""
    pre=assess(request,requirements,capability,expected,test_only=test_only)
    require(pre['interface_compatible'],'Reject incompatible profile before constructing program targets')
    import derived_compiler_program_adapter as adapter
    require(compiler_inputs['request']==request,'Same frozen request for actual external compiler audit')
    compiled,partition=adapter.derive(**compiler_inputs)
    require(compiled==runtime_inputs['compiled'],'Original whole compiler derivation, not a rebound substitute')
    for row in compiled['physical_inputs']:
        descriptor=row['descriptor']
        require(descriptor['level']==requirements['input_level'] and descriptor['scale_log2']==requirements['input_scale_log2'] and
                descriptor['context']==capability['runtime_context'],'Every original physical input matches frozen public state')
    runtime=adapter.runtime_origins(**runtime_inputs)
    plans=runtime_inputs['plan_bytes']; cap=capability; instructions=values=0; observed_steps=set()
    transport=adapter.original_program.transport()
    for unit in compiled['units']:
        plan=transport.decode(plans[unit['plan_sha256']],unit['plan_sha256']); values+=len(plan['values'])
        for row in plan['values']:
            require(row['context']==cap['runtime_context'] and row['ntt']is True and row['place']==dict(kind='host',rank=0),
                    'Exact original context, NTT and CPU place; no retagging')
            require(_integer(row['level'],cap['level_min'],cap['level_max']) and
                    type(row['scale_log2'])in(int,float) and math.isfinite(row['scale_log2']) and
                    cap['scale_log2_min']<=row['scale_log2']<=cap['scale_log2_max'],
                    'Every original unit value level/scale is structurally representable')
        for phase in ['initialization','execution','finalization']:
            instructions+=len(plan[phase])
            for operation in plan[phase]:
                if operation.get('op')=='rotate':
                    step=operation.get('attrs',{}).get('steps')
                    require(type(step)is int and step in cap['rotations'] and step in requirements['rotations'],
                            'Actual compiled rotation must be in both frozen public plan and backend keys')
                    observed_steps.add(step)
    require(len(compiled['units'])<=requirements['maximum_units'] and
            instructions<=requirements['maximum_instructions'] and values<=requirements['maximum_values'],
            'Actual aggregate units/instructions/values within frozen planner claims')
    fragment=adapter.compact_target(compiled,runtime,compiled['binding'],runtime['binding'])
    bridge,decoder=adapter.pinned_B_modules()
    rows=bridge.output_alignment(request['model'],fragment['target_outputs'],dict(outputs=fragment['decoder_outputs']),
        runtime_inputs['selection'],fragment['logical_IO'],fragment['compiler_program_map'],expected_period=compiled['input_slot_period'])
    files={row['decoded_file']for row in runtime_inputs['selection']['outputs']}
    logical,periodic=decoder.validate_outputs(fragment['decoder_outputs'],files,compiled['input_slot_period'])
    require((logical,periodic)==(fragment['logical_values'],fragment['periodic_alias_values']),
            'All logical cells and periodic aliases retained; physical finals stay unique')
    return bound(dict(format='poseidon-complete-client-external-program-data-audit-v1',stage='external_after_frozen_DSL_output',
        request_id=request['request_id'],source_sha256=sha(compiler_inputs['source']),pre_generation_admission_binding=pre['binding'],
        original_whole_plan_sha256=compiled['whole_plan_sha256'],derived_compiled_binding=compiled['binding'],
        runtime_origin_binding=runtime['binding'],compact_target_binding=fragment['binding'],
        actual_unit_plans_checked=len(compiled['units']),physical_finals=len(files),logical_values=logical,periodic_alias_values=periodic,
        complete_real_imag_components=sum(row['complex_components_with_periodic_aliases']for row in rows),
        observed_actual_rotations=sorted(observed_steps),instructions=instructions,values=values,
        original_source_or_unit_plan_modified=False,feedback_or_output_repair=False,test_only=test_only,
        compiler_host_input_samekey_terminal_math_CKKS_or_first_qualification=False,execution_authority=False))
