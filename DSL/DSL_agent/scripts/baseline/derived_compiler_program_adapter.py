"""Opt-in compiler derivation -> runtime origins -> compact complete selectors.

Only metadata is produced/checked. Native program, key/host evidence, numerical
results and first-output eligibility remain external. No Provider is imported.
"""
import copy
import importlib.util
from pathlib import Path

from candidate_contract import strict_json
from component_contract import reconstruct_request
from delivery_component import bound,canonical,checked_bound,read_bytes,require,sha
import compiler_partition_derivation as partition
import compiler_target_coordinates as coordinates
import compiler_program_target as original_program

POLICY='poseidon-frozen-generic-compiler-partition-policy-v1'
FORMAT='poseidon-whole-compiler-derived-program-v1'
RUNTIME='poseidon-derived-program-runtime-origin-audit-v1'
TARGET='poseidon-derived-program-compact-target-fragment-v1'
COMPOSED='poseidon-compiler-program-qualified-final-correspondence-v1'
BRIDGE=Path('/home/lhohy/Code/Poseidon/DSL/DSL_agent/validation/backend_certification/program_complex_target_bridge_v3.py')
BRIDGE_SHA='0c22f3523f94d6228b306ec05b8329af0e6d38a4d78324f6cee99c250155b8fe'
DECODER=BRIDGE.with_name('program_complex_decoder_v2.py')
DECODER_SHA='103c57624f43b3fa5514820eeb0175412c3b09185f54e9adb8a043faff60fba5'


def _bounded(value):require(len(canonical(value))<=1024**2,'Original1MiB native program metadata limit')


def source_manifest():
    require(sha(read_bytes(BRIDGE))==BRIDGE_SHA and sha(read_bytes(DECODER))==DECODER_SHA,'Audited B compact interfaces required before freezing compiler policy')
    import candidate_contract
    import component_contract
    import delivery_component
    import benchmark_graph
    paths=[Path(__file__).absolute(),*[Path(m.__file__).absolute()for m in (partition,coordinates,original_program)],
        *[Path(m.__file__).absolute()for m in (candidate_contract,component_contract,delivery_component,benchmark_graph)],
        original_program.TRANSPORT,BRIDGE,DECODER]
    return {str(p):sha(read_bytes(p))for p in paths}


def freeze_policy(maximum_execution_operations):
    require(type(maximum_execution_operations)is int and 1<=maximum_execution_operations<=4096,'Frozen source-only compiler chunk rule')
    return bound(dict(format=POLICY,selection='fixed_complete_instruction_spans_v1',
        maximum_execution_operations=maximum_execution_operations,source_sha256=source_manifest(),
        source_only_no_compiler_error_or_numerical_feedback=True,limits=dict(units=16,instructions=4096,values=8192,
            physical_inputs=16,physical_outputs=16,metadata_bytes=1024**2),
        invalid_plaintext_cuts_or_resource_limits_reject_without_repair=True,
        mechanism_must_freeze_this_policy_before_first_generation=True))


def derive(request,source,receipt,whole_raw,policy,expected):
    """Compiler stage only: one actual whole proof, zero fake unit compilations."""
    require(type(expected)is dict and set(expected)=={'receipt_binding','request_id','source_sha256','whole_plan_sha256','policy_binding'},'Original compiler pipeline freeze')
    checked_bound(policy,expected['policy_binding'])
    require(policy==freeze_policy(policy['maximum_execution_operations']),'Original compiler rule/source/version changed')
    require(type(source)is bytes and sha(source)==expected['source_sha256']and request['request_id']==expected['request_id'], 'Original untouched whole DSL/request')
    reconstruct_request(copy.deepcopy(request))
    source_expected={k:v for k,v in expected.items()if k!='policy_binding'}
    whole,values,initial,producer,uses=partition._source(receipt,whole_raw,source_expected)
    coordinate=coordinates.make(receipt,expected['receipt_binding'],request_id=request['request_id'],source_sha256=sha(source))
    maximum=policy['maximum_execution_operations'];cuts=list(range(maximum,len(whole['execution']),maximum))
    artifact=partition.partition(receipt,whole_raw,cuts,source_expected)
    audit=partition.audit(receipt,whole_raw,artifact,source_expected,artifact['binding'])
    origins={};reserved=set()
    def assign(value,logical):
        key=(logical['value'],logical['chunk'])
        require(value not in origins or origins[value]==logical,'Whole compiler input/final origin mismatch')
        require(key not in reserved or value in origins and origins[value]==logical,'Distinct whole SSA values cannot share logical origins')
        origins[value]=logical;reserved.add(key)
    period=coordinate['input_slot_period']
    for row in coordinate['physical_inputs']:
        role=row['role']
        logical=dict(value=role['dsl_name'],chunk=0)if role.get('kind')=='encrypted_zero'else dict(value=role['name'],chunk=role.get('offset',0)//period)
        assign(row['value_id'],logical)
    for row in coordinate['logical_outputs']:
        for index,segment in enumerate(row['segments']):assign(segment['value_id'],dict(value=row['model_value'],chunk=index))
    for value in values:
        if value in origins:continue
        stem='__compiled_value_'+value;name=stem;index=0
        while(name,0)in reserved:index+=1;name=stem+'_'+str(index)
        assign(value,dict(value=name,chunk=0))
    units=[];produced={}
    for unit in artifact['units']:
        child=unit['plan'];mapping=unit['local_to_whole'];uid='u%03d'%unit['unit_index']
        def rows(side):
            return [dict(local_value_id=local,whole_value_id=mapping[local],logical=copy.deepcopy(origins[mapping[local]]))for local in child[side]]
        incoming,outgoing=rows('external_inputs'),rows('final_outputs')
        for row in outgoing:
            require(row['whole_value_id']not in produced,'One original producing partition per whole SSA value')
            produced[row['whole_value_id']]=dict(unit_id=uid,local_value_id=row['local_value_id'])
        units.append(bound(dict(format='poseidon-whole-compiler-derived-unit-origins-v1',unit_id=uid,unit_index=unit['unit_index'],
            plan_sha256=unit['plan_sha256'],partition_unit_binding=unit['binding'],whole_compiler_receipt_binding=receipt['binding'],
            inputs=incoming,outputs=outgoing,actually_compiled_as_separate_unit=False)))
    finals=[]
    for ct,row in enumerate(coordinate['physical_outputs']):
        value=row['value_id']
        require(value in produced,'Original native program requires a source-produced final, not a new identity operation')
        finals.append(dict(ciphertext=ct,whole_value_id=value,logical=copy.deepcopy(origins[value]),
            **produced[value],type=copy.deepcopy(row['type'])))
    result=bound(dict(format=FORMAT,request_id=request['request_id'],source_sha256=sha(source),
        source_model_sha256=sha(canonical(request['model'])),layout_sha256=sha(canonical(request['layout'])),
        whole_plan_sha256=sha(whole_raw),whole_compiler_receipt_binding=receipt['binding'],
        compiler_policy=copy.deepcopy(policy),partition_binding=artifact['binding'],partition_audit=audit,
        whole_coordinates=coordinate,units=units,finals=finals,input_slot_period=period,
        physical_inputs=[dict(whole_value_id=row['value_id'],logical=copy.deepcopy(origins[row['value_id']]),
            role='encrypted_zero'if row['role'].get('kind')=='encrypted_zero'else'data',descriptor=copy.deepcopy(values[row['value_id']]))
            for row in coordinate['physical_inputs']],
        source_plan_data_derived_without_numeric_feedback=True,separate_unit_real_compilation_receipts_created=False,
        native_program_key_or_host_or_mathematics_or_first_delivery_eligibility_created=False))
    _bounded(result)
    return result,artifact


def _state(value):return {k:v for k,v in value.items()if k!='id'}


def runtime_origins(compiled,program,source_container_raw,selection,plan_bytes,expected):
    require(type(expected)is dict and set(expected)=={'compiled_binding','program_binding','source_container_binding',
        'source_container_sha256','selection_binding'},'Independently frozen actual runtime-origin selection')
    checked_bound(compiled,expected['compiled_binding']);checked_bound(program,expected['program_binding'])
    checked_bound(selection,expected['selection_binding']);original=original_program.transport()
    container=original.decode(source_container_raw,expected['source_container_sha256']);checked_bound(container,expected['source_container_binding'])
    require(compiled['format']==FORMAT and compiled['separate_unit_real_compilation_receipts_created']is False,'Explicit derived compiler program')
    checked_bound(compiled['partition_audit']);checked_bound(compiled['whole_coordinates'])
    require(compiled['native_program_key_or_host_or_mathematics_or_first_delivery_eligibility_created']is False
        and compiled['source_plan_data_derived_without_numeric_feedback']is True,'Compiler metadata does not grant fake native or first-output authority')
    require(container['format']=='poseidon-bound-tensor-multiunit-source-container-v1'and container['execution_authority']is False
        and container['origin_role']=='exact_partition_of_one_whole_compiler', 'Original source container with explicit compiler derivation role')
    require(container['whole_compiler_receipt_binding']==compiled['whole_compiler_receipt_binding']
        and container['partition_audit_binding']==compiled['partition_audit']['binding'],'Original whole compiler/partition source provenance')
    require(program['binding_stage']=='physical_bound'and program['source']==dict(container_sha256=sha(source_container_raw),
        origins_sha256=sha(source_container_raw),model_sha256=compiled['source_model_sha256'],layout_sha256=compiled['layout_sha256']),
        'Original source/model/layout/physical program identity')
    require(container['source_model_sha256']==compiled['source_model_sha256']and container['parent_layout_sha256']==compiled['layout_sha256']
        and container['scope']==program['scope']and container['key_epoch']==program['key']['epoch']and container['key_origin']==program['key']['origin'],
        'Original program source scope and declared same key')
    for value in (program,selection,container):_bounded(value)
    require(set(plan_bytes)=={u['plan_sha256']for u in compiled['units']}
        and all(type(raw)is bytes and len(raw)<=1024**2 for raw in plan_bytes.values()),'All original bounded derived unit plans')
    require(len(program['units'])==len(container['units'])==len(compiled['units']),'No omitted source-derived units')
    for unit,source,origin in zip(program['units'],container['units'],compiled['units']):
        checked_bound(source);checked_bound(origin)
        require(unit['unit_id']==origin['unit_id']and source['unit_index']==origin['unit_index']
            and unit['raw_plan_sha256']==unit['plan_sha256']==source['source_plan_sha256']==source['derived_plan_sha256']==origin['plan_sha256'],
            'Only the exact derived compiler bytes; no extra boundary rescale or normalization')
        require(unit['source_origin_map_sha256']==sha(canonical(source))and source['origin_role']=='exact_partition_of_one_whole_compiler'
            and source['actually_compiled_as_separate_unit']is False and source['whole_compiler_receipt_binding']==compiled['whole_compiler_receipt_binding']
            and source['partition_unit_binding']==origin['partition_unit_binding'],'No invented separate-unit compilation or origin')
        for side in ('inputs','outputs'):
            require(unit[side]==[dict(local_value_id=r['local_value_id'],logical=r['logical'])for r in origin[side]]
                and source[side]==[dict(local_value_id=r['local_value_id'],whole_value_id=r['whole_value_id'],logical_value=r['logical']['value'],
                    chunk_index=r['logical']['chunk'])for r in origin[side]],'Complete original local/whole/logical SSA boundary correspondence')
    checked=original.validate_program(program,program['binding'],plan_bytes)
    require(checked['instruction_count']==compiled['partition_audit']['derived_instructions']
        and checked['value_count']==compiled['partition_audit']['derived_values'],'Original aggregate derived inventory')
    require(len(program['inputs'])==len(compiled['physical_inputs'])==len(container['inputs']),'All source inputs including trusted zero')
    for runtime,declared,source in zip(program['inputs'],compiled['physical_inputs'],container['inputs']):
        require(runtime['logical']==declared['logical']and runtime['role']==declared['role']
            and canonical(_state(runtime['descriptor']))==canonical(_state(declared['descriptor']))
            and dict(value=source['logical_value'],chunk=source['chunk_index'])==declared['logical'],
            'Original external input roles and nominal state without retagging')
    require(program['final_outputs']==[r['logical']for r in compiled['finals']]
        and program['final_outputs']==[dict(value=r['logical_value'],chunk=r['chunk_index'])for r in container['outputs']],
        'Every original logical final, in physical compiler order')
    require(selection['format']=='poseidon-program-final-selection-v1'and selection['program_binding']==program['binding']
        and selection['key']==program['key']and len(selection['outputs'])==len(compiled['finals']), 'Original final selection and declared same key')
    original.hex_value(selection['case_binding']);original.hex_value(selection['input_bank_binding'])
    require(len(selection['terminals'])==len(program['units']), 'All original unit terminals')
    for terminal,unit in zip(selection['terminals'],program['units']):
        checked_bound(terminal)
        require(terminal['unit_id']==unit['unit_id']and terminal['program_binding']==program['binding']
            and terminal['case_binding']==selection['case_binding']and terminal['input_bank_binding']==selection['input_bank_binding']
            and terminal['unit_plan_sha256']==unit['plan_sha256'],'Same original unit/program/case/input-bank terminal metadata')
    returned=[];names=set();unit_by_id={u['unit_id']:u for u in program['units']}
    for final,selected in zip(compiled['finals'],selection['outputs']):
        uid,local=final['unit_id'],final['local_value_id'];tag=original.logical(final['logical'])
        require(checked['namespace'].get(tag)==(uid,unit_by_id[uid]['plan_sha256'],local)
            and selected['logical']==final['logical']and selected['unit_id']==uid and selected['local_value_id']==local
            and selected['unit_plan_sha256']==unit_by_id[uid]['plan_sha256'],'Exact source-derived final producer and local ValueId')
        filename=selected['decoded_file'];require(type(filename)is str and filename and Path(filename).name==filename and filename not in names,'Original unique final byte files')
        names.add(filename);returned.append(dict(ssa='%'+uid+'.v'+local,value_id=uid+':'+local,type=final['type'],unit_id=uid,
            local_value_id=local,whole_value_id=final['whole_value_id'],decoded_file=filename))
    return bound(dict(format=RUNTIME,compiled_binding=compiled['binding'],program_binding=program['binding'],
        source_container_binding=container['binding'],final_selection_binding=selection['binding'],returns=returned,
        input_slot_period=compiled['input_slot_period'],value_count=checked['value_count'],instruction_count=checked['instruction_count'],
        physical_identity=dict(program_binding=program['binding'],case_binding=selection['case_binding'],input_bank_binding=selection['input_bank_binding'],key=program['key']),
        actual_encryption_host_key_or_terminal_resource_witnesses_verified=False,numerical_values_decoded=0,
        mathematical_or_CKKS_or_first_delivery_eligibility_granted=False))


def compact_target(compiled,runtime,expected_compiled,expected_runtime):
    checked_bound(compiled,expected_compiled);checked_bound(runtime,expected_runtime)
    require(compiled['format']==FORMAT and runtime['format']==RUNTIME and runtime['compiled_binding']==compiled['binding']
        and runtime['input_slot_period']==compiled['input_slot_period'],'Original compact target program/period')
    require(runtime['actual_encryption_host_key_or_terminal_resource_witnesses_verified']is False
        and runtime['mathematical_or_CKKS_or_first_delivery_eligibility_granted']is False and runtime['numerical_values_decoded']==0,
        'Target mapping is not a forged numerical/host certificate')
    returned=[{k:v for k,v in row.items()if k not in ('decoded_file','whole_value_id')}for row in runtime['returns']]
    composed=bound(dict(format=COMPOSED,source_kind='one_whole_compiler_exact_derived_partitions',program_binding=runtime['program_binding'],
        one_compiler_global_DAG_or_whole_source_decomposition_claimed=False,
        whole_compiler_receipt_binding=compiled['whole_compiler_receipt_binding'],partition_audit_binding=compiled['partition_audit']['binding'],
        unit_maps=[dict(unit_id=u['unit_id'],whole_compiler_receipt_binding=u['whole_compiler_receipt_binding'],
            partition_unit_binding=u['partition_unit_binding'],actually_compiled_as_separate_unit=False)for u in compiled['units']],
        returns=returned,value_count=runtime['value_count'],instruction_count=runtime['instruction_count']))
    coordinate=compiled['whole_coordinates'];period=compiled['input_slot_period'];outputs=[];targets=[];decoder=[];owners={}
    for row in coordinate['logical_outputs']:
        segments=[];native=[];name,value=row['name'],row['model_value'];owner=owners.setdefault(value,name)
        require(row['alias_of']==(None if owner==name else owner),'Original exact logical alias owner')
        for original in row['segments']:
            ct=original['ciphertext'];physical=runtime['returns'][ct]
            require(physical['whole_value_id']==original['value_id'],'Original whole compiler return identity')
            segments.append(dict(original,value_id=physical['value_id']))
            native.append(dict(file=physical['decoded_file'],logical_start=original['logical_start'],count=original['elements'],
                slot_start=original['slot_start'],periodic=dict(period=period,stride=period,count=16384//period)))
        outputs.append(dict(name=name,model_value=value,shape=row['shape'],alias_of=row['alias_of'],segments=segments))
        targets.append(dict(request_name=name,model_value=value,runtime_value=value,decoder_name=name,shape=row['shape'],physical_owner=owner))
        decoder.append(dict(name=name,shape=row['shape'],segments=native))
    io=bound(dict(format='poseidon-generic-compiler-logical-I-O-map-v1',origin='single_whole_compiler_exact_derived_partitions',
        request_id=compiled['request_id'],source_sha256=compiled['source_sha256'],compiler_DAG_binding=composed['binding'],outputs=outputs,
        logical_values=coordinate['logical_values'],unique_physical_cells=coordinate['unique_physical_cells'],alias_cells=coordinate['alias_cells'],
        output_selectors_sha256=sha(canonical([[s['ciphertext'],i]for row in outputs for s in row['segments']for i in range(s['elements'])]))))
    periodic=coordinate['logical_values']*(16384//period-1)
    result=bound(dict(format=TARGET,compiled_binding=compiled['binding'],runtime_origin_binding=runtime['binding'],
        request_id=compiled['request_id'],source_sha256=compiled['source_sha256'],input_slot_period=period,
        compiler_program_map=composed,logical_IO=io,target_outputs=targets,decoder_outputs=decoder,physical_identity=runtime['physical_identity'],
        logical_values=coordinate['logical_values'],unique_physical_cells=coordinate['unique_physical_cells'],logical_alias_cells=coordinate['alias_cells'],
        periodic_alias_values=periodic,complex_components_with_periodic_aliases=2*(coordinate['logical_values']+periodic),
        every_periodic_copy_included=True,numerical_values_decoded=0,
        full_decoder_contract_or_actual_host_input_math_CKKS_or_first_delivery_qualification=False))
    _bounded(result);return result


def pinned_B_modules():
    """Only versioned project byte/metadata validators, never candidate code."""
    result=[]
    for path,expected in [(BRIDGE,BRIDGE_SHA),(DECODER,DECODER_SHA)]:
        code=read_bytes(path);require(sha(code)==expected,'Frozen B periodic interface source changed')
        spec=importlib.util.spec_from_file_location('pinned_B_'+path.stem,str(path));module=importlib.util.module_from_spec(spec)
        exec(compile(code,str(path),'exec'),module.__dict__);result.append(module)
    return tuple(result)
