"""Exact generic RuntimePlan partitioning and an independent data auditor.

This is an external compiler transformation, never candidate repair. The
original DSL, compiler receipt and whole plan stay immutable. It grants no live
compiler/HE, source-model mathematics or first-output qualification.
"""
import copy

from candidate_contract import strict_json
from delivery_component import bound,canonical,checked_bound,require,sha
from retained_compiler_provenance import FORMAT as COMPILER
from compiler_plan_provenance import FORMAT as DAG

FORMAT='poseidon-exact-compiler-partition-derivation-v1'
UNIT='poseidon-derived-compiler-partition-unit-v1'
MAX_VALUES=8192
MAX_OPERATIONS=4096
MAX_UNITS=16
MAX_IO=16


class PartitionUnsupported(ValueError):pass


def _source(receipt,raw,expected):
    require(type(expected)is dict and set(expected)=={'receipt_binding','request_id','source_sha256','whole_plan_sha256'},'Frozen whole compiler source selection')
    checked_bound(receipt,expected['receipt_binding'])
    require(receipt['format']==COMPILER and receipt['request_id']==expected['request_id']
        and receipt['source_sha256']==expected['source_sha256'],'Original whole-request/compiler/DSL identity')
    require(type(raw)is bytes and len(raw)<=16*1024**2 and sha(raw)==expected['whole_plan_sha256']
        ==receipt['files_read']['attempt-00/output/bound/plan.json']['sha256'],'Exact original compiler plan bytes')
    plan=strict_json(raw.decode());dag=checked_bound(receipt['DAG']);checked_bound(receipt['periodic_binding'])
    require(dag['format']==DAG and receipt['periodic_binding']['format']=='poseidon-original-periodic-payload-correspondence-v1'
        and receipt['periodic_binding']['all_non_payload_fields_unchanged']is True
        and receipt['periodic_binding']['plaintext_values_changed']is False,'Original bit-preserving public constant binding')
    require(plan['format_version']==1 and plan['finalization']==[],'Original CPU plan without finalization side effects')
    values={r['id']:r for r in plan['values']}
    require(len(values)==len(plan['values'])==dag['value_count']<=MAX_VALUES,'Whole source value inventory')
    require(plan['external_inputs']==[r['value_id']for r in dag['parameters']]
        and plan['final_outputs']==[r['value_id']for r in dag['returns']],'Original whole compiler I/O order')
    require(1<=len(plan['external_inputs'])<=MAX_IO and 1<=len(set(plan['final_outputs']))==len(plan['final_outputs'])<=MAX_IO,
        'Original physical caller I/O bounds')
    require(all(values[v]['kind']=='ciphertext'for v in plan['external_inputs']+plan['final_outputs']), 'Ciphertext boundaries only')
    defined=set(plan['external_inputs']);initial={};producer={v:-1 for v in defined};uses={v:[]for v in values}
    all_ops=plan['initialization']+plan['execution']
    require(len(all_ops)==dag['instruction_count']<=MAX_OPERATIONS and plan['execution'],'Original finite whole instruction inventory')
    for ordinal,op in enumerate(all_ops):
        phase='initialization'if ordinal<len(plan['initialization'])else'execution'
        require(op['ordinal']==ordinal and op['kind']in('encode','compute')
            and (phase!='initialization'or op['kind']=='encode'),'Supported pure encode/compute compiler schedule')
        require(type(op['output'])is str and op['output']in values and op['output']not in defined,'Unique original SSA definition')
        args=op.get('inputs',[]);require(type(args)is list and all(v in defined for v in args),'Original topological operand definitions')
        for value in args:uses[value].append(ordinal-len(plan['initialization']))
        defined.add(op['output'])
        producer[op['output']]=-1 if phase=='initialization'else ordinal-len(plan['initialization'])
        if phase=='initialization':
            require(values[op['output']]['kind']=='plaintext','Only pure public initial encodes may be replicated')
            initial[op['output']]=op
    require(defined==set(values),'No unproved or missing whole-plan SSA values')
    return plan,values,initial,producer,uses


def _mapping(op,ids,*,ordinal=None):
    require(op['output']in ids and all(v in ids for v in op.get('inputs',[])), 'All original SSA operand mappings present')
    value=copy.deepcopy(op);value['output']=ids[op['output']]
    if 'inputs'in value:value['inputs']=[ids[v]for v in value['inputs']]
    if ordinal is not None:value['ordinal']=ordinal
    return value


def _cuts(plan,cuts):
    require(type(cuts)is list and cuts==sorted(set(cuts))and all(type(n)is int and 0<n<len(plan['execution'])for n in cuts),
        'Explicit ordered compiler instruction cuts')
    endpoints=[0,*cuts,len(plan['execution'])]
    require(len(endpoints)-1<=MAX_UNITS,'Original maximum16 physical program units')
    return list(zip(endpoints,endpoints[1:]))


def partition(receipt,raw,cuts,expected):
    """Preserve every compute/streaming op; replicate only needed pure encodes."""
    plan,values,initial,producer,uses=_source(receipt,raw,expected);spans=_cuts(plan,cuts);units=[]
    for index,(start,end)in enumerate(spans):
        operations=plan['execution'][start:end];made={op['output']for op in operations}
        referenced={v for op in operations for v in op.get('inputs',[])}
        constants=[v for v in initial if v in referenced]
        external=list(plan['external_inputs'])if index==0 else [v for v in values if v in referenced-made-set(initial)]
        # Needed public initialization is a pure bit-preserving materialization.
        # Streaming encodes remain in their original execution position.
        require(all(values[v]['kind']=='ciphertext'for v in external),'Plaintext live across cut: choose a legal compiler cut')
        produced=[op['output']for op in operations]
        live=[v for v in produced if v in plan['final_outputs']or any(n>=end for n in uses[v])]
        if not live:raise PartitionUnsupported('Partition has no source-produced ciphertext boundary')
        if any(values[v]['kind']!='ciphertext'for v in live):raise PartitionUnsupported('Streaming plaintext crosses a proposed compiler cut')
        require(1<=len(external)<=MAX_IO and 1<=len(live)<=MAX_IO,'Original per-unit ciphertext boundary limit')
        ordered=[*external,*constants,*produced];require(len(set(ordered))==len(ordered),'Unique derived SSA inventory')
        local={v:str(i)for i,v in enumerate(ordered)}
        child={k:copy.deepcopy(v)for k,v in plan.items()if k not in ('values','external_inputs','initialization','execution','final_outputs')}
        child.update(values=[dict(copy.deepcopy(values[v]),id=local[v])for v in ordered],external_inputs=[local[v]for v in external],
            initialization=[_mapping(initial[v],local,ordinal=i)for i,v in enumerate(constants)],
            execution=[_mapping(op,local,ordinal=len(constants)+i)for i,op in enumerate(operations)],
            final_outputs=[local[v]for v in live])
        require(len(canonical(child))<=1024**2,'Original per-program file1MiB limit')
        unit=bound(dict(format=UNIT,unit_index=index,execution_span=[start,end],plan=child,plan_sha256=sha(canonical(child)),
            local_to_whole={local[v]:v for v in ordered},replicated_initial_values=constants,
            original_source_plan_sha256=sha(raw),compiler_receipt_binding=receipt['binding'],
            actually_compiled_as_separate_unit=False,exact_whole_compiler_plan_partition=True))
        units.append(unit)
    artifact=bound(dict(format=FORMAT,request_id=receipt['request_id'],source_sha256=receipt['source_sha256'],
        whole_compiler_receipt_binding=receipt['binding'],whole_plan_sha256=sha(raw),cuts=copy.deepcopy(cuts),units=units,
        compiler_DSL_or_whole_plan_modified=False,numerical_feedback_used=False,
        new_unit_compilation_or_native_execution=False,first_delivery_or_mathematical_or_CKKS_success_certified=False))
    audit(receipt,raw,artifact,expected,artifact['binding'])
    return artifact


def audit(receipt,raw,artifact,expected,expected_partition_binding):
    """Audit grafted SSA/operation coverage directly, without calling partition."""
    plan,values,initial,producer,uses=_source(receipt,raw,expected)
    checked_bound(artifact,expected_partition_binding)
    require(artifact['format']==FORMAT and artifact['whole_plan_sha256']==sha(raw)
        and artifact['whole_compiler_receipt_binding']==receipt['binding']
        and artifact['request_id']==receipt['request_id']and artifact['source_sha256']==receipt['source_sha256'],
        'Original partition source and compiler receipt')
    require(all(artifact.get(k)is False for k in ('compiler_DSL_or_whole_plan_modified','numerical_feedback_used',
        'new_unit_compilation_or_native_execution','first_delivery_or_mathematical_or_CKKS_success_certified')),
        'Source derivation cannot acquire fake execution or first-output authority')
    spans=_cuts(plan,artifact['cuts']);require(len(artifact['units'])==len(spans),'No omitted compiler partition')
    consumed=[];defined=set(plan['external_inputs']);initial_replicas=0;aggregate_values=aggregate_ops=0;boundaries=[]
    for index,(unit,(start,end))in enumerate(zip(artifact['units'],spans)):
        checked_bound(unit)
        require(unit['format']==UNIT and unit['unit_index']==index and unit['execution_span']==[start,end]
            and unit['original_source_plan_sha256']==sha(raw)and unit['compiler_receipt_binding']==receipt['binding']
            and unit['actually_compiled_as_separate_unit']is False and unit['exact_whole_compiler_plan_partition']is True,
            'Explicit source-derived unit, not fake compilation')
        child=unit['plan'];require(unit['plan_sha256']==sha(canonical(child))and len(canonical(child))<=1024**2,'Exact bounded derived plan bytes')
        require(canonical({k:v for k,v in child.items()if k not in ('values','external_inputs','initialization','execution','final_outputs')})
            ==canonical({k:v for k,v in plan.items()if k not in ('values','external_inputs','initialization','execution','final_outputs')}),
            'Original context/target/bundle/plan header unchanged')
        mapping=unit['local_to_whole'];local_values={v['id']:v for v in child['values']}
        require(len(local_values)==len(child['values'])==len(mapping)and set(mapping)==set(local_values)
            and len(set(mapping.values()))==len(mapping),'Bijective original local/global ValueIds')
        for key,global_id in mapping.items():
            require(global_id in values and canonical({k:v for k,v in local_values[key].items()if k!='id'})
                ==canonical({k:v for k,v in values[global_id].items()if k!='id'}),'Every original nominal descriptor, without retagging')
        require(1<=len(child['external_inputs'])<=MAX_IO and len(set(child['external_inputs']))==len(child['external_inputs']),
            'Complete unique bounded external SSA')
        incoming=[mapping[v]for v in child['external_inputs']]
        require(all(v in defined and values[v]['kind']=='ciphertext'for v in incoming),'Produced ciphertexts at the cut')
        if index==0:require(incoming==plan['external_inputs'],'All original whole inputs and order preserved')
        active=set(incoming);init_global=[]
        for ordinal,op in enumerate(child['initialization']):
            require(op['output']in mapping and mapping[op['output']]in initial,'Original public initialization source')
            mapped=_mapping(op,mapping,ordinal=initial[mapping[op['output']]]['ordinal'])
            global_id=mapped['output'];require(global_id in initial and canonical(mapped)==canonical(initial[global_id])
                and op['ordinal']==ordinal and global_id not in active,'Only exact original pure public Encode replication')
            init_global.append(global_id);active.add(global_id)
        require(init_global==unit['replicated_initial_values']and len(set(init_global))==len(init_global), 'Explicit unique public initialization')
        require(len(child['execution'])==end-start,'Complete execution span size')
        made=[]
        for local_index,op in enumerate(child['execution']):
            whole_index=start+local_index;original=plan['execution'][whole_index]
            mapped=_mapping(op,mapping,ordinal=original['ordinal'])
            require(op['ordinal']==len(init_global)+local_index and canonical(mapped)==canonical(original),
                'Every original arithmetic/operand/attrs/payload, once and in order')
            require(all(v in active for v in mapped.get('inputs',[]))and mapped['output']not in active,'Unit SSA dependencies, including streaming encodes')
            active.add(mapped['output']);made.append(mapped['output']);consumed.append(whole_index)
        require(active==set(mapping.values()),'No extra locally undeclared/unproduced SSA placeholders')
        require(1<=len(child['final_outputs'])<=MAX_IO and len(set(child['final_outputs']))==len(child['final_outputs']), 'Unique original physical unit returns')
        outgoing=[mapping[v]for v in child['final_outputs']]
        needed=[v for v in made if v in plan['final_outputs']or any(n>=end for n in uses[v])]
        require(outgoing==needed and all(values[v]['kind']=='ciphertext'for v in outgoing),
            'Every residual/live/final ciphertext at the original boundary')
        defined.update(outgoing);initial_replicas+=len(init_global)
        aggregate_values+=len(local_values);aggregate_ops+=len(child['initialization'])+len(child['execution'])
        require(aggregate_values<=MAX_VALUES and aggregate_ops<=MAX_OPERATIONS,'Original aggregate native program limits')
        boundaries.append(dict(unit_index=index,inputs=incoming,outputs=outgoing,plan_sha256=unit['plan_sha256'],
            actually_compiled_as_separate_unit=False))
    require(consumed==list(range(len(plan['execution'])))and all(v in defined for v in plan['final_outputs']), 'Whole original operation coverage and final reachability')
    return bound(dict(format='poseidon-exact-compiler-partition-audit-v1',partition_binding=expected_partition_binding,
        whole_compiler_receipt_binding=receipt['binding'],request_id=receipt['request_id'],source_sha256=receipt['source_sha256'],
        whole_plan_sha256=sha(raw),units=boundaries,whole_compute_and_streaming_operations_preserved=True,
        original_instructions=len(plan['initialization'])+len(plan['execution']),derived_instructions=aggregate_ops,
        derived_values=aggregate_values,public_encode_materializations=initial_replicas,
        whole_final_values=copy.deepcopy(plan['final_outputs']),native_final_selection_constructed=False,
        source_DSL_or_reference_or_arithmetic_or_nominal_states_changed=False,numerical_feedback_or_new_compilation_used=False,
        whole_DSL_mathematical_equivalence_or_live_compiler_host_certified=False,
        actual_input_same_key_final_output_or_CKKS_or_first_delivery_certified=False))
