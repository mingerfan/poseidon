"""Prototype: drop only completely unreferenced input descriptors, preserving raw compilation.

This is a data-only Agent adapter, not a library patch, numerical FHE simulation
or released SDK. Every original arithmetic instruction and retained physical
descriptor is preserved. Native execution and genuine Agent acceptance remain
required for the new protocol version.
"""
import copy
import hashlib

import native_boot_plan_link_v2 as prior

POLICY='poseidon-native-certified-unused-input-link-prototype-v3'
LinkError=prior.LinkError
encode=prior.encode
parse=prior.parse
digest=prior.digest
require=prior.require


def _projection(plan,input_labels):
    references={value:['source_input',input_labels[value]]for value in plan['external_inputs']}
    nodes=[]
    for instruction in plan['execution']:
        operands=[references[value]for value in instruction['inputs']]
        if instruction['op']=='mul_cc':
            nodes.append(dict(operation='mul_cc',inputs=operands))
            references[instruction['output']]=['node',len(nodes)-1]
        else:
            require(instruction['op']in('relinearize','rescale','boot'),'nontransparent_operation')
            references[instruction['output']]=operands[0]
    return dict(nodes=nodes,outputs=[references[value]for value in plan['final_outputs']])


def link_fragment(source_raw,capability_raw,spec_raw):
    capability,spec=prior.authority(capability_raw,spec_raw)
    source=parse(source_raw)
    descriptors=prior.check_plan(source,capability,spec,allow_boot=False)
    # An unused input is not a way to hide a mismatched compiler ABI.
    require(all((descriptors[value]['level'],descriptors[value]['scale_log2'],
                 descriptors[value]['components'])==(1,44,2)
                for value in source['external_inputs']),
            'all_original_inputs_must_match_certified_post_Boot_state')
    referenced=set(source['final_outputs'])
    for instruction in source['execution']:
        referenced.update(instruction['inputs'])
    retained=[value for value in source['external_inputs']if value in referenced]
    removed=[value for value in source['external_inputs']if value not in referenced]
    require(retained,'at_least_one_referenced_cipher_input')
    pruned=copy.deepcopy(source)
    pruned['external_inputs']=retained
    pruned['values']=[value for value in pruned['values']if value['id']not in removed]
    require(pruned['execution']==source['execution']and pruned['final_outputs']==source['final_outputs'],
            'original_compiled_instructions_and_output_order_unchanged')
    pruned_raw=encode(pruned)
    linked,record=prior.link_fragment(pruned_raw,capability_raw,spec_raw)
    require(record['uniform_physical_level_delta']==0,'physical_translation_forbidden')
    linked['plan_id']=str(int(hashlib.sha256(source_raw).hexdigest()[:15],16))
    linked_descriptors=prior.check_plan(linked,capability,spec,allow_boot=True)
    boots=sum(op['op']=='boot'for op in linked['execution'])
    require(1<=boots<=2,'certified_two_Boot_invocation_budget')
    mapping=record['source_to_derived_values']
    for identifier,descriptor in descriptors.items():
        if identifier in removed:continue
        mapped=linked_descriptors[mapping[identifier]]
        require({k:v for k,v in mapped.items()if k!='id'}==
                {k:v for k,v in descriptor.items()if k!='id'},'retained_physical_descriptor_changed')
    labels={value:value for value in source['external_inputs']}
    derived_labels={derived:original for derived,original in zip(linked['external_inputs'],retained)}
    original_projection=_projection(source,labels)
    require(original_projection==_projection(linked,derived_labels),'original_arithmetic_DAG_or_input_binding_changed')
    record.update(format=POLICY,source_plan_sha256=hashlib.sha256(source_raw).hexdigest(),
        source_pruned_fragment_sha256=hashlib.sha256(pruned_raw).hexdigest(),
        derived_plan_sha256=hashlib.sha256(encode(linked)).hexdigest(),
        original_source_external_input_ids=source['external_inputs'],
        retained_source_external_input_ids=retained,
        removed_unreferenced_source_external_input_ids=removed,
        removed_source_input_descriptors=[descriptors[value]for value in removed],
        original_input_values_independent_of_removed_arguments=True,
        source_input_to_derived_external_ids={value:derived for value,derived in zip(retained,linked['external_inputs'])},
        all_retained_source_physical_descriptors_preserved=True,
        original_all_arithmetic_instructions_preserved=True,
        original_parameter_binding_projection_sha256=digest(original_projection),
        derived_Boot_calls=boots,actual_native_execution=False,SDK_gate_enabled=False)
    record['pruned_fragment_semantic_dag_sha256']=record.pop('semantic_dag_sha256')
    record.pop('binding',None);record['binding']=digest(record)
    return linked,record
