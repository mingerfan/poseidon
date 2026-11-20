"""Bind retained raw compiler parameters to declared model and zero inputs.

This is host preparation of an encryption payload, not candidate evaluation.
No model reference, key, compiler, executable or external service is used.
"""
import hashlib
import math
import re
import numpy as np
from component_contract import reconstruct_request
from candidate_contract import request_input_names
import native_execution_profile as profile
import native_boot_plan_link_v3 as linker


def require(condition,reason):
    if not condition:raise ValueError('Native input binding: '+reason)


def _logical_array(value,shape):
    if type(value)is np.ndarray:
        require(value.dtype.kind in 'fi'and value.dtype.itemsize<=8,
                'binary64-compatible real numeric array, never boolean/object')
    else:
        def numeric(item):
            return all(numeric(v)for v in item)if type(item)in(list,tuple)else type(item)in(int,float)
        require(numeric(value),'real numeric values, never boolean or strings')
    try:array=np.asarray(value,dtype=np.float64)
    except (TypeError,ValueError,OverflowError)as error:
        raise ValueError('Native input binding: numeric array conversion')from error
    require(list(array.shape)==shape and np.isfinite(array).all(),'exact finite logical shape')
    require(array.size<=256 and float(abs(array).max())<=.5,'original closed input domain')
    return array


def prepare_inputs(request,source_raw,derived_raw,link_raw,capability_raw,spec_raw,
                   task_binding,logical_sample):
    """Return exact client input JSON data and a value-free positional receipt."""
    reconstruct_request(request)
    profile.validate_request(request)
    require(type(task_binding)is str and re.fullmatch(r'[0-9a-f]{64}',task_binding)is not None,
            'exact task binding')
    for raw in [source_raw,derived_raw,link_raw,capability_raw,spec_raw]:
        require(type(raw)is bytes and 0<len(raw)<=256*1024,'bounded immutable protocol JSON bytes')
    expected,expected_link=linker.link_fragment(source_raw,capability_raw,spec_raw)
    derived=linker.parse(derived_raw);record=linker.parse(link_raw)
    require(linker.encode(derived)==linker.encode(expected)and
            linker.encode(record)==linker.encode(expected_link),'independently recomputed complete derivation')
    source=linker.parse(source_raw);names=request_input_names(request)
    require(len(source['external_inputs'])==len(names),'raw compiler parameter count')
    require(record['original_source_external_input_ids']==source['external_inputs'],
            'exact original compiler parameter order')
    layout=request['layout'];period=layout['input_slot_period']
    declaration=layout['inputs'][0];logical=request['model']['inputs'][0]
    require(type(logical_sample)is dict and set(logical_sample)=={logical['name']},
            'exact named model inputs')
    array=_logical_array(logical_sample[logical['name']],logical['shape'])
    require(declaration['name']==logical['name']and declaration['shape']==logical['shape'],
            'logical-to-physical declaration')
    require(array.size<=period,'logical data fits declared periodic input')
    model_values=np.pad(array.reshape(-1),(0,period-array.size)).tolist()
    auxiliary=layout.get('auxiliary_ciphertexts',[])
    require(len(auxiliary)==1 and auxiliary[0]['dsl_name']=='zero_ct'and
            auxiliary[0]['kind']=='encrypted_zero'and auxiliary[0]['source']=='trusted_client'and
            auxiliary[0]['slot_period']==period,'declared trusted-client encrypted zero')
    inputs=[];bindings=[]
    for original,derived_identifier in zip(record['retained_source_external_input_ids'],
                                           derived['external_inputs']):
        position=source['external_inputs'].index(original);name=names[position]
        require(record['source_input_to_derived_external_ids'][original]==derived_identifier,
                'retained compiler-to-derived input association')
        if name==declaration['dsl_name']:
            values=model_values;kind='model_input'
        else:
            require(name=='zero_ct','retained parameter must be a declared input')
            values=[0.]*period;kind='encrypted_zero'
        inputs.append(dict(value_id=derived_identifier,slot_period=period,values=values))
        bindings.append(dict(source_parameter_index=position,source_value_id=original,
            source_parameter=name,derived_value_id=derived_identifier,kind=kind))
    require(len(inputs)==len(record['retained_source_external_input_ids'])==len(derived['external_inputs']),
            'complete retained input inventory')
    payload=dict(format='poseidon-native-sdk-plain-input-v1',task_binding=task_binding,inputs=inputs)
    receipt=dict(format='poseidon-native-source-parameter-input-binding-v1',
        request_id=request['request_id'],task_binding=task_binding,
        source_plan_sha256=hashlib.sha256(source_raw).hexdigest(),
        runtime_plan_sha256=hashlib.sha256(derived_raw).hexdigest(),
        link_record_sha256=hashlib.sha256(link_raw).hexdigest(),
        retained_parameter_bindings=bindings,original_parameter_count=len(names),
        dropped_only_unreferenced_source_inputs=record['removed_unreferenced_source_external_input_ids'],
        slot_period=period,numerical_or_encrypted_acceptance=False,new_keys=0)
    return payload,receipt
