"""Lossless internal graph-name compaction before a Provider request exists.

Public input/output names, operators, attributes and all numeric values remain
exact. The inverse mapping must reproduce the entire original graph. No DSL
source or response is read or rewritten.
"""
import copy
import hashlib
import json

from benchmark_graph import validate, require

VERSION='poseidon-lossless-internal-graph-names-v1'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _rename(model,constants,values,nodes):
    result=copy.deepcopy(model);refs={**constants,**values}
    result['constants']={constants[k]:v for k,v in result['constants'].items()}
    for node in result['nodes']:
        node['id']=nodes[node['id']]
        node['inputs']=[refs.get(v,v)for v in node['inputs']]
        node['outputs']=[values[v]for v in node['outputs']]
    for output in result['outputs']:output['value']=refs.get(output['value'],output['value'])
    return result


def restore(model,receipt):
    body={k:v for k,v in receipt.items()if k!='binding'}
    require(receipt.get('binding')==hashlib.sha256(canonical(body)).hexdigest()and receipt.get('format')==VERSION,
        'Frozen reversible graph-name receipt')
    require(hashlib.sha256(canonical(model)).hexdigest()==receipt['normalized_model_sha256'],'Renamed graph changed')
    result=_rename(model,{v:k for k,v in receipt['constant_names'].items()},
        {v:k for k,v in receipt['value_names'].items()},{v:k for k,v in receipt['node_names'].items()})
    require(hashlib.sha256(canonical(result)).hexdigest()==receipt['original_model_sha256'],'Original graph cannot be recovered')
    return result


def normalize(model):
    checked=validate(model);original=canonical(model)
    reserved={s['name']for s in model['inputs']}|{s['name']for s in model['outputs']}
    def allocate(prefix,count):
        result=[];index=0
        while len(result)<count:
            name=prefix+str(index);index+=1
            if name not in reserved:result.append(name);reserved.add(name)
        return result
    constant_order=sorted(model['constants'])
    value_order=[v for node in model['nodes']for v in node['outputs']]
    node_order=[node['id']for node in model['nodes']]
    constants=dict(zip(constant_order,allocate('c',len(constant_order))))
    values=dict(zip(value_order,allocate('v',len(value_order))))
    nodes=dict(zip(node_order,allocate('n',len(node_order))))
    result=_rename(model,constants,values,nodes);final=validate(result)
    require(result['inputs']==model['inputs']and [o['name']for o in result['outputs']]==[o['name']for o in model['outputs']]
        and checked['output_shapes']==final['output_shapes'],'Graph-name compaction changed public signature')
    receipt=dict(format=VERSION,original_model_sha256=hashlib.sha256(original).hexdigest(),
        normalized_model_sha256=hashlib.sha256(canonical(result)).hexdigest(),constant_names=constants,value_names=values,node_names=nodes,
        original_bytes=len(original),normalized_bytes=len(canonical(result)),node_count_unchanged=True,
        operators_attributes_and_weight_values_unchanged=True,public_input_and_output_names_preserved=True,
        model_name_dispatch=False,DSL_source_or_response_rewritten=False,compiler_or_numeric_feedback_used=False,resource_limits_increased=False)
    receipt['binding']=hashlib.sha256(canonical(receipt)).hexdigest()
    require(canonical(restore(result,receipt))==original and canonical(model)==original,'Graph-name compaction is not lossless')
    return result,receipt
