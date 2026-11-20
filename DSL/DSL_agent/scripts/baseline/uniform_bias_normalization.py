"""Lift a uniform public convolution bias into its exact scalar addition.

This is a mathematical graph normalization before generation. It avoids a
separate masked bias vector and its repeated origin records. It neither changes
weights nor assumes anything about encrypted data, padding lanes or CKKS error.
"""
import copy
import hashlib
import json

from benchmark_graph import validate, require

VERSION = 'poseidon-uniform-public-conv-bias-v1'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def normalize(model, *, maximum_nodes=1024):
    require(type(maximum_nodes)is int and 1<=maximum_nodes<=1024,'Finite bias normalization node budget')
    checked=validate(model);original=canonical(model)
    result=copy.deepcopy(model);result['nodes']=[]
    occupied=set(checked['shapes'])|{node['id']for node in model['nodes']}
    scalars={canonical(v):name for name,v in model['constants'].items()if type(v)in(int,float)}
    serial=0;changes=[]
    def fresh():
        nonlocal serial
        while True:
            name='bias'+str(serial);serial+=1
            if name not in occupied and name+'v0'not in occupied:
                occupied.add(name);occupied.add(name+'v0');return name
    for node in model['nodes']:
        refs=node['inputs']
        if node['op']in('conv1d','conv2d')and len(refs)==3:
            bias=model['constants'][refs[2]]
            uniform=type(bias)is list and bool(bias)and all(canonical(v)==canonical(bias[0])for v in bias)
        else:uniform=False
        if not uniform:
            result['nodes'].append(copy.deepcopy(node))
        else:
            key=canonical(bias[0])
            if key not in scalars:
                name=fresh();result['constants'][name]=copy.deepcopy(bias[0]);scalars[key]=name
            scalar=scalars[key];ident=fresh();value=ident+'v0'
            result['nodes'].append(dict(id=ident,op=node['op'],inputs=refs[:2],attrs=copy.deepcopy(node['attrs']),outputs=[value]))
            result['nodes'].append(dict(id=node['id'],op='add',inputs=[value,scalar],attrs={},outputs=copy.deepcopy(node['outputs'])))
            changes.append(dict(node_id=node['id'],original_bias=refs[2],exact_scalar=scalar,
                original_scalar_sha256=hashlib.sha256(key).hexdigest(),new_weight_values=False))
        require(len(result['nodes'])<=maximum_nodes,'Bias normalization node budget')
    final=validate(result)
    require(result['inputs']==model['inputs']and result['outputs']==model['outputs']
            and all(result['constants'][name]==value for name,value in model['constants'].items()),'Bias normalization changed original signature/weights')
    require(final['output_shapes']==checked['output_shapes']and canonical(model)==original,'Bias normalization changed output or input object')
    receipt=dict(format=VERSION,original_model_sha256=hashlib.sha256(original).hexdigest(),
        normalized_model_sha256=hashlib.sha256(canonical(result)).hexdigest(),changes=changes,
        original_nodes=len(model['nodes']),normalized_nodes=len(result['nodes']),maximum_nodes=maximum_nodes,
        original_input_and_named_outputs_preserved=True,original_weight_values_preserved=True,model_name_dispatch=False,
        compiler_feedback_used=False,numerical_reference_used=False,DSL_generated=False,resource_limits_increased=False,
        scope='uniform public bias only; nonuniform/signed-zero-distinct vectors retain original convolution')
    receipt['binding']=hashlib.sha256(canonical(receipt)).hexdigest()
    return result,receipt
