"""Expose complete reductions when every retained axis is a singleton.

The operator and all values remain the same real function. The exact original
output shape is restored when scalar-shape conventions differ. Floating error
from reduction order must still be independently measured by the caller.
"""
import copy
from benchmark_graph import validate,canonical,digest,require
from delivery_component import bound

VERSION='poseidon-singleton-aware-complete-reductions-v1'


def normalize(model,*,maximum_nodes=1024):
    require(type(maximum_nodes)is int and 1<=maximum_nodes<=1024,'Finite singleton-reduction node budget')
    checked=validate(model);before=canonical(model);new=copy.deepcopy(model);new['nodes']=[]
    occupied=set(checked['shapes'])|{n['id']for n in model['nodes']};serial=0;changes=[]
    for node in model['nodes']:
        replacement=copy.deepcopy(node)
        if node['op']in('mean','sum'):
            shape=checked['shapes'][node['inputs'][0]];axes={a%len(shape)for a in node['attrs']['axes']};all_axes=set(range(len(shape)))
            if axes!=all_axes and all(n==1 or a in axes for a,n in enumerate(shape)):
                replacement['attrs']['axes']=list(range(len(shape)))
                target=checked['shapes'][node['outputs'][0]];whole_shape=[1]*len(shape)if node['attrs']['keepdims']else[1]
                if whole_shape!=target:
                    while True:
                        ident='singletonreduce'+str(serial);serial+=1
                        if ident not in occupied and ident+'v0'not in occupied:break
                    occupied.add(ident);occupied.add(ident+'v0')
                    replacement['id']=ident;replacement['outputs']=[ident+'v0'];new['nodes'].append(replacement)
                    replacement=dict(id=node['id'],op='reshape',inputs=[ident+'v0'],attrs=dict(shape=target),outputs=list(node['outputs']))
                changes.append(dict(node=node['id'],original_axes=node['attrs']['axes'],complete_axes=list(range(len(shape))),output_shape=target))
        new['nodes'].append(replacement);require(len(new['nodes'])<=maximum_nodes,'Singleton-reduction expansion node budget')
    final=validate(new)
    require(new['inputs']==model['inputs']and new['outputs']==model['outputs']and new['constants']==model['constants']
        and final['output_shapes']==checked['output_shapes'],'Singleton reduction changed public function signature')
    require(canonical(model)==before,'Singleton reduction mutated input')
    return new,bound(dict(format=VERSION,input_model_sha256=digest(model),output_model_sha256=digest(new),changes=changes,
        exact_real_identity=True,output_shapes_and_values_preserved=True,floating_reassociation_error_must_be_measured=True,
        model_name_dispatch=False,reference_used=False,compiler_feedback_used=False,DSL_generated=False))
