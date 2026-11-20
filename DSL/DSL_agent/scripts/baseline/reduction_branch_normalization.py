"""Exact real reduction rewrites before output; no DSL or feedback inputs."""
import copy
from benchmark_graph import validate,canonical,digest,require
from delivery_component import bound


def _workspace(model,maximum_nodes):
    require(type(maximum_nodes)is int and 1<=maximum_nodes<=1024,'Finite reduction graph budget')
    checked=validate(model);result=copy.deepcopy(model);result['nodes']=[]
    occupied=set(checked['shapes'])|{n['id']for n in model['nodes']};serial=0
    def fresh():
        nonlocal serial
        while True:
            name='reducepart'+str(serial);serial+=1
            if name not in occupied and name+'v0'not in occupied:
                occupied.add(name);occupied.add(name+'v0');return name
    def emit(op,refs,attrs,old=None):
        require(len(result['nodes'])<maximum_nodes,'Reduction expansion node budget')
        ident=old['id']if old else fresh();outputs=old['outputs']if old else[ident+'v0']
        result['nodes'].append(dict(id=ident,op=op,inputs=list(refs),attrs=copy.deepcopy(attrs),outputs=list(outputs)));return outputs[0]
    return checked,result,emit


def fuse_means(model,*,maximum_nodes=1024):
    checked,result,emit=_workspace(model,maximum_nodes);producers={v:n for n in model['nodes']for v in n['outputs']};changes=[]
    for node in model['nodes']:
        parent=producers.get(node['inputs'][0])
        if node['op']=='mean'and parent is not None and parent['op']=='mean':
            source=parent['inputs'][0];shape=checked['shapes'][source];axes={a%len(shape)for a in parent['attrs']['axes']}
            remaining=list(range(len(shape)))if parent['attrs']['keepdims']else[a for a in range(len(shape))if a not in axes]
            # A full reduction with keepdims=False uses the graph's [1]
            # scalar representation. Reducing that singleton adds no axis.
            if remaining:axes.update(remaining[a%len(remaining)]for a in node['attrs']['axes'])
            keep=[1 if a in axes else n for a,n in enumerate(shape)];target=checked['shapes'][node['outputs'][0]]
            value=emit('mean',[source],dict(axes=sorted(axes),keepdims=True),old=node if keep==target else None)
            if keep!=target:emit('reshape',[value],dict(shape=target),old=node)
            changes.append(dict(parent=parent['id'],node=node['id'],axes=sorted(axes),target_shape=target))
        else:result['nodes'].append(copy.deepcopy(node));require(len(result['nodes'])<=maximum_nodes,'Mean fusion node budget')
    by_value={v:n for n in result['nodes']for v in n['outputs']};needed=set();pending=[o['value']for o in result['outputs']]
    while pending:
        n=by_value.get(pending.pop())
        if n is not None and n['id']not in needed:needed.add(n['id']);pending.extend(n['inputs'])
    result['nodes']=[n for n in result['nodes']if n['id']in needed];actual=validate(result)
    require(result['inputs']==model['inputs']and result['outputs']==model['outputs']and actual['output_shapes']==checked['output_shapes'],'Mean fusion public signature')
    return result,bound(dict(format='poseidon-consecutive-mean-fusion-v1',input_model_sha256=digest(model),output_model_sha256=digest(result),changes=changes,
        exact_real_identity=True,floating_reassociation_error_must_be_measured=True,reference_used=False,DSL_generated=False))


def shard_retained_axes(model,*,maximum_nodes=1024):
    checked,result,emit=_workspace(model,maximum_nodes);changes=[]
    for node in model['nodes']:
        if node['op']not in('sum','mean'):
            result['nodes'].append(copy.deepcopy(node));require(len(result['nodes'])<=maximum_nodes,'Reduction sharding node budget');continue
        source=node['inputs'][0];shape=checked['shapes'][source];axes={a%len(shape)for a in node['attrs']['axes']}
        retained=[a for a,n in enumerate(shape)if n>1 and a not in axes]
        if not retained:
            result['nodes'].append(copy.deepcopy(node));require(len(result['nodes'])<=maximum_nodes,'Reduction sharding node budget');continue
        axis=min(retained,key=lambda a:(shape[a],a));target_axis=axis if node['attrs']['keepdims']else axis-sum(a<axis for a in axes)
        parts=[]
        for index in range(shape[axis]):
            piece=emit('slice',[source],dict(axis=axis,start=index,stop=index+1,step=1))
            parts.append(emit(node['op'],[piece],node['attrs']))
        while len(parts)>8:
            parts=[emit('concat',parts[i:i+8],dict(axis=target_axis))if len(parts[i:i+8])>1 else parts[i]for i in range(0,len(parts),8)]
        emit('concat',parts,dict(axis=target_axis),old=node)
        changes.append(dict(node=node['id'],retained_axis=axis,partitions=shape[axis],target_axis=target_axis))
    actual=validate(result);require(result['inputs']==model['inputs']and result['outputs']==model['outputs']and actual['output_shapes']==checked['output_shapes'],'Reduction sharding public signature')
    return result,bound(dict(format='poseidon-retained-axis-reduction-sharding-v1',input_model_sha256=digest(model),output_model_sha256=digest(result),changes=changes,
        selection='smallest non-singleton retained dimension, then axis index',reference_used=False,DSL_generated=False))
