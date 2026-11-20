"""Exact real unpadded 2x2/stride2 pooling with binary column compaction.

Only public geometry chooses this graph representation. Four-neighbour sums
are compacted by overwrite butterflies, followed by one block-prefix gather.
Everything is an existing graph operator; no DSL, runtime or feedback is used.
"""
import copy
import math
from benchmark_graph import validate,canonical,digest,require
from delivery_component import bound

VERSION='poseidon-dyadic-pool-graph-v1'


def normalize(model,*,maximum_nodes=1024):
    require(type(maximum_nodes)is int and 1<=maximum_nodes<=1024,'Finite dyadic pool node budget')
    checked=validate(model);before=canonical(model);new=copy.deepcopy(model);new['nodes']=[]
    occupied=set(checked['shapes'])|{n['id']for n in model['nodes']};serial=0;changes=[];fallback=[]
    constants={canonical(v):k for k,v in new['constants'].items()}
    def fresh():
        nonlocal serial
        while True:
            name='dyadicpool'+str(serial);serial+=1
            if name not in occupied and name+'v0'not in occupied:occupied.add(name);occupied.add(name+'v0');return name
    def public(value):
        raw=canonical(value)
        if raw not in constants:
            key=fresh();new['constants'][key]=value;constants[raw]=key
        return constants[raw]
    def emit(op,refs,attrs,old=None):
        name=old['id']if old else fresh();outputs=list(old['outputs'])if old else[name+'v0']
        new['nodes'].append(dict(id=name,op=op,inputs=list(refs),attrs=copy.deepcopy(attrs),outputs=outputs))
        require(len(new['nodes'])<=maximum_nodes,'Dyadic pool expansion node budget');return outputs[0]
    def tensor(flat,shape):
        def nested(start,dims):
            if len(dims)==1:return flat[start:start+dims[0]]
            stride=math.prod(dims[1:]);return[nested(start+i*stride,dims[1:])for i in range(dims[0])]
        return nested(0,shape)
    for node in model['nodes']:
        if node['op']!='avg_pool2d':
            new['nodes'].append(copy.deepcopy(node));require(len(new['nodes'])<=maximum_nodes,'Dyadic pool node budget');continue
        shape=checked['shapes'][node['inputs'][0]];height,width=shape[-2:];a=node['attrs']
        eligible=(a['kernel']==[2,2]and a['stride']==[2,2]and a['padding']==[0,0]
            and height%2==0 and width>=2 and width&(width-1)==0)
        if not eligible:
            new['nodes'].append(copy.deepcopy(node));require(len(new['nodes'])<=maximum_nodes,'Dyadic pool node budget')
            fallback.append(node['id']);continue
        source=node['inputs'][0]
        shifted=emit('rotate',[source],dict(step=1));horizontal=emit('add',[source,shifted],{})
        shifted=emit('rotate',[horizontal],dict(step=width));value=emit('add',[horizontal,shifted],{})
        stages=[];step=1
        while step<width//2:
            mask=[float(step<=i%width%(4*step)<2*step)for i in range(math.prod(shape))]
            key=public(tensor(mask,shape));shifted=emit('rotate',[value],dict(step=step))
            delta=emit('subtract',[shifted,value],{});delta=emit('multiply',[delta,key],{});value=emit('add',[value,delta],{})
            stages.append(dict(step=step,mask=key));step*=2
        # After compaction, each row's first W/2 cells are the even original
        # columns. Group (row parity, column half) into one four-way axis.
        view=[*shape[:-2],height//2,4,width//2]
        value=emit('reshape',[value],dict(shape=view))
        value=emit('slice',[value],dict(axis=len(view)-2,start=0,stop=1,step=1))
        value=emit('reshape',[value],dict(shape=checked['shapes'][node['outputs'][0]]))
        emit('multiply',[value,public(.25)],{},old=node)
        changes.append(dict(node=node['id'],source_shape=shape,target_shape=checked['shapes'][node['outputs'][0]],
            stages=stages,window='horizontal then vertical neighbour sums; only valid even anchors retained',
            gather='first of four equal blocks after a view',original_count_include_pad=a['count_include_pad']))
    final=validate(new)
    require(new['inputs']==model['inputs']and new['outputs']==model['outputs']and final['output_shapes']==checked['output_shapes'],
        'Dyadic pool changed public signature')
    require(all(canonical(new['constants'][k])==canonical(v)for k,v in model['constants'].items()),'Dyadic pool changed an original public value')
    require(canonical(model)==before,'Dyadic pool mutated input')
    return new,bound(dict(format=VERSION,input_model_sha256=digest(model),output_model_sha256=digest(new),changes=changes,fallback_nodes=fallback,
        new_constants='geometry-derived exact binary masks and exact quarter only',exact_real_identity=True,
        floating_reassociation_error_must_be_measured=True,model_name_dispatch=False,numerical_reference_used=False,
        compiler_feedback_used=False,DSL_generated=False,backend_or_memory_certified=False))
