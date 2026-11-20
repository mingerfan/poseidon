"""Generic pre-output convolution representation using slices and linear maps.

Original coefficient values are only rearranged. Implicit padding/group zeros
become explicit. This emits no DSL and establishes no encrypted capability.
"""
import copy
import math

from benchmark_graph import validate, infer, graph_resource_profile, canonical, digest, require
from model_resources import resolve
from delivery_component import bound

VERSION='poseidon-conv2d-im2col-normalization-v1'


def normalize(model, *, maximum_nodes=1024, maximum_patch_elements=65536):
    require(type(maximum_nodes)is int and 1<=maximum_nodes<=1024,'Finite convolution normalization node budget')
    require(type(maximum_patch_elements)is int and 1<=maximum_patch_elements<=65536,'Finite convolution patch tensor budget')
    checked=validate(model);profile=graph_resource_profile(model);limits=resolve(profile)
    original=canonical(model);result=copy.deepcopy(model);result['nodes']=[]
    shapes={name:tuple(shape)for name,shape in checked['shapes'].items()};occupied=set(shapes)|{n['id']for n in model['nodes']};serial=0
    changes=[];unchanged=[]

    def fresh():
        nonlocal serial
        while True:
            name='imcol'+str(serial);serial+=1
            if name not in occupied and name+'v0'not in occupied:
                occupied.add(name);occupied.add(name+'v0');return name

    def constant(value):
        encoded=canonical(value)
        for name,old in result['constants'].items():
            if canonical(old)==encoded:return name
        require(len(result['constants'])<limits.constants,'Convolution constant count budget')
        name=fresh();result['constants'][name]=copy.deepcopy(value)
        from benchmark_graph import shape
        shapes[name]=shape(value,resource_profile=profile)
        return name

    def emit(op,refs,attrs,old=None):
        require(len(result['nodes'])<maximum_nodes,'Convolution normalization expanded node budget')
        ident=old['id']if old else fresh();outputs=old['outputs']if old else[ident+'v0']
        inferred=infer(op,[shapes[r]for r in refs],attrs,resource_profile=profile)
        require(len(inferred)==len(outputs)==1,'Single convolution intermediate result')
        result['nodes'].append(dict(id=ident,op=op,inputs=list(refs),attrs=copy.deepcopy(attrs),outputs=list(outputs)))
        shapes[outputs[0]]=inferred[0];return outputs[0]

    def concat(values,axis):
        while len(values)>1:
            values=[emit('concat',values[i:i+8],dict(axis=axis))if len(values[i:i+8])>1 else values[i]for i in range(0,len(values),8)]
        return values[0]

    def cut(value,axis,start,stop,step=1):
        if start==0 and stop==shapes[value][axis]and step==1:return value
        return emit('slice',[value],dict(axis=axis,start=start,stop=stop,step=step))

    for node in model['nodes']:
        if node['op']!='conv2d':
            result['nodes'].append(copy.deepcopy(node));require(len(result['nodes'])<=maximum_nodes,'Convolution normalization node budget');continue
        x,wname=node['inputs'][:2];a=node['attrs'];s=shapes[x];wshape=shapes[wname]
        batched=len(s)==4;caxis=1 if batched else 0;haxis=caxis+1;waxis=caxis+2
        n=s[0]if batched else 1;c=s[caxis];height,width=s[-2:];outc,cgroup,kh,kw=wshape
        oh,ow=shapes[node['outputs'][0]][-2:];groups=a['groups'];ph,pw=a['padding'];sh,sw=a['stride'];dh,dw=a['dilation']
        patch=n*oh*ow*c*kh*kw;weight_elements=outc*c*kh*kw
        padded_elements=n*c*(height+2*ph)*(width+2*pw)
        total_original_constants=sum(math.prod(shapes[k])for k in result['constants'])
        reason=('patch_tensor_rank_budget'if len(s)+1>limits.rank else
                'padding_width_exceeds_source_axis'if ph>height or pw>width else
                'patch_tensor_budget'if patch>min(maximum_patch_elements,limits.tensor_elements)else
                'padded_tensor_budget'if padded_elements>limits.tensor_elements else
                'public_matrix_budget'if weight_elements>limits.constant_elements or total_original_constants+weight_elements+1>limits.total_constant_elements else None)
        if reason:
            result['nodes'].append(copy.deepcopy(node));unchanged.append(dict(node_id=node['id'],reason=reason,patch_elements=patch));continue
        padded=x
        if ph or pw:
            zero=emit('multiply',[x,constant(0.0)],{})
            if pw:
                side=cut(zero,waxis,0,pw);padded=concat([side,padded,side],waxis)
            if ph:
                top=cut(zero,haxis,0,ph)
                if pw:
                    corner=cut(top,waxis,0,pw);top=concat([corner,top,corner],waxis)
                padded=concat([top,padded,top],haxis)
        patches=[]
        for ky in range(kh):
            for kx in range(kw):
                view=cut(padded,haxis,ky*dh,ky*dh+(oh-1)*sh+1,sh)
                view=cut(view,waxis,kx*dw,kx*dw+(ow-1)*sw+1,sw)
                target=([n,c,1,oh,ow]if batched else[c,1,oh,ow])
                patches.append(emit('reshape',[view],dict(shape=target)))
        columns=concat(patches,caxis+1)
        columns=emit('reshape',[columns],dict(shape=([n,c*kh*kw,oh,ow]if batched else[c*kh*kw,oh,ow])))
        columns=emit('permute',[columns],dict(dims=([0,2,3,1]if batched else[1,2,0])))
        columns=emit('reshape',[columns],dict(shape=[n*oh*ow,c*kh*kw]))
        weights=model['constants'][wname];matrix=[]
        per_output_group=outc//groups
        for o in range(outc):
            row=[];group=o//per_output_group
            for i in range(c):
                for ky in range(kh):
                    for kx in range(kw):
                        row.append(weights[o][i-group*cgroup][ky][kx]if group*cgroup<=i<(group+1)*cgroup else 0.0)
            matrix.append(row)
        weight=constant(matrix)
        linear=emit('linear',[columns,weight,*node['inputs'][2:]],{})
        linear=emit('reshape',[linear],dict(shape=([n,oh,ow,outc]if batched else[oh,ow,outc])))
        emit('permute',[linear],dict(dims=([0,3,1,2]if batched else[2,0,1])),old=node)
        changes.append(dict(node_id=node['id'],original_weight=wname,linear_weight=weight,groups=groups,
            input_shape=list(s),output_shape=list(shapes[node['outputs'][0]]),patch_elements=patch,
            original_weight_elements=math.prod(wshape),linear_weight_elements=weight_elements,
            added_values='implicit padding/group zeros only; original coefficients reordered exactly'))
    actual=validate(result)
    require(canonical(model)==original,'Convolution normalization mutated caller')
    require(result['inputs']==model['inputs']and result['outputs']==model['outputs']and actual['output_shapes']==checked['output_shapes'],
            'Convolution public input/output signature changed')
    return result,bound(dict(format=VERSION,input_model_sha256=digest(model),output_model_sha256=digest(result),changes=changes,unchanged=unchanged,
        old_nodes=len(model['nodes']),new_nodes=len(result['nodes']),maximum_nodes=maximum_nodes,maximum_patch_elements=maximum_patch_elements,
        model_name_dispatch=False,DSL_generated=False,compiler_or_numeric_feedback_used=False,encrypted_capability_certified=False))
