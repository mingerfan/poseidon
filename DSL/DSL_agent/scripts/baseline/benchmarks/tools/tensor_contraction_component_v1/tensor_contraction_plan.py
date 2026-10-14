"""Static mathematical tensordot plan using existing tensor/matrix primitives."""
import math
from benchmark_graph import require,digest,canonical
from model_resources import resolve
VERSION="poseidon-static-tensor-contraction-v1"

def shape(value,limits):
    require(type(value)is list and len(value)<=limits.rank and
        all(type(x)is int and 0<x<=limits.axis_size for x in value)and
        math.prod(value)<=limits.tensor_elements,"Contraction tensor shape budget")
    return list(value)

def axes(a_rank,b_rank,dims):
    if type(dims)is int:
        require(0<=dims<=min(a_rank,b_rank),"Contraction dimension count")
        return list(range(a_rank-dims,a_rank)),list(range(dims))
    require(type(dims)in(list,tuple)and len(dims)==2,"Contraction static axis pair")
    result=[]
    for selected,rank in zip(dims,(a_rank,b_rank)):
        require(type(selected)in(list,tuple)and len(selected)<=rank and
            all(type(x)is int and -rank<=x<rank for x in selected),"Contraction static axes")
        normalized=[x%rank for x in selected]
        require(len(set(normalized))==len(normalized),"Contraction duplicate axis")
        result.append(normalized)
    require(len(result[0])==len(result[1]),"Contraction axis count mismatch")
    return result[0],result[1]

def plan(a_shape,b_shape,dims=2,*,resource_profile="legacy-v1"):
    limits=resolve(resource_profile)
    a,b=shape(a_shape,limits),shape(b_shape,limits)
    aa,bb=axes(len(a),len(b),dims)
    left_free=[i for i in range(len(a))if i not in aa]
    right_free=[i for i in range(len(b))if i not in bb]
    expanded_a,expanded_b=list(a),list(b)
    contraction=[]
    for i,j in zip(aa,bb):
        require(a[i]==b[j]or a[i]==1 or b[j]==1,"Contraction axis size mismatch")
        size=max(a[i],b[j]);contraction.append(size)
        expanded_a[i]=expanded_b[j]=size
    output=[a[i]for i in left_free]+[b[i]for i in right_free]
    require(len(output)<=limits.rank and math.prod(output)<=limits.tensor_elements,
        "Contraction output tensor budget")
    require(math.prod(expanded_a)<=limits.tensor_elements and
        math.prod(expanded_b)<=limits.tensor_elements,"Contraction broadcast tensor budget")
    m=math.prod(a[i]for i in left_free);n=math.prod(b[i]for i in right_free);k=math.prod(contraction)
    steps=[]
    for name,original,expanded,order,matrix in (
        ("a",a,expanded_a,left_free+aa,[m,k]),
        ("b",b,expanded_b,bb+right_free,[k,n])):
        if original!=expanded:steps.append(dict(op="expand",operand=name,attrs=dict(shape=expanded)))
        steps.append(dict(op="permute",operand=name,attrs=dict(axes=order)))
        steps.append(dict(op="reshape",operand=name,attrs=dict(shape=matrix)))
    steps.extend([dict(op="matmul",inputs=["a","b"]),
        dict(op="reshape",operand="result",attrs=dict(shape=output)),dict(op="clone",operand="result")])
    require(len(steps)<=limits.expanded_nodes,"Contraction minimum node budget")
    result=dict(version=VERSION,a_shape=a,b_shape=b,a_axes=aa,b_axes=bb,
        broadcast_a_shape=expanded_a,broadcast_b_shape=expanded_b,output_shape=output,
        matrix_shapes=[[m,k],[k,n]],steps=steps,resource_profile=resource_profile,
        scalar_multiply_terms=m*k*n,
        decomposition_estimate=dict(high_level_steps=len(steps),certainty="exact_high_level_plan",
            core_and_layout_validation_required=True,physical_ciphertext_cost="unknown"),
        production_active=False,backend_or_layout_validated=False)
    require(len(canonical(result))<=limits.json_bytes,"Contraction plan byte budget")
    result["binding"]=digest(result)
    return result

def check_plan(declared,a_shape,b_shape,dims=2,**options):
    require(type(declared)is dict and
        canonical(declared)==canonical(plan(a_shape,b_shape,dims,**options)),"Contraction plan identity")
    return declared
