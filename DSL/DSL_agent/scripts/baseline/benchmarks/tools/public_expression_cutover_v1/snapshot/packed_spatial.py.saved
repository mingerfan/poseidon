"""Channel-first packed Conv/average pool, with optional leading batch dimension."""
import math
import spatial_ops as base
from seal_artifact_gate import require


def geometry(op,shape,weight_shape,kernel,stride,padding,include_pad=True,*,dilation=None,groups=1):
    require(op in base.OPS,'Unknown packed spatial operation')
    dims=1 if op.endswith('1d') else 2
    require(type(shape) in (tuple,list) and len(shape) in (dims+1,dims+2) and
            all(type(n) is int and 1<=n<=256 for n in shape) and math.prod(shape)<=256,
            'Packed spatial needs C/spatial or N/C/spatial layout, at most256 elements')
    batched=len(shape)==dims+2;batch=shape[0] if batched else 1
    require(batch<=16,'Spatial batch exceeds total output budget')
    sample=shape[1:] if batched else shape
    out=base.geometry(op,sample,weight_shape,kernel,stride,padding,include_pad,dilation=dilation,groups=groups,
        max_input_elements=256,max_output_elements=16//batch)
    return (batch,*out) if batched else out


def lowering(op,shape,weight,weight_shape,bias,kernel,stride,padding,include_pad=True,*,dilation=None,groups=1):
    output=geometry(op,shape,weight_shape,kernel,stride,padding,include_pad,dilation=dilation,groups=groups)
    dims=1 if op.endswith('1d') else 2;batched=len(shape)==dims+2
    batch=shape[0] if batched else 1;sample=shape[1:] if batched else shape
    matrix,biases,_=base.lowering(op,sample,weight,weight_shape,bias,kernel,stride,padding,include_pad,
        dilation=dilation,groups=groups,max_input_elements=256,max_output_elements=16//batch)
    if not batched:return matrix,biases,output
    width=math.prod(sample);rows=[]
    for b in range(batch):
        for row in matrix:
            rows.append([0.]*(b*width)+row+[0.]*((batch-b-1)*width))
    return rows,biases*batch,output


def reference(op,value,shape,weight,bias,kernel,stride,padding,include_pad=True,*,dilation=None,groups=1):
    """Direct windows on each original sample, never materialize lowering matrices."""
    dims=1 if op.endswith('1d') else 2
    if len(shape)==dims+2:
        return [base.reference(op,sample,shape[1:],weight,bias,kernel,stride,padding,include_pad,
                               dilation=dilation,groups=groups) for sample in value]
    return base.reference(op,value,shape,weight,bias,kernel,stride,padding,include_pad,dilation=dilation,groups=groups)
