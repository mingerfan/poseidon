"""Multidimensional schema5 graphs with last-axis Linear and public broadcasts."""
import math


def affine(shape,broadcast_shape,tag):
    import itertools
    from spatial_ops import nested
    n=math.prod(broadcast_shape)
    gain=nested([(.25 if i%2 else -.5) for i in range(n)],broadcast_shape)
    return dict(schema=5,id='tensor-affine-'+tag,input_shape=shape,
        constants=dict(gain=gain,bias=.0625),nodes=[
            dict(id='scaled',op='multiply',inputs=['x','gain']),
            dict(id='out',op='add',inputs=['scaled','bias'])],output='out')


def linear(shape,width,tag,kind='linear'):
    weight=[[(((i+1)*(j+2))%11-5)/32 for j in range(shape[-1])] for i in range(width)]
    bias=[(i+1)/64 for i in range(width)]
    constants=dict(weight=weight,bias=bias)
    nodes=[dict(id='projected',op='linear',inputs=['x'],weight='weight',bias='bias')]
    out='projected'
    if kind in ('mlp','residual'):
        constants.update(second=[[.25 if i==j else -.125 for j in range(width)] for i in range(2)],
                         final_bias=[.0625,-.03125])
        nodes += [dict(id='activated',op='square',inputs=['projected']),
                  dict(id='out',op='linear',inputs=['activated'],weight='second',bias='final_bias')]
        out='out'
        if kind=='residual':
            assert width==2
            nodes += [dict(id='residual',op='add',inputs=['projected','out'])]
            out='residual'
    if kind=='zero_rows':constants['weight'][0]=[0.]*shape[-1]
    if kind=='reshape':
        nodes += [dict(id='restored',op='reshape',inputs=['projected'],shape=[-1,2])]
        out='restored'
    return dict(schema=5,id='tensor-'+kind+'-'+tag,input_shape=shape,constants=constants,nodes=nodes,output=out)


def cases():
    out=[affine([3,5],[5],'row'),affine([3,5],[3,1],'column'),
         affine([2,3,5],[1,3,1],'rank3'),affine([2,2,4,8],[1,2,1,1],'rank4'),
         linear([2,8],3,'rank2'),linear([2,2,8],2,'rank3'),linear([1,2,2,16],2,'rank4'),
         linear([2,7],3,'rank2-mlp','mlp'),linear([2,3,5],2,'rank3-residual','residual'),
         linear([2,8],2,'rank2-zero','zero_rows'),linear([2,2,8],2,'scalar-output','reshape')]
    out.append(dict(schema=5,id='tensor-reshape-packed-input',input_shape=[3,5],
        constants=dict(gain=[.25,-.5,.75],offset=.0625),nodes=[
            dict(id='view',op='reshape',inputs=['x'],shape=[5,-1]),
            dict(id='scaled',op='multiply',inputs=['view','gain']),
            dict(id='offsetted',op='subtract',inputs=['scaled','offset']),
            dict(id='out',op='reshape',inputs=['offsetted'],shape=[1,3,5])],output='out'))
    return out
