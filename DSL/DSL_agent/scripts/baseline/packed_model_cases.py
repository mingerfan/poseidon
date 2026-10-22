"""Synthetic schema5 cases. Fixed weights are public, never test-fitted."""
import math

from chunked_model_cases import case as small_case


def case(shape, kind='linear', width=2):
    if kind == 'affine':
        n=math.prod(shape)
        return dict(schema=5,id='packed-affine-'+'x'.join(map(str,shape)),input_shape=list(shape),
            constants=dict(gain=[(.5 if i%2 else -.25) for i in range(n)],bias=.125),
            nodes=[dict(id='flat',op='flatten',inputs=['x']),
                   dict(id='scaled',op='multiply',inputs=['flat','gain']),
                   dict(id='output',op='add',inputs=['scaled','bias'])],output='output')
    result=small_case(shape,kind,width)
    result.update(schema=5,id=result['id'].replace('chunked-','packed-'))
    return result


def cases():
    return [case([17]),case([1,31],width=4),case([2,2,8],width=8),
            case([1,3,3,7]),case([2,2,4,4],width=16),case([127]),
            case([8,16],'mlp',8),case([3,5,17],'fanout'),case([4,4,4,4],width=4),
            case([16,16],'affine'),case([8,8],'residual'),case([1,255],'zero_rows')]
