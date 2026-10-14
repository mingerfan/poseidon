"""Public synthetic schema-4 cases, independent of candidate programs."""
import math


def case(shape, kind='linear', width=2):
    n = math.prod(shape)
    weights = [[(((i+2)*(j+3))%13-6)/32 for j in range(n)] for i in range(width)]
    constants = dict(weight=weights, bias=[(i+1)/32 for i in range(width)])
    nodes = [dict(id='flat', op='flatten', inputs=['x'])]
    source = 'flat'
    if kind == 'fanout':
        constants.update(gain=.25, shift=.125)
        nodes += [dict(id='scaled', op='multiply', inputs=['flat','gain']),
                  dict(id='offset', op='add', inputs=['flat','shift']),
                  dict(id='combined', op='subtract', inputs=['offset','scaled'])]
        source = 'combined'
    nodes += [dict(id='projection', op='linear', inputs=[source],weight='weight',bias='bias')]
    output = 'projection'
    if kind in ('mlp','residual'):
        constants.update(second=[[.25 if i==j else -.125 for j in range(width)] for i in range(2)],
                         final_bias=[.0625,-.03125])
        nodes += [dict(id='activated',op='square',inputs=['projection']),
                  dict(id='logits',op='linear',inputs=['activated'],weight='second',bias='final_bias')]
        output = 'logits'
        if kind == 'residual':
            assert width == 2
            nodes += [dict(id='residual',op='add',inputs=['logits','projection'])]
            output = 'residual'
    if kind == 'zero_rows':
        constants['weight'][0] = [0.0]*n
    return dict(schema=4,id='chunked-'+kind+'-'+'x'.join(map(str,shape)),input_shape=list(shape),
                constants=constants,nodes=nodes,output=output)


def cases():
    return [case([5]),case([1,7]),case([2,4]),case([1,3,3]),case([2,2,3]),case([2,2,2,2]),
            case([16],'mlp',8),case([8],'fanout'),case([3,4],'residual'),case([15],'zero_rows')]
