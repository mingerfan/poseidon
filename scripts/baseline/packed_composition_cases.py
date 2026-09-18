"""Public normalization and branch-concatenation models; fixed inference only."""
from packed_spatial_cases import conv


def append_bn(d,source,channels,*,affine=True,zero=False):
    d['constants'].update(mean=[(i-1)/8 for i in range(channels)],
        variance=[.5+i/4 for i in range(channels)],
        gamma=[0. if zero else (-.75 if i%2 else .5) for i in range(channels)],
        beta=[(i+1)/32 for i in range(channels)])
    if not affine:
        del d['constants']['gamma'];del d['constants']['beta']
    d['nodes'].append(dict(id='normalized',op='batch_norm',inputs=[source],
        running_mean='mean',running_var='variance',weight='gamma' if affine else None,
        bias='beta' if affine else None,eps=.125))
    d['output']='normalized'
    return d


def bn(shape,tag,**options):
    return append_bn(dict(schema=5,id='pc-'+tag,input_shape=shape,constants={},nodes=[],output='normalized'),
                     'x',shape[1],**options)


def cases():
    rows=[bn([3,5],'bn-rank2'),bn([2,3,5],'bn-rank3'),bn([2,2,4,8],'bn-rank4'),
          bn([2,4,4,8],'bn-max256'),bn([2,3,3],'bn-no-affine',affine=False),
          bn([2,3,5],'bn-zero-gamma',zero=True)]
    d=conv([2,1,4,4],2,2,[2,2],[2,2],[0,0],'conv-bn');d['id']='pc-conv-bn'
    append_bn(d,'out',2);d['constants']['gamma'][0]=0.;rows.append(d)
    d=conv([2,1,8,16],2,1,[3,3],[4,4],[1,1],'bn-conv');d['id']='pc-bn-conv-max256'
    node=d['nodes'].pop();append_bn(d,'x',1);node['inputs']=['normalized'];d['nodes'].append(node);d['output']='out';rows.append(d)
    rows.append(dict(schema=5,id='pc-concat-packed-negative-axis',input_shape=[2,3],constants={},nodes=[
        dict(id='negative',op='negate',inputs=['x']),dict(id='out',op='concat',inputs=['x','negative'],axis=-1)],output='out'))
    d=conv([2,1,4,4],2,1,[2,2],[2,2],[0,0],'concat-conv');d['id']='pc-concat-conv-channels'
    d['nodes'] += [dict(id='negative',op='negate',inputs=['out']),
                   dict(id='joined',op='concat',inputs=['negative','out'],axis=1)]
    d['output']='joined';rows.append(d)
    rows.append(dict(schema=5,id='pc-concat-mixed-layout',input_shape=[2,4],constants=dict(
        weight=[[((i+j)%4+1)/32 for j in range(4)] for i in range(4)],bias=[.125,-.0625,.03125,-.125]),nodes=[
        dict(id='projected',op='linear',inputs=['x'],weight='weight',bias='bias'),
        dict(id='out',op='concat',inputs=['projected','x'],axis=0)],output='out'))
    d=bn([2,3],'bn-concat-linear')
    d['constants'].update(weight=[[(-1 if i%2 else 1)*(i+1)/32 for i in range(6)],
                                    [(.25 if i<3 else -.125) for i in range(6)]],bias=[.125,-.0625])
    d['nodes'] += [dict(id='negative',op='negate',inputs=['normalized']),
                   dict(id='joined',op='concat',inputs=['normalized','negative'],axis=1),
                   dict(id='out',op='linear',inputs=['joined'],weight='weight',bias='bias')]
    d['output']='out';rows.append(d)
    return rows
