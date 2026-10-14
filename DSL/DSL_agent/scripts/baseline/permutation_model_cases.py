"""Static axis-reordering cases; reshape is deliberately not a substitute."""
from packed_spatial_cases import conv
from packed_composition_cases import append_bn
from tensor_model_cases import linear


def permute(shape,dims,tag):
    return dict(schema=5,id='perm-'+tag,input_shape=shape,constants={},nodes=[
        dict(id='reordered',op='permute',inputs=['x'],dims=dims)],output='reordered')


def transpose(shape,a,b,tag):
    d=permute(shape,list(range(len(shape))),tag)
    d['nodes'][0]=dict(id='reordered',op='transpose',inputs=['x'],dim0=a,dim1=b)
    return d


def cases():
    rows=[transpose([2,3],0,1,'nonsquare'),permute([2,3,5],[2,0,1],'rank3'),
          permute([2,3,2,4],[0,2,3,1],'rank4'),transpose([16,16],0,1,'max256'),
          permute([2,3,4],[-1,-3,-2],'negative-axes')]
    d=transpose([3,8],0,1,'linear');d['constants']=dict(weight=[[.25,-.125,.375],[-.5,.125,.25]],bias=[.0625,-.03125])
    d['nodes'].append(dict(id='out',op='linear',inputs=['reordered'],weight='weight',bias='bias'));d['output']='out';rows.append(d)
    d=conv([1,2,4,4],2,2,[2,2],[2,2],[0,0],'nhwc');d['id']='perm-nhwc-conv';d['input_shape']=[1,4,4,2]
    d['nodes'][0]['inputs']=['reordered'];d['nodes'].insert(0,dict(id='reordered',op='permute',inputs=['x'],dims=[0,3,1,2]));rows.append(d)
    d=linear([2,4],3,'scalar');d['id']='perm-scalar-linear'
    d['nodes'].append(dict(id='out',op='transpose',inputs=['projected'],dim0=-1,dim1=-2));d['output']='out';rows.append(d)
    d=conv([2,1,4,4],2,2,[2,2],[2,2],[0,0],'scalar');d['id']='perm-scalar-conv'
    d['nodes'].append(dict(id='reordered',op='permute',inputs=['out'],dims=[0,2,3,1]));d['output']='reordered';rows.append(d)
    d=permute([2,3,4],[0,2,1],'batchnorm');append_bn(d,'reordered',4);rows.append(d)
    d=transpose([2,3],0,1,'concat');d['nodes'] += [dict(id='negative',op='negate',inputs=['reordered']),
        dict(id='out',op='concat',inputs=['reordered','negative'],axis=-1)];d['output']='out';rows.append(d)
    d=permute([2,3,4],[2,0,1],'reshape-linear');d['constants']=dict(
        weight=[[.25,-.125,.375,-.25,.0625,.125],[-.5,.25,.125,-.125,.25,.375]],bias=[.125,-.0625])
    d['nodes'] += [dict(id='reshaped',op='reshape',inputs=['reordered'],shape=[4,6]),
                  dict(id='out',op='linear',inputs=['reshaped'],weight='weight',bias='bias')]
    d['output']='out';rows.append(d)
    return rows
