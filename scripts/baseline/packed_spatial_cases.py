"""Public Conv/average-pool graphs for the variable-period input ABI."""
import copy
import math
from spatial_ops import nested


def conv(shape,dims,out,kernel,stride,padding,tag,*,groups=1,dilation=None):
    channels=shape[-dims-1]
    wshape=(out,channels//groups,*kernel)
    weights=nested([((i*3)%11-5)/32 for i in range(math.prod(wshape))],wshape)
    return dict(schema=5,id='ps-'+tag,input_shape=shape,constants=dict(weight=weights,bias=[(i+1)/64 for i in range(out)]),
        nodes=[dict(id='out',op='conv'+str(dims)+'d',inputs=['x'],weight='weight',bias='bias',
                    stride=stride,padding=padding,groups=groups,dilation=dilation or [1]*dims)],output='out')


def pool(shape,dims,kernel,stride,padding,include,tag):
    return dict(schema=5,id='ps-'+tag,input_shape=shape,constants={},nodes=[dict(id='out',
        op='avg_pool'+str(dims)+'d',inputs=['x'],kernel=kernel,stride=stride,padding=padding,
        count_include_pad=include)],output='out')


def cases():
    rows=[conv([1,17],1,1,[3],[2],[1],'conv1d17'),
          conv([2,9],1,2,[3],[2],[1],'grouped1d',groups=2),
          conv([1,17],1,1,[3],[3],[1],'dilated1d',dilation=[2]),
          conv([1,5,5],2,1,[2,2],[2,2],[0,0],'dilated2d',dilation=[2,2]),
          conv([2,4,4],2,4,[2,2],[2,2],[0,0],'depthwise-multiplier',groups=2),
          conv([2,1,9],1,1,[3],[2],[1],'batch-conv1d'),
          conv([2,1,4,4],2,1,[2,2],[2,2],[0,0],'batch-conv2d'),
          pool([1,9],1,[3],[2],[1],True,'avg1d-include'),
          pool([1,9],1,[3],[2],[1],False,'avg1d-exclude'),
          pool([2,1,4,4],2,[3,3],[2,2],[1,1],True,'batch-avg2d-include')]
    cnn=conv([1,5,5],2,1,[3,3],[1,1],[0,0],'conv-square-pool-linear')
    cnn['constants'].update(final_weight=[[.25],[-.5]],final_bias=[.03125,-.0625])
    cnn['nodes'] += [dict(id='act',op='square',inputs=['out']),dict(id='pool',op='avg_pool2d',inputs=['act'],
        kernel=[2,2],stride=[2,2],padding=[0,0],count_include_pad=False),dict(id='flat',op='flatten',inputs=['pool']),
        dict(id='logits',op='linear',inputs=['flat'],weight='final_weight',bias='final_bias')]
    cnn['output']='logits';rows.append(cnn)
    residual=conv([1,4,4],2,1,[1,1],[1,1],[0,0],'residual-conv')
    residual['constants'].update(other_weight=[[[[.375]]]],other_bias=[-.03125])
    branch=copy.deepcopy(residual['nodes'][0]);branch.update(id='branch',weight='other_weight',bias='other_bias')
    residual['nodes'] += [branch,dict(id='sum',op='add',inputs=['out','branch'])];residual['output']='sum';rows.append(residual)
    zero=conv([1,7],1,2,[3],[2],[0],'zero-conv-row');zero['constants']['weight'][0]=[[0.,0.,0.]];rows.append(zero)
    rows.append(pool([2,1,4,4],2,[3,3],[2,2],[1,1],False,'batch-avg2d-exclude'))
    return rows


def large_cases():
    return [conv([2,1,8,16],2,1,[3,3],[4,4],[1,1],'max256-batched'),
            conv([2,8,16],2,2,[3,3],[4,4],[1,1],'max256-depthwise',groups=2)]


def paid_cases():return cases()+large_cases()
