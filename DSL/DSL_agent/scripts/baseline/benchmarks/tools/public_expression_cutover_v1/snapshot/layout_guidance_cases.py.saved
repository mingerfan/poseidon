"""Two prior failures plus four new layout compositions; no reference sent to LLM."""
from permutation_model_cases import cases as prior,permute
from packed_spatial_cases import conv
from packed_composition_cases import append_bn


def cases():
    old=prior();rows=[old[4],old[6]]
    rows.append(permute([2,2,3,4],[-1,0,-2,1],'v2-negative-rank4'))
    d=conv([1,2,4,4],2,2,[2,2],[2,2],[0,0],'grouped-layout',groups=2)
    d['id']='perm-v2-nhwc-grouped';d['input_shape']=[1,4,4,2]
    d['nodes'][0]['inputs']=['reordered']
    d['nodes'].insert(0,dict(id='reordered',op='permute',inputs=['x'],dims=[0,3,1,2]));rows.append(d)
    d=permute([2,3,2,2],[0,2,3,1],'v2-nhwc-linear')
    d['constants']=dict(weight=[[.25,-.125,.375],[-.5,.125,.25]],bias=[.0625,-.03125])
    d['nodes'].append(dict(id='out',op='linear',inputs=['reordered'],weight='weight',bias='bias'));d['output']='out';rows.append(d)
    d=permute([2,3,4],[2,0,1],'v2-two-permutations-bn')
    d['nodes'].append(dict(id='second',op='permute',inputs=['reordered'],dims=[1,0,2]))
    append_bn(d,'second',4);rows.append(d)
    return rows
