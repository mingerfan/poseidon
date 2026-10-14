"""Static C-order logical reshape: no permutation, repacking or data access."""
import math
from seal_artifact_gate import require

def reshape_shape(source,target,*,max_elements=8):
    require(type(max_elements) is int and max_elements in (8,256),'Unknown reshape budget')
    require(type(source) in (tuple,list) and 1<=len(source)<=4 and
            all(type(d) is int and 1<=d<=max_elements for d in source) and math.prod(source)<=max_elements,
            'Invalid reshape source shape')
    require(type(target) in (tuple,list) and 1<=len(target)<=4 and
            all(type(d) is int and (d==-1 or 1<=d<=max_elements) for d in target) and
            sum(d==-1 for d in target)<=1,'Invalid static reshape target')
    size=math.prod(source)
    specified=math.prod(d for d in target if d!=-1)
    require(specified<=max_elements,'Reshape target exceeds element budget')
    if -1 in target:
        require(size%specified==0 and size//specified>=1,'Reshape inferred dimension is not integral')
        target=tuple(size//specified if d==-1 else d for d in target)
    require(math.prod(target)==size,'Reshape must preserve element count')
    return tuple(target)
