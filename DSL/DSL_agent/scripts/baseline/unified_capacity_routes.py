"""Static C-order public maps for joins/reductions; independent of DSL/reference."""
import math
from benchmark_graph import require

OPS=("concat","stack","sum","mean","slice","transpose","permute")

def coordinates(index,shape):
    result=[0]*len(shape)
    for axis in reversed(range(len(shape))):
        result[axis]=index%shape[axis];index//=shape[axis]
    require(index==0,"Capacity public index")
    return result

def flatten(coords,shape):
    require(len(coords)==len(shape),"Capacity public index rank")
    index=0
    for c,n in zip(coords,shape):
        require(0<=c<n,"Capacity public index bound");index=index*n+c
    return index

def contributions(node,shapes):
    op=node["op"];refs=node["inputs"];a=node["attrs"]
    target=shapes[node["outputs"][0]]
    require(op in OPS,"Capacity routing operation")
    if op in ("transpose","permute"):
        source=shapes[refs[0]];dims=list(range(len(source)))
        if op=="transpose":
            i,j=a["dim0"]%len(source),a["dim1"]%len(source)
            dims[i],dims[j]=dims[j],dims[i]
        else:dims=[i%len(source) for i in a["dims"]]
        require(sorted(dims)==list(range(len(source))) and tuple(source[i] for i in dims)==tuple(target),
                "Capacity permutation binding")
        for dst in range(math.prod(target)):
            coord=coordinates(dst,target);src=[0]*len(source)
            for i,axis in enumerate(dims):src[axis]=coord[i]
            yield 0,flatten(src,source),dst,1.
    elif op=="slice":
        source=shapes[refs[0]];axis=a["axis"]%len(source)
        indices=range(*slice(a["start"],a["stop"],a["step"]).indices(source[axis]))
        for dst in range(math.prod(target)):
            coord=coordinates(dst,target);coord[axis]=indices[coord[axis]]
            yield 0,flatten(coord,source),dst,1.
    elif op in ("sum","mean"):
        source=shapes[refs[0]];axes={v%len(source) for v in a["axes"]}
        factor=1./math.prod(source[v] for v in axes) if op=="mean" else 1.
        for src in range(math.prod(source)):
            coord=coordinates(src,source)
            out=[0 if i in axes else c for i,c in enumerate(coord)] if a["keepdims"] else [
                 c for i,c in enumerate(coord) if i not in axes]
            yield 0,src,flatten(out or [0],target),factor
    else:
        axis=a["axis"]%len(target);offset=0
        for operand,ref in enumerate(refs):
            shape=shapes[ref]
            for src in range(math.prod(shape)):
                coord=coordinates(src,shape)
                if op=="stack":coord.insert(axis,operand)
                else:coord[axis]+=offset
                yield operand,src,flatten(coord,target),1.
            if op=="concat":offset+=shape[axis]

def vectors(node,shapes,period):
    groups={}
    for operand,src,dst,factor in contributions(node,shapes):
        key=(operand,src//period,dst//period,(src-dst)%period)
        vector=groups.setdefault(key,[0.]*period)
        require(vector[dst%period]==0.,"Capacity public route collision")
        vector[dst%period]=factor
    return groups

def groups_by_vector(routes):
    groups={}
    for (operand,source,target,step),vector in routes.items():
        groups.setdefault((operand,source,target,tuple(vector)),[]).append(step)
    return {key:sorted(steps) for key,steps in groups.items()}

def runs(steps):
    require(steps==sorted(set(steps)) and steps,"Capacity rotation sum steps")
    result=[];start=previous=steps[0]
    for step in steps[1:]:
        if step!=previous+1:result.append((start,previous-start+1));start=step
        previous=step
    result.append((start,previous-start+1))
    return result

def work_estimate(routes,period):
    cost=len({tuple(vector) for vector in routes.values()})
    for (_,_,_,vector),steps in groups_by_vector(routes).items():
        segments=runs(steps)
        for start,count in segments:
            offset=0
            cost+=start.bit_count()+2*(count.bit_length()-1)+count.bit_count()-1
            for bit in reversed(range(count.bit_length())):
                if count&(1<<bit):
                    cost+=offset.bit_count();offset+=1<<bit
        cost+=len(segments)-1+int(not all(v==1. for v in vector))+1
    return cost

def suffix_spec(node,shapes):
    if node["op"] not in ("sum","mean"):return None
    shape=shapes[node["inputs"][0]];axes={a%len(shape) for a in node["attrs"]["axes"]}
    if axes!=set(range(min(axes),len(shape))):return None
    width=math.prod(shape[a] for a in axes);batches=math.prod(shape)//width
    factor=1./width if node["op"]=="mean" else 1.
    return [[factor]*width],batches

def tile_dimensions(model,node,shapes):
    if node["op"]=="linear":
        weights=model["constants"][node["inputs"][1]]
        return len(weights[0]),len(weights)
    if node["op"]=="stack":return 1,len(node["inputs"])
    spec=suffix_spec(node,shapes)
    require(spec is not None,"Capacity tiled reduction axes")
    return len(spec[0][0]),1

def aligned_suffix_plan(node,shapes,period,*,allow_complete_rows=False):
    """Sum disjoint power-of-two suffix rows, then gather only row starts.

    Flat rotation windows can cross row boundaries at unobserved positions.
    Every selected row start observes exactly its complete declared suffix.
    This is a public layout plan; no model inputs or reference are evaluated.
    """
    if node["op"] not in ("sum","mean"):return None
    source=shapes[node["inputs"][0]]
    axes={axis%len(source) for axis in node["attrs"]["axes"]}
    if axes!=set(range(min(axes),len(source))):return None
    width=math.prod(source[axis] for axis in axes)
    if width>period:return None
    rows=math.prod(source)//width
    if allow_complete_rows:
        # Rolling windows at these row starts must never wrap into the same
        # ciphertext when the logical row actually continues in another one.
        if any((row*width)//period!=(row*width+width-1)//period for row in range(rows)):return None
    elif width&(width-1) or period%width:return None
    factor=1./width if node["op"]=="mean" else 1.
    routes={}
    for row in range(rows):
        src=row*width;dst=row
        key=(0,src//period,dst//period,(src-dst)%period)
        vector=routes.setdefault(key,[0.]*period)
        require(vector[dst%period]==0.,"Capacity suffix gather collision")
        vector[dst%period]=factor
    return dict(width=width,rows=rows,routes=routes)
