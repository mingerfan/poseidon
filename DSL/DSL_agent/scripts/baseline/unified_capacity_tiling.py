"""Public row-stride permutations and local Linear maps, without DSL/reference."""
from benchmark_graph import require

def ordered_steps(steps,period):
    return sorted(steps,key=lambda s:s if s<=period//2 else s-period)

def row_stages(width,stride,batches,period,*,compress=False):
    """Binary row displacement: expand high bits first, compress in reverse."""
    delta=stride-width;bits=list(reversed(range((batches-1).bit_length())))
    if not delta:return []
    processed=0;stages=[]
    for bit in bits:
        fixed=[0.]*period;moving=[0.]*period
        for batch in range(batches):
            vector=moving if batch&(1<<bit) else fixed
            for column in range(width):
                position=batch*width+column+(batch&processed)*delta
                require(position<period,"Capacity tile permutation")
                vector[position]=1.
        shift=delta*(1<<bit)
        if compress:
            # The destination partition of expansion is the input partition of
            # compression. Both are exact public permutations of logical cells.
            moved=[0.]*period
            for i,v in enumerate(moving):
                if v:moved[i+shift]=v
            moving=moved
        stages.append(dict(shift=(shift if compress else -shift)%period,
                           fixed=fixed,moving=moving))
        processed|=1<<bit
    return list(reversed(stages)) if compress else stages

def template(weights,batches,period,*,row_sum=False):
    outputs,width=len(weights),len(weights[0])
    stride=1<<(max(outputs,width)-1).bit_length()
    require(stride*batches<=period,"Capacity Linear tile size")
    if row_sum:
        rows={}
        for output,row in enumerate(weights):
            vector=[0.]*period
            for batch in range(batches):vector[batch*stride:batch*stride+width]=row
            rows[output]=vector
        mask=[float(i%stride==0 and i//stride<batches) for i in range(period)]
        return dict(stride=stride,expand=row_stages(width,stride,batches,period),
                    row_weights=rows,row_mask=mask,
                    compress=row_stages(outputs,stride,batches,period,compress=True))
    groups={}
    for batch in range(batches):
        for output,row in enumerate(weights):
            dst=batch*stride+output
            for j,weight in enumerate(row):
                if weight==0.:continue
                step=(j-output)%period
                vector=groups.setdefault(step,[0.]*period)
                require(vector[dst]==0.,"Capacity Linear tile collision")
                vector[dst]=weight
    return dict(stride=stride,
                expand=row_stages(width,stride,batches,period),
                weights=groups,compress=row_stages(outputs,stride,batches,period,compress=True))

def template_vectors(t):
    for role in ("expand","compress"):
        for phase,stage in enumerate(t[role]):
            for side in ("fixed","moving"):
                yield role,phase,side,stage[side]
    if "row_weights" in t:
        for output,vector in sorted(t["row_weights"].items()):yield "row_weight",output,"",vector
        yield "row_mask",0,"",t["row_mask"]
    else:
        for step,vector in sorted(t["weights"].items()):yield "weight",step,"",vector

def template_cost(t,period):
    if "row_weights" in t:
        count=len(t["row_weights"]);bits=t["stride"].bit_length()-1
        cost=count*(2+2*bits)+max(0,count-1)
        cost+=sum((-output%period).bit_count() for output in t["row_weights"])
    else:
        steps=ordered_steps(t["weights"],period);previous=0;cost=0
        for step in steps:
            cost+=((step-previous)%period).bit_count();previous=step
        if steps:cost+=2*len(steps)-1
    for stage in t["expand"]+t["compress"]:
        cost+=3+stage["shift"].bit_count()
    return cost

def spans(start,length,period):
    result=[]
    for chunk in range(start//period,(start+length-1)//period+1):
        lo=max(start,chunk*period)-start
        hi=min(start+length,(chunk+1)*period)-start
        result.append(dict(chunk=chunk,lo=lo,hi=hi))
    return result

def range_vector(lo,hi,period):
    require(0<=lo<hi<=period,"Capacity Linear tile mask")
    return [float(lo<=i<hi) for i in range(period)]

def plan(weights,batches,period,alternative_cost,*,row_sum=False,depth_first=False):
    """Choose only a lower-work bounded map; final candidate checks stay authoritative."""
    outputs,width=len(weights),len(weights[0]);stride=1<<(max(outputs,width)-1).bit_length()
    maximum=min(batches,period//stride)
    if outputs==width or maximum<2:return None
    best=None
    for size in (1<<i for i in range(maximum.bit_length())):
        blocks=[];templates={};cost=0;unique=set()
        for offset in range(0,batches,size):
            count=min(size,batches-offset)
            if count not in templates:
                templates[count]=template(weights,count,period,row_sum=row_sum)
                unique.update(tuple(v) for *_,v in template_vectors(templates[count]))
            gather=spans(offset*width,count*width,period)
            scatter=spans(offset*outputs,count*outputs,period)
            for span in gather:span["step"]=(offset*width-span["chunk"]*period)%period
            for span in scatter:span["step"]=(span["chunk"]*period-offset*outputs)%period
            cost+=template_cost(templates[count],period)+1
            cost+=sum(s["step"].bit_count() for s in gather+scatter)
            cost+=len(gather)-1+len(scatter)
            for spans_ in (gather,scatter):
                if len(spans_)>1:
                    cost+=len(spans_)
                    unique.update(tuple(range_vector(s["lo"],s["hi"],period)) for s in spans_)
            blocks.append(dict(batch_offset=offset,batches=count,gather=gather,scatter=scatter))
        # Native checks charge every function's constant registry and expand
        # helper calls. Account for that existing rule in the public estimate.
        cost+=(len(blocks)+1)*len(unique)
        if cost>=min(1024,alternative_cost) or len(unique)>256:continue
        candidate=dict(tile_batches=size,stride=stride,estimated_work=cost,blocks=blocks)
        if row_sum:candidate["method"]="row_sum"
        if depth_first:
            from capacity_layout_costs import tiled_costs
            depth=tiled_costs(candidate,templates,period)["plaintext_depth_upper_bound"]
            score=(depth,cost,len(unique),size)
        else:score=(cost,len(unique),size)
        if best is None or score<best[0]:
            best=(score,candidate,templates)
    return None if best is None else (best[1],best[2])
