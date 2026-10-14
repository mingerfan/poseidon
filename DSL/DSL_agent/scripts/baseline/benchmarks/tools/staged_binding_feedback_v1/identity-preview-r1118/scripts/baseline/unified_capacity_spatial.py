"""Public cross-correlation/pooling routes; no DSL or numerical reference imports."""
import itertools
import math
from benchmark_graph import require
from unified_capacity_routes import coordinates,flatten

OPS=("conv1d","conv2d","avg_pool1d","avg_pool2d")
MAX_ROUTE_GROUPS=1024

def geometry(node,shapes):
    op=node["op"];require(op in OPS,"Capacity spatial operation")
    rank=1 if op.endswith("1d") else 2
    source=tuple(shapes[node["inputs"][0]])
    target=tuple(shapes[node["outputs"][0]])
    batched=len(source)==rank+2
    source=source if batched else (1,*source)
    target=target if batched else (1,*target)
    a=node["attrs"];conv=op.startswith("conv")
    weight=shapes[node["inputs"][1]] if conv else None
    return dict(rank=rank,source=source,target=target,conv=conv,
                kernel=tuple(weight[2:] if conv else a["kernel"]),
                stride=tuple(a["stride"]),padding=tuple(a["padding"]),
                dilation=tuple(a["dilation"] if conv else [1]*rank),
                groups=a["groups"] if conv else 1)

def coefficient(value,coords):
    for c in coords:value=value[c]
    return float(value)

def contributions(node,shapes,constants):
    """Yield public source/destination coefficients, without evaluating input data."""
    g=geometry(node,shapes);source,target=g["source"],g["target"]
    local_channels=source[1]//g["groups"] if g["conv"] else 1
    for batch in range(target[0]):
        for channel in range(target[1]):
            first=(channel//(target[1]//g["groups"]))*local_channels if g["conv"] else channel
            for location in itertools.product(*(range(n) for n in target[2:])):
                dst=flatten((batch,channel,*location),target)
                valid=[]
                for offset in itertools.product(*(range(n) for n in g["kernel"])):
                    position=tuple(o*s-p+k*d for o,s,p,k,d in
                        zip(location,g["stride"],g["padding"],offset,g["dilation"]))
                    if all(0<=c<n for c,n in zip(position,source[2:])):valid.append((offset,position))
                divisor=math.prod(g["kernel"]) if g["conv"] or node["attrs"]["count_include_pad"] else len(valid)
                require(divisor>0,"Capacity pool window has no valid elements")
                for cin in range(local_channels):
                    for offset,position in valid:
                        factor=coefficient(constants[node["inputs"][1]],(channel,cin,*offset)) if g["conv"] else 1./divisor
                        if factor==0.:continue
                        src=flatten((batch,first+cin,*position),source)
                        yield 0,src,dst,factor

def vectors(node,shapes,constants,period):
    g=geometry(node,shapes)
    channels=g["source"][1]//g["groups"] if g["conv"] else 1
    require(math.prod(g["target"])*math.prod(g["kernel"])*channels<=MAX_ROUTE_GROUPS*period,
            "Capacity spatial map work budget")
    result={}
    for operand,src,dst,factor in contributions(node,shapes,constants):
        key=(operand,src//period,dst//period,(src-dst)%period)
        if key not in result:
            require(len(result)<MAX_ROUTE_GROUPS,"Capacity spatial route budget")
            result[key]=[0.]*period
        vector=result[key]
        require(vector[dst%period]==0.,"Capacity spatial route collision")
        vector[dst%period]=factor
    return result

def bias_vectors(node,shapes,constants,period):
    if not node["op"].startswith("conv") or len(node["inputs"])!=3:return {}
    g=geometry(node,shapes);target=g["target"];result={}
    for dst in range(math.prod(target)):
        channel=coordinates(dst,target)[1]
        result.setdefault(dst//period,[0.]*period)[dst%period]=float(constants[node["inputs"][2]][channel])
    return result


def pool_window_plan(node,shapes,constants,period):
    """Factor an unpadded average window, then gather its valid anchors.

    Intermediate flat windows may cross a row/channel boundary. None of those
    cells is selected by the anchor map; every selected window is inside its
    declared spatial domain. Chunk shifts fill out-of-range cells with zero.
    """
    if not node["op"].startswith("avg_pool"):return None
    g=geometry(node,shapes)
    if any(g["padding"]):return None
    n=math.prod(g["source"])
    require(math.prod(g["target"])*math.prod(g["kernel"])<=MAX_ROUTE_GROUPS*period,
            "Capacity spatial map work budget")
    axes=[];needed=set();stride=1
    for size,count in reversed(list(zip(g["source"][2:],g["kernel"]))):
        axes.append(dict(count=count,stride=stride))
        for bit in range(1,count.bit_length()):needed.add(stride*(1<<(bit-1)))
        offset=0
        for bit in reversed(range(count.bit_length())):
            if count&(1<<bit):
                if offset:needed.add(offset)
                offset+=stride*(1<<bit)
        stride*=size
    shifts={}
    for offset in sorted(needed):
        routes={}
        for dst in range(max(0,n-offset)):
            src=dst+offset;key=(0,src//period,dst//period,offset%period)
            routes.setdefault(key,[0.]*period)[dst%period]=1.
        shifts[offset]=routes
    anchors={};factor=1./math.prod(g["kernel"])
    for dst in range(math.prod(g["target"])):
        batch,channel,*location=coordinates(dst,g["target"])
        src=flatten((batch,channel,*(o*s for o,s in zip(location,g["stride"]))),g["source"])
        key=(0,src//period,dst//period,(src-dst)%period)
        vector=anchors.setdefault(key,[0.]*period)
        require(vector[dst%period]==0.,"Capacity pool anchor collision")
        vector[dst%period]=factor
    require(len(anchors)<=MAX_ROUTE_GROUPS,"Capacity spatial route budget")
    return dict(axes=axes,shifts=shifts,anchors=anchors)

def bsgs_vectors(routes,period):
    """Exactly the public mask transform used by the spatial emitter."""
    return {key: [vector[(i-(key[-1]-key[-1]%min(16,period)))%period]
                  for i in range(period)] for key,vector in routes.items()}

def pool_window_work(plan,period):
    """Conservative emitted-op count, including every binary rotation."""
    chunks=0
    shift_cost={}
    for offset,routes in plan["shifts"].items():
        targets={}
        cost=0
        for (_,source,target,step),vector in routes.items():
            cost+=step.bit_count()+int(not all(v==1. for v in vector))
            targets[target]=targets.get(target,0)+1
        cost+=sum(max(0,count-1) for count in targets.values())
        shift_cost[offset]=cost
        chunks=max(chunks,max((key[2]+1 for key in routes),default=0))
    cost=0
    for axis in plan["axes"]:
        count,stride=axis["count"],axis["stride"]
        for bit in range(1,count.bit_length()):
            cost+=shift_cost[stride*(1<<(bit-1))]+chunks
        offset=0;terms=0
        for bit in reversed(range(count.bit_length())):
            if count&(1<<bit):
                if offset:cost+=shift_cost[offset]
                terms+=1;offset+=stride*(1<<bit)
        cost+=max(0,terms-1)*chunks
    groups={};steps={}
    for (_,source,target,step),vector in plan["anchors"].items():
        baby=step%min(16,period);giant=step-baby
        steps.setdefault(source,set()).add(baby)
        groups.setdefault((source,target,giant),0);groups[source,target,giant]+=1
    for values in steps.values():
        previous=0
        for step in sorted(values):cost+=(step-previous).bit_count();previous=step
    cost+=sum(2*count-1+giant.bit_count()+1 for (_,_,giant),count in groups.items())
    return cost


def pool_stream_plan(node,shapes,period):
    """Sum the same kernel fragment within a source chunk, then mask anchors.

    Each stream contains only rotations/additions. Its registered gather mask
    observes cells for which every selected source index is in that chunk.
    No intermediate plaintext mask or padded/cross-channel window is observed.
    """
    if not node["op"].startswith("avg_pool"):return None
    g=geometry(node,shapes)
    if any(g["padding"]):return None
    require(math.prod(g["target"])*math.prod(g["kernel"])<=MAX_ROUTE_GROUPS*period,
            "Capacity spatial map work budget")
    streams=[];identities={};routes={};factor=1./math.prod(g["kernel"])
    for dst in range(math.prod(g["target"])):
        batch,channel,*location=coordinates(dst,g["target"]);parts={}
        for offset in itertools.product(*(range(n) for n in g["kernel"])):
            position=tuple(o*s+k for o,s,k in zip(location,g["stride"],offset))
            src=flatten((batch,channel,*position),g["source"])
            parts.setdefault(src//period,[]).append(src%period)
        for source,indices in sorted(parts.items()):
            first=min(indices);offsets=tuple(sorted(i-first for i in indices))
            identity=(source,offsets)
            if identity not in identities:
                identities[identity]=len(streams);streams.append(dict(input_chunk=source,offsets=list(offsets)))
            stream=identities[identity];key=(0,stream,dst//period,(first-dst)%period)
            vector=routes.setdefault(key,[0.]*period)
            require(vector[dst%period]==0.,"Capacity pool stream collision")
            vector[dst%period]=factor
    require(len(routes)<=MAX_ROUTE_GROUPS and len(streams)<=64,"Capacity spatial stream budget")
    return dict(streams=streams,anchors=routes)
