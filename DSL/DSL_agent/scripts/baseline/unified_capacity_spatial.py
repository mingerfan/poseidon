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

def _plain_spatial_metadata(node,shapes):
    pending=[node,shapes];seen=set();work=0
    while pending:
        item=pending.pop();kind=type(item)
        if (kind is str or kind is int or kind is float or kind is bool
                or kind is type(None)):
            continue
        if kind is not dict and kind is not list and kind is not tuple:
            return False
        if id(item) in seen:
            continue
        seen.add(id(item));work+=len(item)
        if work>65536:
            return False
        if kind is dict:
            if any(type(key) is not str for key in item):
                return False
            pending.extend(item.values())
        else:
            pending.extend(item)
    return True

def _plain_conv_terms(node,shapes,constants,period,g):
    """Per-call numeric preparation only; all admission/route checks remain.

    Exact builtin metadata/weights, positive geometry and bounded numeric work
    exclude callbacks and avoid reading weights for empty/invalid geometry.
    Unknown/custom/invalid data use the original lazy contribution generator.
    No validation result, request or weight matrix is cached across calls.
    """
    if (type(node) is not dict or type(shapes) is not dict
            or type(constants) is not dict or len(constants)>4096
            or type(period) is not int or not 1<=period<=16384
            or not _plain_spatial_metadata(node,shapes)
            or any(type(key) is not str for key in constants)):
        return None
    if (type(node.get("op")) is not str or node["op"] not in ("conv1d","conv2d")
            or type(node.get("inputs")) is not list or len(node["inputs"]) not in (2,3)
            or any(type(ref) is not str for ref in node["inputs"])):
        return None
    rank=g["rank"];source,target=g["source"],g["target"];kernel=g["kernel"]
    if (len(source)!=rank+2 or len(target)!=rank+2 or len(kernel)!=rank
            or any(type(n) is not int or n<=0 for n in (*source,*target,*kernel))
            or any(len(g[name])!=rank for name in ("stride","padding","dilation"))
            or any(type(n) is not int or n<=0 for name in ("stride","dilation") for n in g[name])
            or any(type(n) is not int or n<0 for n in g["padding"])
            or type(g["groups"]) is not int or g["groups"]<=0
            or source[1]%g["groups"] or target[1]%g["groups"]):
        return None
    channels=source[1]//g["groups"]
    expected=(target[1],channels,*kernel)
    if math.prod(expected)>131072 or node["inputs"][1] not in constants:
        return None
    pending=[(constants[node["inputs"][1]],0)];flat=[]
    while pending:
        value,axis=pending.pop()
        if axis==len(expected):
            if type(value) is not int and type(value) is not float:
                return None
            try:factor=float(value)
            except OverflowError:return None
            if not math.isfinite(factor):
                return None
            flat.append(factor)
        else:
            if type(value) is not list or len(value)!=expected[axis]:
                return None
            pending.extend((child,axis+1)for child in reversed(value))
    values=iter(flat);offsets=tuple(itertools.product(*(range(n)for n in kernel)))
    terms=[]
    for channel in range(target[1]):
        rows=[]
        for cin in range(channels):
            entries=[]
            for offset in offsets:
                factor=next(values)
                if factor!=0.:
                    entries.append((offset,factor))
            rows.append(tuple(entries))
        terms.append(tuple(rows))
    return tuple(terms)

def _sparse_contributions(node,shapes,terms):
    # Preserve the original second geometry pass and per-window divisor check.
    g=geometry(node,shapes);source,target=g["source"],g["target"]
    local_channels=source[1]//g["groups"]
    for batch in range(target[0]):
        for channel in range(target[1]):
            first=(channel//(target[1]//g["groups"]))*local_channels
            for location in itertools.product(*(range(n) for n in target[2:])):
                dst=flatten((batch,channel,*location),target)
                valid=[]
                for offset in itertools.product(*(range(n) for n in g["kernel"])):
                    position=tuple(o*s-p+k*d for o,s,p,k,d in
                        zip(location,g["stride"],g["padding"],offset,g["dilation"]))
                    if all(0<=c<n for c,n in zip(position,source[2:])):valid.append((offset,position))
                divisor=math.prod(g["kernel"])
                require(divisor>0,"Capacity pool window has no valid elements")
                positions=dict(valid)
                for cin,entries in enumerate(terms[channel]):
                    for offset,factor in entries:
                        position=positions.get(offset)
                        if position is None:continue
                        src=flatten((batch,first+cin,*position),source)
                        yield 0,src,dst,factor

def vectors(node,shapes,constants,period):
    g=geometry(node,shapes)
    channels=g["source"][1]//g["groups"] if g["conv"] else 1
    require(math.prod(g["target"])*math.prod(g["kernel"])*channels<=MAX_ROUTE_GROUPS*period,
            "Capacity spatial map work budget")
    terms=_plain_conv_terms(node,shapes,constants,period,g)
    routes=contributions(node,shapes,constants) if terms is None else _sparse_contributions(node,shapes,terms)
    result={}
    for operand,src,dst,factor in routes:
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
