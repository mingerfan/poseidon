"""Bounded multi-ciphertext baseline; never computes reference or rewrites Agent DSL."""
import math
import numpy as np
from benchmark_graph import require,validate
from unified_graph_contract import validate_request,validate_candidate
from unified_capacity_layout import ABIS
from unified_capacity_constants import packing,key_for,request_version,CHANNEL_VERSION,INDEX_VERSION
from unified_graph_lowering import Emitter,Scalar,PackedPrefix,packed_elementwise

class PublicVector:
    def __init__(self,name,values):
        self.name=name;self.zero=all(v==0. for v in values)
        self.one=all(v==1. for v in values)

def public_vector(em,packed,node_id,role,**indices):
    key=key_for(packed,node_id,role,**indices)
    return PublicVector(key,packed["vectors"][key])

def vector_arithmetic(em,op,a,b):
    if isinstance(a,PublicVector):
        require(isinstance(b,Scalar),"Capacity arithmetic needs ciphertext")
        if op=="add":return vector_arithmetic(em,"add",b,a)
        if op=="multiply":return vector_arithmetic(em,"multiply",b,a)
        return vector_arithmetic(em,"add",-b,a)
    require(isinstance(a,Scalar),"Capacity arithmetic needs ciphertext")
    if isinstance(b,PublicVector):
        if b.zero:return em.zero if op=="multiply" else a
        if op=="multiply" and b.one:return a
        symbol={"add":"+","subtract":"-","multiply":"*"}[op]
        return em.emit(a.name+" "+symbol+" "+b.name)
    return a+b if op=="add" else a-b if op=="subtract" else a*b

def all_slot_sum(em,x):
    from unified_capacity_layout import EXTENDED_ABIS
    if em.request["layout"]["execution_abi"] in EXTENDED_ABIS:
        # Retain every primitive operation and charge its expanded cost at each call.
        # A shared native body avoids sixteen copies exhausting the source AST budget.
        cache=getattr(em,"capacity_helpers",None)
        if cache is None:cache=em.capacity_helpers={}
        key=("period_sum",em.period)
        if key not in cache:
            name="cap_period_sum_"+str(em.period)
            local=Emitter(em.request);value=Scalar(local,"a0")
            for i in range(em.period.bit_length()-1):
                value=value+local.rotate(value,1<<i)
            body='@hc.func("c")\ndef '+name+'(a0):\n'+"\n".join(local.lines)+'\n    return '+value.name+'\n'
            cache[key]=(name,body)
        return em.emit(cache[key][0]+"("+x.name+")")
    for i in range(em.period.bit_length()-1):
        x=x+em.rotate(x,1<<i)
    return x

def tile_helper(em,packed,node,count):
    """Real typed native helper; work is expanded by the unchanged checker."""
    from unified_capacity_tiling import ordered_steps,row_stages
    cache=getattr(em,"capacity_helpers",None)
    if cache is None:cache=em.capacity_helpers={}
    identity=(node["id"],count)
    if identity in cache:return cache[identity][0]
    name="cap_tile_"+str(len(cache));local=Emitter(em.request)
    local.zero=Scalar(local,"z0",True);value=Scalar(local,"a0")
    from unified_capacity_routes import tile_dimensions
    widths=tile_dimensions(em.request["model"],node,validate(em.request["model"])["shapes"])
    stride=packed["linear_tiles"][node["id"]]["stride"]
    def permute(value,role,width):
        for phase,stage in enumerate(row_stages(width,stride,count,em.period,compress=role=="compress")):
            fixed=public_vector(local,packed,node["id"],"linear_tile_"+role,
                                tile_batches=count,phase=phase,side="fixed")
            moving=public_vector(local,packed,node["id"],"linear_tile_"+role,
                                 tile_batches=count,phase=phase,side="moving")
            stay=vector_arithmetic(local,"multiply",value,fixed)
            move=local.rotate(vector_arithmetic(local,"multiply",value,moving),stage["shift"])
            value=stay+move
        return value
    value=permute(value,"expand",widths[0])
    if packed["linear_tiles"][node["id"]].get("method")=="row_sum":
        result=local.zero
        mask=public_vector(local,packed,node["id"],"linear_tile_row_mask",
                           tile_batches=count,phase=0,side="")
        for output in range(widths[1]):
            weight=public_vector(local,packed,node["id"],"linear_tile_row_weight",
                                 tile_batches=count,phase=output,side="")
            term=vector_arithmetic(local,"multiply",value,weight)
            for bit in range(stride.bit_length()-1):term=term+local.rotate(term,1<<bit)
            term=vector_arithmetic(local,"multiply",term,mask)
            result=result+local.rotate(term,-output)
    else:
        view=value;result=local.zero;previous=0
        entries={b["phase"]:b for b in packed["bindings"] if b["node_id"]==node["id"] and
                 b["role"]=="linear_tile_weight" and b["tile_batches"]==count}
        for step in ordered_steps(entries,em.period):
            view=local.rotate(view,step-previous);previous=step
            weight=public_vector(local,packed,node["id"],"linear_tile_weight",tile_batches=count,phase=step,side="")
            result=result+vector_arithmetic(local,"multiply",view,weight)
    result=permute(result,"compress",widths[1])
    source='@hc.func("c,c")\ndef '+name+'(a0, z0):\n'
    source+="\n".join(local.lines)+"\n    return "+result.name+"\n"
    # Public registry names and ciphertext formal arguments are part of this
    # exact comparison. Reusing a body does not merge its independent calls.
    body=source.split("\n",2)[2]
    for previous_name,previous_source in cache.values():
        if previous_source.split("\n",2)[2]==body:
            return previous_name
    cache[identity]=(name,source)
    require(len(cache)<=16,"Capacity native helper count")
    return name

def window_sum(em,value,count):
    require(1<=count<=em.period,"Capacity rotation sum window")
    powers=[value]
    for bit in range(1,count.bit_length()):
        powers.append(powers[-1]+em.rotate(powers[-1],1<<(bit-1)))
    result=em.zero;offset=0
    for bit in reversed(range(count.bit_length())):
        if count&(1<<bit):
            result=result+em.rotate(powers[bit],offset);offset+=1<<bit
    return result

def accumulate_rotated(em,entries):
    """Sum the same public rotated terms using successive differences."""
    buckets={}
    for step,value in entries:
        require(type(step) is int and 0<=step<em.period,"Capacity rotation accumulation step")
        buckets[step]=buckets[step]+value if step in buckets else value
    if not buckets:return em.zero
    steps=sorted(buckets);previous=steps[-1];result=buckets[previous]
    for step in reversed(steps[:-1]):
        result=buckets[step]+em.rotate(result,previous-step);previous=step
    return em.rotate(result,previous)

def spatial_tensor(em,node,packed,operand,outshape,role="tensor_route",source_parts=None):
    """Baby/giant rotations with public pre-rotated masks; one plaintext product per route."""
    groups={};required={}
    for b in packed["bindings"]:
        if b["node_id"]==node["id"] and b["role"]==role:
            source=b.get("stream",b["input_chunk"])
            groups.setdefault((source,b["output_chunk"],b["giant_step"]),[]).append(b)
            required.setdefault(source,set()).add(b["baby_step"])
    views={};sources=operand.parts if source_parts is None else source_parts
    for source,steps in sorted(required.items()):
        value=sources[source];previous=0
        for step in sorted(steps):
            value=em.rotate(value,step-previous);previous=step
            views[source,step]=value
    def total(terms):
        if len(terms)==1:return terms[0]
        mid=len(terms)//2
        return "("+total(terms[:mid])+" + "+total(terms[mid:])+")"
    parts=[em.zero]*math.ceil(math.prod(outshape)/em.period)
    from unified_capacity_layout import WIDE_ABI
    wide=em.request["layout"]["execution_abi"]==WIDE_ABI
    deferred={}
    for (source,target,giant),entries in sorted(groups.items()):
        terms=[views[source,b["baby_step"]].name+" * "+b["key"]
               for b in sorted(entries,key=lambda b:b["baby_step"])]
        value=em.emit(total(terms))
        if wide:deferred.setdefault(target,[]).append((giant,value))
        else:parts[target]=parts[target]+em.rotate(value,giant)
    if wide:
        for target,entries in deferred.items():parts[target]=accumulate_rotated(em,entries)
    return Chunks(em,parts,outshape)


def spatial_window(em,node,packed,operand,outshape):
    """Apply flat binary windows; only valid spatial anchors enter the output."""
    plan=packed["spatial_windows"][node["id"]]
    def shifted(value,offset):
        if offset==0:return value
        parts=[em.zero]*len(value.parts)
        for b in packed["bindings"]:
            if b["node_id"]!=node["id"] or b["role"]!="pool_shift" or b["offset"]!=offset:continue
            view=em.rotate(value.parts[b["input_chunk"]],b["step"])
            term=vector_arithmetic(em,"multiply",view,PublicVector(b["key"],packed["vectors"][b["key"]]))
            target=b["output_chunk"];parts[target]=parts[target]+term
        return Chunks(em,parts,value.shape)
    def added(a,b):return Chunks(em,[x+y for x,y in zip(a.parts,b.parts)],a.shape)
    value=operand
    for axis in plan["axes"]:
        count,stride=axis["count"],axis["stride"];powers=[value]
        for bit in range(1,count.bit_length()):
            powers.append(added(powers[-1],shifted(powers[-1],stride*(1<<(bit-1)))))
        result=Chunks(em,[em.zero]*len(value.parts),value.shape);offset=0
        for bit in reversed(range(count.bit_length())):
            if count&(1<<bit):
                result=added(result,shifted(powers[bit],offset));offset+=stride*(1<<bit)
        value=result
    return spatial_tensor(em,node,packed,value,outshape,role="pool_gather")


def spatial_stream(em,node,packed,operand,outshape):
    from unified_capacity_routes import runs
    streams=[];windows={}
    for spec in packed["spatial_streams"][node["id"]]:
        source=spec["input_chunk"];value=em.zero
        for start,count in runs(spec["offsets"]):
            key=(source,count)
            if key not in windows:windows[key]=window_sum(em,operand.parts[source],count)
            value=value+em.rotate(windows[key],start)
        streams.append(value)
    return spatial_tensor(em,node,packed,operand,outshape,role="pool_stream_gather",source_parts=streams)


def route_tensor(em,node,packed,operands,outshape):
    from unified_capacity_routes import runs
    require(all(isinstance(x,Chunks) for x in operands),"Capacity routing requires cipher expressions")
    groups={}
    for b in packed["bindings"]:
        if b["node_id"]==node["id"] and b["role"]=="tensor_route":
            groups.setdefault((b["operand"],b["input_chunk"],b["output_chunk"],b["key"]),[]).append(b["step"])
    require(groups or packed["tensor_strategies"].get(node["id"])=="spatial_direct",
            "Capacity tensor route binding")
    parts=[em.zero]*math.ceil(math.prod(outshape)/em.period)
    views={}
    if packed["tensor_strategies"].get(node["id"])=="spatial_direct":
        starts={}
        for (operand,source,target,key),steps in groups.items():
            starts.setdefault((operand,source),set()).update(start for start,count in runs(sorted(steps)))
        for (operand,source),steps in sorted(starts.items()):
            value=operands[operand].parts[source];previous=0
            for step in sorted(steps):
                value=em.rotate(value,step-previous);previous=step
                views[operand,source,step]=value
    for (operand,source,target,key),steps in sorted(groups.items()):
        value=em.zero
        for start,count in runs(sorted(steps)):
            view=views[operand,source,start] if views else em.rotate(operands[operand].parts[source],start)
            value=value+window_sum(em,view,count)
        term=vector_arithmetic(em,"multiply",value,PublicVector(key,packed["vectors"][key]))
        parts[target]=parts[target]+term
    return Chunks(em,parts,outshape)


def split_tensor(em, value, node, shapes):
    """Route static split outputs; contiguous chunks may keep unspecified tails."""
    from unified_capacity_routes import coordinates, flatten
    require(isinstance(value, Chunks), "Capacity split requires ciphertext input")
    axis = node["attrs"]["axis"] % len(value.shape)
    offset = 0
    result = {}
    for name in node["outputs"]:
        shape = shapes[name]
        parts = [em.zero] * math.ceil(math.prod(shape) / em.period)
        for chunk in range(len(parts)):
            start = chunk * em.period
            count = min(em.period, math.prod(shape) - start)
            sources = []
            for destination in range(start, start + count):
                coord = coordinates(destination, shape)
                coord[axis] += offset
                sources.append(flatten(coord, value.shape))
            first = sources[0]
            if (sources == list(range(first, first + count))
                    and first // em.period == sources[-1] // em.period):
                # Logical prefixes are exact. Tail slots remain unspecified;
                # reductions/scalar broadcasts apply their existing bound masks.
                parts[chunk] = em.rotate(value.parts[first // em.period], first % em.period)
                continue
            for slot, source in enumerate(sources):
                term = em.rotate(value.parts[source // em.period], source % em.period)
                term = em.emit(term.name + " * mask0")
                term = em.rotate(term, -slot)
                parts[chunk] = parts[chunk] + term
        result[name] = Chunks(em, parts, shape)
        offset += shape[axis]
    require(offset == value.shape[axis], "Capacity split complete axis")
    return result

class Chunks:
    def __init__(self,em,parts,shape,replicated=False):
        self.em=em;self.parts=tuple(parts);self.shape=tuple(shape);self.replicated=replicated
        require(len(parts)==math.ceil(math.prod(shape)/em.period),"Capacity expression chunk count")
        require(not replicated or math.prod(shape)==1,"Capacity replicated scalar")
    def broadcast_scalar(self):
        require(math.prod(self.shape)==1,"Capacity scalar broadcast")
        if self.replicated:return self.parts[0]
        return all_slot_sum(self.em,self.em.emit(self.parts[0].name+" * mask0"))
    def reduction(self,op,attrs,outshape):
        axes={i%len(self.shape) for i in attrs["axes"]}
        require(axes==set(range(len(self.shape))),"No bounded capacity baseline for partial-axis reduction")
        n=math.prod(self.shape);result=self.em.zero
        for j,x in enumerate(self.parts):
            count=min(self.em.period,n-j*self.em.period)
            if count<self.em.period:x=self.em.emit(x.name+" * tail"+str(count))
            result=result+all_slot_sum(self.em,x)
        if op=="mean":result=result/n
        return Chunks(self.em,[result],outshape,replicated=True)
    def linear(self,node,packed,outshape):
        em=self.em;outcount=math.prod(outshape);parts=[em.zero]*math.ceil(outcount/em.period)
        if packed["linear_strategies"].get(node["id"]) in ("tiled_diagonal","tiled_row_sum"):
            for block in packed["linear_tiles"][node["id"]]["blocks"]:
                gathered=em.zero
                for span in block["gather"]:
                    value=em.rotate(self.parts[span["chunk"]],span["step"])
                    if len(block["gather"])>1:
                        mask=public_vector(em,packed,node["id"],"linear_tile_gather",
                             batch_offset=block["batch_offset"],chunk=span["chunk"])
                        value=vector_arithmetic(em,"multiply",value,mask)
                    gathered=gathered+value
                name=tile_helper(em,packed,node,block["batches"])
                value=em.emit(name+"("+gathered.name+", "+em.zero.name+")")
                for span in block["scatter"]:
                    part=value
                    if len(block["scatter"])>1:
                        mask=public_vector(em,packed,node["id"],"linear_tile_scatter",
                             batch_offset=block["batch_offset"],chunk=span["chunk"])
                        part=vector_arithmetic(em,"multiply",part,mask)
                    part=em.rotate(part,span["step"])
                    parts[span["chunk"]]=parts[span["chunk"]]+part
            if node["op"]=="linear" and len(node["inputs"])==3:
                parts=[vector_arithmetic(em,"add",x,public_vector(em,packed,node["id"],"linear_bias",chunk=i))
                       for i,x in enumerate(parts)]
            return Chunks(em,parts,outshape)
        if packed["linear_strategies"].get(node["id"])=="diagonal":
            for b in packed["bindings"]:
                if b["node_id"]!=node["id"] or b["role"]!="linear_diagonal":continue
                w=public_vector(em,packed,node["id"],"linear_diagonal",
                     input_chunk=b["input_chunk"],output_chunk=b["output_chunk"],step=b["step"])
                value=vector_arithmetic(em,"multiply",em.rotate(self.parts[b["input_chunk"]],b["step"]),w)
                parts[b["output_chunk"]]=parts[b["output_chunk"]]+value
            if node["op"]=="linear" and len(node["inputs"])==3:
                parts=[vector_arithmetic(em,"add",x,public_vector(em,packed,node["id"],"linear_bias",chunk=i))
                       for i,x in enumerate(parts)]
            return Chunks(em,parts,outshape)
        weights={}
        for b in packed["bindings"]:
            if b["node_id"]==node["id"] and b["role"]=="linear_weight":
                weights.setdefault(b["output"],[]).append(b["chunk"])
        require(set(weights)==set(range(outcount)),"Capacity Linear output binding")
        for output in range(outcount):
            total=em.zero
            for chunk in weights[output]:
                w=public_vector(em,packed,node["id"],"linear_weight",output=output,chunk=chunk)
                if w.zero:continue
                term=vector_arithmetic(em,"multiply",self.parts[chunk],w)
                total=total+all_slot_sum(em,term)
            if not total.zero:
                single=em.emit(total.name+" * mask0")
                dst=output//em.period
                parts[dst]=parts[dst]+em.rotate(single,-(output%em.period))
        if node["op"]=="linear" and len(node["inputs"])==3:
            parts=[vector_arithmetic(em,"add",x,public_vector(em,packed,node["id"],"linear_bias",chunk=i))
                   for i,x in enumerate(parts)]
        return Chunks(em,parts,outshape)

def broadcast_cipher(value,shape):
    """Static C-order broadcast using the existing bound mask0 and rotations."""
    from unified_capacity_routes import coordinates,flatten
    em=value.em
    require(len(value.shape)<=len(shape),"Capacity cipher broadcast rank")
    padded=(1,)*(len(shape)-len(value.shape))+value.shape
    require(all(n==1 or n==m for n,m in zip(padded,shape)),"Capacity cipher broadcast shape")
    parts=[None]*math.ceil(math.prod(shape)/em.period)
    cells={}
    for dst in range(math.prod(shape)):
        coord=coordinates(dst,shape)
        source_coord=[0 if n==1 else c for n,c in zip(padded,coord)]
        src=flatten(source_coord,padded)
        if src not in cells:
            cell=em.rotate(value.parts[src//em.period],src%em.period)
            cells[src]=em.emit(cell.name+" * mask0")
        chunk,slot=divmod(dst,em.period)
        term=em.rotate(cells[src],(-slot)%em.period)
        parts[chunk]=term if parts[chunk] is None else parts[chunk]+term
    require(all(part is not None for part in parts),"Capacity broadcast output chunks")
    return parts

def aligned_operand(value,shape,parts,em,packed,node,operand):
    if isinstance(value,Chunks):
        if value.shape==tuple(shape) or (math.prod(value.shape)==math.prod(shape) and
                np.broadcast_shapes(value.shape,tuple(shape))==tuple(shape)):return value.parts
        if math.prod(value.shape)==1:return (value.broadcast_scalar(),)*parts
        return broadcast_cipher(value,shape)
    if isinstance(value,np.ndarray) and value.size:
        if np.all(value==value.reshape(-1)[0]):return (float(value.reshape(-1)[0]),)*parts
        return tuple(public_vector(em,packed,node["id"],"elementwise",operand=operand,chunk=j) for j in range(parts))
    raise ValueError("No bounded capacity public operand")


def compact_scalar_source(source):
    """Compact only the trusted scalar SSA emitted here, never Agent input.

    One-use decorated helpers are expanded once. One-use scalar assignments are
    folded to a bounded expression depth. Primitive operations remain identical;
    the ordinary AST, native typing, expansion and work checks run afterwards.
    """
    import ast,copy
    tree=ast.parse(source);functions={n.name:n for n in tree.body}
    require(all(type(n) is ast.FunctionDef for n in tree.body),"Capacity generated function source")
    occupied={n.id for n in ast.walk(tree) if type(n) is ast.Name}
    serial=0
    while True:
        calls={}
        for n in ast.walk(tree):
            if type(n) is ast.Call and type(n.func) is ast.Name and n.func.id in functions:
                calls.setdefault(n.func.id,[]).append(n)
        selected=next((name for name,sites in calls.items() if name!="golden" and len(sites)==1),None)
        if selected is None:break
        fn=functions[selected];site=calls[selected][0]
        require(not site.keywords and len(site.args)==len(fn.args.args) and all(type(a) is ast.Name for a in site.args),
                "Capacity generated scalar call")
        require(type(fn.body[-1]) is ast.Return and type(fn.body[-1].value) is ast.Name,
                "Capacity generated scalar return")
        prefix="cap_inline"+str(serial)+"_";serial+=1
        names={a.arg:value.id for a,value in zip(fn.args.args,site.args)}
        body=[]
        for stmt in fn.body[:-1]:
            require(type(stmt) is ast.Assign and len(stmt.targets)==1 and type(stmt.targets[0]) is ast.Name,
                    "Capacity generated scalar assignment")
            target=stmt.targets[0].id;replacement=prefix+target
            while replacement in occupied:replacement+="_"
            occupied.add(replacement);names[target]=replacement
        class Rename(ast.NodeTransformer):
            def visit_Name(self,n):
                if n.id in names:n.id=names[n.id]
                return n
        body=[Rename().visit(copy.deepcopy(stmt)) for stmt in fn.body[:-1]]
        returned=names[fn.body[-1].value.id]
        found=False
        for caller in tree.body:
            if caller is fn:continue
            for i,stmt in enumerate(caller.body):
                if type(stmt) is ast.Assign and stmt.value is site:
                    stmt.value=ast.Name(id=returned,ctx=ast.Load())
                    caller.body[i:i]=body;found=True;break
            if found:break
        require(found,"Capacity generated direct scalar call")
        tree.body.remove(fn);functions.pop(selected)
    def depth(node):
        return 1+max((depth(child) for child in ast.iter_child_nodes(node)),default=0)
    for fn in tree.body:
        counts={}
        for n in ast.walk(fn):
            if type(n) is ast.Name and type(n.ctx) is ast.Load:counts[n.id]=counts.get(n.id,0)+1
        for i in reversed(range(len(fn.body))):
            stmt=fn.body[i]
            if type(stmt) is not ast.Assign or len(stmt.targets)!=1 or type(stmt.targets[0]) is not ast.Name:continue
            target=stmt.targets[0].id
            if counts.get(target)!=1:continue
            replacement=stmt.value
            slot=next((j for j in range(i+1,len(fn.body)) if any(type(n) is ast.Name and
                       type(n.ctx) is ast.Load and n.id==target for n in ast.walk(fn.body[j]))),None)
            if slot is None:continue
            class Fold(ast.NodeTransformer):
                def visit_Name(self,n):
                    return copy.deepcopy(replacement) if type(n.ctx) is ast.Load and n.id==target else n
            candidate=Fold().visit(copy.deepcopy(fn.body[slot]))
            if depth(candidate)>24:continue
            fn.body[slot]=candidate;del fn.body[i]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)+"\n"

def lower(request, *, semantic_pairs=False):
    validate_request(request);require(request["layout"]["execution_abi"] in ABIS,"Capacity ABI")
    g=request["model"];check=validate(g);em=Emitter(request);packed=packing(g,request["layout"],request_version(request))
    from unified_capacity_constants import VERSION,WINDOW_VERSION
    em.capacity_inline_rotations=request_version(request) in (WINDOW_VERSION,VERSION,CHANNEL_VERSION,INDEX_VERSION)
    values={k:np.asarray(v,dtype=np.float64) for k,v in g["constants"].items()}
    for spec in g["inputs"]:
        bindings=[b for b in request["layout"]["inputs"] if b["name"]==spec["name"]]
        values[spec["name"]]=Chunks(em,[Scalar(em,b["dsl_name"]) for b in bindings],spec["shape"])
    from capacity_pair_lowering import centered_pair, arithmetic as pair_arithmetic, unary as pair_unary, reduction as pair_reduction
    producers={out:node for node in g["nodes"] for out in node["outputs"]}
    pair_count=0
    for node in g["nodes"]:
        op=node["op"];attrs=node["attrs"];xs=[values[r] for r in node["inputs"]]
        if op=="split":
            values.update(split_tensor(em,xs[0],node,check["shapes"]));continue
        require(len(node["outputs"])==1,"No bounded capacity baseline for multi-result operator")
        shape=check["shapes"][node["outputs"][0]];x=xs[0]
        if semantic_pairs:
            special=centered_pair(em,node,x,producers,check["shapes"])
            if special is not None:pair_count+=1
            elif op in ("add","subtract","multiply"):
                special=pair_arithmetic(em,op,xs,shape,node=node,packed=packed)
            elif op in ("sum","mean"):
                special=pair_reduction(em,op,x,attrs,shape)
            else:special=pair_unary(em,op,x,attrs,shape)
            if special is not None:
                values[node["outputs"][0]]=special;continue
        if op in ("add","subtract","multiply"):
            count=math.ceil(math.prod(shape)/em.period)
            left=aligned_operand(x,shape,count,em,packed,node,0)
            right=aligned_operand(xs[1],shape,count,em,packed,node,1)
            parts=[vector_arithmetic(em,op,a,b) if isinstance(a,(Scalar,PublicVector))
                   else (b+a if op=="add" else a-b if op=="subtract" else b*a)
                   for a,b in zip(left,right)]
            replicated=math.prod(shape)==1 and all(not isinstance(v,Chunks) or v.replicated for v in xs)
            y=Chunks(em,parts,shape,replicated)
        else:
            require(isinstance(x,Chunks),"Capacity cipher expression")
            if op in ("reshape","flatten"):y=Chunks(em,x.parts,shape,x.replicated)
            elif op in ("sum","mean"):
                mode=packed["tensor_strategies"].get(node["id"])
                if mode=="aligned_suffix":
                    width=packed["suffix_windows"][node["id"]]["width"];parts=[]
                    for part in x.parts:
                        for bit in range(width.bit_length()-1):
                            part=part+em.rotate(part,1<<bit)
                        parts.append(part)
                    y=spatial_tensor(em,node,packed,x,shape,role="suffix_gather",source_parts=parts)
                else:
                    y=x.linear(node,packed,shape) if mode=="tiled_suffix" else route_tensor(em,node,packed,xs,shape) if mode=="direct" else x.reduction(op,attrs,shape)
            elif op=="batch_norm":
                require(packed["tensor_strategies"].get(node["id"])=="channel_affine","No bounded capacity BatchNorm binding")
                y=Chunks(em,[vector_arithmetic(em,"add",
                    vector_arithmetic(em,"multiply",part,public_vector(em,packed,node["id"],"bn_gain",chunk=i)),
                    public_vector(em,packed,node["id"],"bn_offset",chunk=i)) for i,part in enumerate(x.parts)],shape)
            elif op=="linear":y=x.linear(node,packed,shape)
            elif op in ("conv1d","conv2d","avg_pool1d","avg_pool2d"):
                mode=packed["tensor_strategies"].get(node["id"])
                require(mode in ("spatial_bsgs","spatial_window","spatial_stream"),"No bounded capacity spatial binding")
                y=spatial_stream(em,node,packed,x,shape) if mode=="spatial_stream" else spatial_window(em,node,packed,x,shape) if mode=="spatial_window" else spatial_tensor(em,node,packed,x,shape)
                if op.startswith("conv") and len(node["inputs"])==3:
                    y=Chunks(em,[vector_arithmetic(em,"add",part,
                        public_vector(em,packed,node["id"],"spatial_bias",chunk=i))
                        for i,part in enumerate(y.parts)],shape)
            elif op in ("negate","square","power","polynomial"):
                parts=[]
                for j,a in enumerate(x.parts):
                    n=min(em.period,math.prod(shape)-j*em.period)
                    args=[PackedPrefix(em,a,[n])]
                    if op=="polynomial":args.append(xs[1])
                    v=packed_elementwise(op,args,attrs,[n],balanced_power=True)
                    require(isinstance(v,PackedPrefix),"No bounded capacity elementwise baseline")
                    parts.append(v.expr)
                y=Chunks(em,parts,shape,x.replicated)
            elif op in ("slice","transpose","permute"):
                require(packed["tensor_strategies"].get(node["id"])=="spatial_bsgs","No bounded capacity static route binding")
                y=spatial_tensor(em,node,packed,x,shape)
            elif op in ("concat","stack"):
                if packed["tensor_strategies"].get(node["id"])=="tiled_stack":
                    parts=[em.zero]*math.ceil(math.prod(shape)/em.period)
                    for child in packed["stack_operands"][node["id"]]:
                        operand=child["operand"]
                        virtual=dict(node,id=child["node_id"],
                                     inputs=[node["inputs"][operand]]+[r for i,r in enumerate(node["inputs"]) if i!=operand])
                        result=xs[operand].linear(virtual,packed,shape)
                        parts=[a+b for a,b in zip(parts,result.parts)]
                    values[node["outputs"][0]]=Chunks(em,parts,shape);continue
                if packed["tensor_strategies"].get(node["id"])=="direct":
                    values[node["outputs"][0]]=route_tensor(em,node,packed,xs,shape);continue
                axis=attrs["axis"]%len(shape)
                require(axis==0 and all(isinstance(v,Chunks) for v in xs),"No bounded capacity baseline for inner-axis join")
                require(all(math.prod(v.shape)%em.period==0 for v in xs[:-1]),"No bounded capacity baseline for unaligned join")
                y=Chunks(em,[p for v in xs for p in v.parts],shape)
            else:raise ValueError("No bounded capacity baseline for "+op)
        values[node["outputs"][0]]=y
    results=[]
    for output,binding in zip(g["outputs"],request["layout"]["outputs"]):
        value=values[output["value"]]
        require(value.shape==tuple(binding["shape"]) and len(value.parts)==len(binding["chunks"]),"Capacity output binding")
        results.extend(v.name for v in value.parts)
    names=[v["dsl_name"] for v in request["layout"]["inputs"]]+["zero_ct"]
    source='@hc.func("'+",".join("c" for _ in names)+'")\ndef golden('+", ".join(names)+'):\n'
    source+="\n".join(em.lines)+"\n    return ["+", ".join(results)+"]\n"
    source="\n".join(s for _,s in getattr(em,"capacity_helpers",{}).values())+source
    require(len(source.encode())<=65536,"Rule source budget")
    neutral=None
    if semantic_pairs:
        from capacity_algebraic_schedule import remove_neutral_factors
        source,neutral=remove_neutral_factors(source,request)
    if request_version(request) in (WINDOW_VERSION,VERSION,CHANNEL_VERSION,INDEX_VERSION):source=compact_scalar_source(source)
    scheduling=None
    from capacity_algebraic_schedule import schedule,VERSION as SCHEDULE_VERSION
    if semantic_pairs:
        try:
            scheduled,scheduling=schedule(source,request)
            validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=scheduled),request)
        except ValueError as error:
            # Preserve the ordinary generated source when expansion exceeds its
            # existing contract; no limits are increased.
            resource_errors={"Capacity scalar expansion budget","Capacity scalar expression depth",
                "Trusted schedule node budget","Trusted schedule recursion budget","Trusted schedule work budget",
                "Trusted schedule serialization depth","Deterministic baseline operation budget","Rule source budget",
                "AST node limit","Expanded ciphertext operation budget exceeded"}
            if str(error) not in resource_errors:raise
            scheduling=dict(version=SCHEDULE_VERSION,applied=False,rejected=str(error))
        else:source=scheduled
    output_identity=None
    if request_version(request)==INDEX_VERSION:
        from capacity_output_identity import distinct_outputs
        source,output_identity=distinct_outputs(source)
    validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
    return source,dict(centered_pairs=pair_count,semantic_pairs=semantic_pairs,scheduling=scheduling,strategy="capacity-chunks-v10" if request_version(request)==INDEX_VERSION else "capacity-chunks-v9" if request_version(request)==CHANNEL_VERSION else "capacity-chunks-v8" if request_version(request)==VERSION else "capacity-chunks-v7" if request_version(request)==WINDOW_VERSION else "capacity-chunks-v6",request_changed=False,reference_changed=False,
                      physical_input_chunks=len(request["layout"]["inputs"]),
                      physical_output_chunks=request["layout"]["output_ciphertexts"],
                      **({"output_identity":output_identity} if output_identity is not None else {}),
                      **({"neutral_factors":neutral} if neutral is not None else {}))
