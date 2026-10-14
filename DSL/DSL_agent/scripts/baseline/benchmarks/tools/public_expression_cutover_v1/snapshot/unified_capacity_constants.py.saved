"""Public tensor packing for capacity graphs, independent of any emitted DSL."""
import math
from benchmark_graph import require,validate

LEGACY_VERSION="capacity-public-tensors-v1"
PREVIOUS_VERSION="capacity-public-tensors-v2"
TILE_VERSION="capacity-public-tensors-v3"
TENSOR_VERSION="capacity-public-tensors-v4"
SPATIAL_VERSION="capacity-public-tensors-v5"
WINDOW_VERSION="capacity-public-tensors-v6"
VERSION="capacity-public-tensors-v7"
CHANNEL_VERSION="capacity-public-tensors-v8"
INDEX_VERSION="capacity-public-tensors-v9"
VERSIONS=(LEGACY_VERSION,PREVIOUS_VERSION,TILE_VERSION,TENSOR_VERSION,SPATIAL_VERSION,WINDOW_VERSION,VERSION,CHANNEL_VERSION,INDEX_VERSION)

def request_version(request):
    descriptor=request["constant_origins"].get("capacity_tensor_packing")
    return descriptor["version"] if descriptor is not None else TILE_VERSION

def scalars(value):
    if isinstance(value,list):return [v for child in value for v in scalars(child)]
    return [float(value)]

def broadcast_values(value,source_shape,target_shape):
    """Static C-order public broadcasting; requires no tracing environment."""
    source=scalars(value);shape=(1,)*(len(target_shape)-len(source_shape))+tuple(source_shape)
    require(len(shape)==len(target_shape) and
            all(n==1 or n==m for n,m in zip(shape,target_shape)),"Capacity public broadcast shape")
    result=[]
    for index in range(math.prod(target_shape)):
        coordinates=[0]*len(target_shape);remaining=index
        for axis in reversed(range(len(target_shape))):
            coordinates[axis]=remaining%target_shape[axis];remaining//=target_shape[axis]
        source_index=0
        for n,coordinate in zip(shape,coordinates):source_index=source_index*n+(0 if n==1 else coordinate)
        result.append(source[source_index])
    return result

def diagonal_vectors(weights,batches,period):
    """Route each source element to its C-order destination, using public masks."""
    outputs,width=len(weights),len(weights[0]);groups={}
    # Row sums provide the alternate construction. This estimate selects only
    # between two public linear maps; it does not evaluate a candidate or reference.
    row_cost=0
    for batch in range(batches):
        start=batch*width;end=start+width
        for output in range(outputs):
            count=sum(any(v!=0. for v in weights[output][max(start,c*period)-start:min(end,(c+1)*period)-start])
                      for c in range(start//period,(end-1)//period+1))
            if count:row_cost+=count*(2*period.bit_length())+(output%period).bit_count()+2
    diagonal_cost=0
    for batch in range(batches):
        for output in range(outputs):
            dst=batch*outputs+output
            for j in range(width):
                weight=weights[output][j]
                if weight==0.:continue
                src=batch*width+j
                key=(src//period,dst//period,(src-dst)%period)
                if key not in groups:
                    diagonal_cost+=2+key[2].bit_count()
                    # Stop materializing the alternate map once it cannot win.
                    if diagonal_cost>=row_cost or len(groups)>=1024:return None
                    groups[key]=[0.]*period
                slot=dst%period
                require(groups[key][slot]==0.,"Capacity Linear diagonal collision")
                groups[key][slot]=weight
    return groups


def row_sum_cost(weights,batches,period):
    """Cost of the actual bounded row-sum emitter, including public vectors."""
    width=len(weights[0]);outputs=len(weights);cost=0;unique=set();destinations={}
    for batch in range(batches):
        start=batch*width;end=start+width
        for output,row in enumerate(weights):
            terms=0
            for chunk in range(start//period,(end-1)//period+1):
                lo=max(start,chunk*period);hi=min(end,(chunk+1)*period)
                vector=[0.]*period;vector[lo-chunk*period:hi-chunk*period]=row[lo-start:hi-start]
                unique.add(tuple(vector))
                if not any(vector):continue
                terms+=1
                cost+=int(not all(v==1. for v in vector))+2*(period.bit_length()-1)
            if terms:
                dst=batch*outputs+output
                cost+=terms-1+1+(dst%period).bit_count()
                destinations[dst//period]=destinations.get(dst//period,0)+1
    return cost+len(unique)+sum(max(0,n-1) for n in destinations.values())


def packing(model,plan,version=VERSION):
    require(version in VERSIONS,"Unknown capacity public packing version")
    check=validate(model);period=plan["input_slot_period"]
    constants=model["constants"]
    vectors={};bindings=[];seen={};packed_refs=set();other_refs=set();strategies={};tiles={};tensor_strategies={};stack_operands={};spatial_windows={};spatial_streams={};suffix_windows={};occupied={n["id"] for n in model["nodes"]}
    def bind(vector,node,role,source,**indices):
        require(len(vector)==period and all(math.isfinite(v) and abs(v)<=1024 for v in vector),
                "Capacity public vector values")
        key=seen.get(tuple(vector))
        if key is None:
            require(len(vectors)<256,"Capacity public tensor registry budget")
            key="pv"+str(len(vectors));seen[tuple(vector)]=key;vectors[key]=vector
        bindings.append(dict(node_id=node["id"],role=role,source=source,key=key,**indices))
        packed_refs.add(source)
    def bind_tile(node,tiled,source):
        strategies[node["id"]]="tiled_row_sum" if tiled[0].get("method")=="row_sum" else "tiled_diagonal";tiles[node["id"]]=tiled[0]
        from unified_capacity_tiling import range_vector,template_vectors
        for count,t in sorted(tiled[1].items()):
            for role,phase,side,vector in template_vectors(t):
                bind(vector,node,"linear_tile_"+role,source,tile_batches=count,phase=phase,side=side)
        for block in tiled[0]["blocks"]:
            for role,ref in (("gather",node["inputs"][0]),("scatter",node["outputs"][0])):
                if len(block[role])==1:continue
                for span in block[role]:
                    bind(range_vector(span["lo"],span["hi"],period),node,"linear_tile_"+role,ref,
                         batch_offset=block["batch_offset"],chunk=span["chunk"])
    for node in model["nodes"]:
        op=node["op"];refs=node["inputs"]
        if op in ("add","subtract","multiply"):
            outshape=check["shapes"][node["outputs"][0]]
            for operand,ref in enumerate(refs):
                if ref not in constants:continue
                a=scalars(constants[ref])
                if all(v==a[0] for v in a):
                    # A uniform tensor may also be a Linear weight/bias. Its
                    # scalar broadcast value must survive public vector packing.
                    # Retain exact legacy registry identities for v1 requests.
                    if version!=LEGACY_VERSION:other_refs.add(ref)
                    continue
                flat=broadcast_values(constants[ref],check["shapes"][ref],outshape)
                for chunk,start in enumerate(range(0,len(flat),period)):
                    part=flat[start:start+period]
                    bind(part+[0.]*(period-len(part)),node,"elementwise",ref,operand=operand,chunk=chunk)
        elif version in (CHANNEL_VERSION,INDEX_VERSION) and op=="batch_norm":
            source_shape=check["shapes"][refs[0]];spatial=math.prod(source_shape[2:])
            mean,var,gamma,beta=[constants[ref] for ref in refs[1:]]
            gains=[float(g)/math.sqrt(float(v)+node["attrs"]["eps"]) for v,g in zip(var,gamma)]
            offsets=[float(b)-float(m)*gain for m,b,gain in zip(mean,beta,gains)]
            tensor_strategies[node["id"]]="channel_affine"
            for chunk,start in enumerate(range(0,math.prod(source_shape),period)):
                count=min(period,math.prod(source_shape)-start)
                for role,values in (("bn_gain",gains),("bn_offset",offsets)):
                    vector=[values[((start+i)//spatial)%source_shape[1]] for i in range(count)]
                    bind(vector+[0.]*(period-count),node,role,refs[1] if role=="bn_gain" else refs[4],chunk=chunk)
            packed_refs.update(refs[1:])
        elif op=="linear":
            weights=[[float(v) for v in row] for row in constants[refs[1]]];width=check["shapes"][refs[0]][-1];outputs=len(weights)
            batches=math.prod(check["shapes"][refs[0]])//width
            routes=diagonal_vectors(weights,batches,period) if version!=LEGACY_VERSION else None
            tiled=None
            if version in (TILE_VERSION,TENSOR_VERSION,SPATIAL_VERSION,WINDOW_VERSION,VERSION,CHANNEL_VERSION,INDEX_VERSION):
                from unified_capacity_tiling import plan as tile_plan,range_vector,template_vectors
                alternate=sum(2+step.bit_count() for _,_,step in routes) if routes is not None else (
                    row_sum_cost(weights,batches,period) if version in (WINDOW_VERSION,VERSION,CHANNEL_VERSION,INDEX_VERSION) else 1024)
                tiled=tile_plan(weights,batches,period,alternate)
                if version in (VERSION,CHANNEL_VERSION,INDEX_VERSION):
                    row_tiled=tile_plan(weights,batches,period,min(alternate,tiled[0]["estimated_work"] if tiled else 1024),row_sum=True)
                    if row_tiled is not None:tiled=row_tiled
            strategies[node["id"]]="diagonal" if routes is not None else "row_sum"
            if tiled is not None:
                packed_refs.add(refs[1]);bind_tile(node,tiled,refs[1])
            elif routes is not None:
                # All-zero weights still consume their public binding, and use
                # the trusted encrypted zero in emitted code.
                packed_refs.add(refs[1])
                for (source,target,step),vector in sorted(routes.items()):
                    bind(vector,node,"linear_diagonal",refs[1],
                         input_chunk=source,output_chunk=target,step=step)
            else:
                for batch in range(batches):
                    start=batch*width;end=start+width
                    for output in range(outputs):
                        for chunk in range(start//period,(end-1)//period+1):
                            lo=max(start,chunk*period);hi=min(end,(chunk+1)*period)
                            vector=[0.]*period
                            vector[lo-chunk*period:hi-chunk*period]=weights[output][lo-start:hi-start]
                            bind(vector,node,"linear_weight",refs[1],output=batch*outputs+output,chunk=chunk)
            if len(refs)==3:
                flat=broadcast_values(constants[refs[2]],check["shapes"][refs[2]],(batches,outputs))
                for chunk,start in enumerate(range(0,len(flat),period)):
                    part=flat[start:start+period]
                    bind(part+[0.]*(period-len(part)),node,"linear_bias",refs[2],chunk=chunk)
        elif version in (SPATIAL_VERSION,WINDOW_VERSION,VERSION,CHANNEL_VERSION,INDEX_VERSION) and op in ("conv1d","conv2d","avg_pool1d","avg_pool2d"):
            from unified_capacity_spatial import vectors as spatial_vectors,bias_vectors
            require(not op.startswith("conv") or all(ref in constants for ref in refs[1:]),
                    "Capacity spatial public weights")
            routes=spatial_vectors(node,check["shapes"],constants,period)
            source=refs[1] if op.startswith("conv") else refs[0]
            if op.startswith("conv"):packed_refs.update(refs[1:])
            window=None;stream=None
            if version in (WINDOW_VERSION,VERSION,CHANNEL_VERSION,INDEX_VERSION) and op.startswith("avg_pool"):
                from unified_capacity_spatial import pool_window_plan,pool_window_work,bsgs_vectors
                option=pool_window_plan(node,check["shapes"],constants,period)
                if option is not None:
                    unique={tuple(v) for v in bsgs_vectors(option["anchors"],period).values()}
                    unique.update(tuple(v) for route in option["shifts"].values() for v in route.values())
                    # Compare structural operation counts; never probe a
                    # candidate or choose a model-dependent expected answer.
                    from unified_capacity_routes import work_estimate
                    if len(set(seen)|unique)<=256 and pool_window_work(option,period)<work_estimate(routes,period):
                        window=option
            if version in (VERSION,CHANNEL_VERSION,INDEX_VERSION) and op.startswith("avg_pool"):
                from unified_capacity_spatial import pool_stream_plan,bsgs_vectors
                stream=pool_stream_plan(node,check["shapes"],period)
                if stream is not None:
                    unique={tuple(v) for v in bsgs_vectors(stream["anchors"],period).values()}
                    if len(set(seen)|unique)>256:stream=None
            tensor_strategies[node["id"]]="spatial_stream" if stream is not None else "spatial_window" if window is not None else "spatial_bsgs"
            if stream is not None:
                spatial_streams[node["id"]]=stream["streams"];routes=stream["anchors"]
            elif window is not None:
                spatial_windows[node["id"]]=dict(axes=window["axes"],shifts=sorted(window["shifts"]))
                for offset,shift_routes in window["shifts"].items():
                    for (operand,input_chunk,output_chunk,step),vector in sorted(shift_routes.items()):
                        bind(vector,node,"pool_shift",source,offset=offset,input_chunk=input_chunk,
                             output_chunk=output_chunk,step=step)
                routes=window["anchors"]
            for (operand,input_chunk,output_chunk,step),vector in sorted(routes.items()):
                baby=step%min(16,period);giant=step-baby
                rotated=[vector[(i-giant)%period] for i in range(period)]
                indices=dict(input_chunk=input_chunk)
                if stream is not None:
                    indices=dict(input_chunk=stream["streams"][input_chunk]["input_chunk"],stream=input_chunk)
                bind(rotated,node,"pool_stream_gather" if stream is not None else "pool_gather" if window is not None else "tensor_route",source,operand=operand,
                     output_chunk=output_chunk,step=step,baby_step=baby,giant_step=giant,**indices)
            for chunk,vector in bias_vectors(node,check["shapes"],constants,period).items():
                bind(vector,node,"spatial_bias",refs[2],chunk=chunk)
        elif version in (CHANNEL_VERSION,INDEX_VERSION) and ((plan.get("execution_abi")=="unified-capacity-chunked-inputs-v4" and op=="slice")
                or (plan.get("resource_profile",{}).get("name")=="general-tensors-v1" and op in ("transpose","permute"))
                or (version==INDEX_VERSION and op=="slice")):
            from unified_capacity_routes import vectors as route_vectors
            tensor_strategies[node["id"]]="spatial_bsgs"
            for (operand,source,target,step),vector in sorted(route_vectors(node,check["shapes"],period).items()):
                baby=step%min(16,period);giant=step-baby
                rotated=[vector[(i-giant)%period] for i in range(period)]
                bind(rotated,node,"tensor_route",refs[0],operand=operand,
                     input_chunk=source,output_chunk=target,step=step,baby_step=baby,giant_step=giant)
        elif version in (TENSOR_VERSION,SPATIAL_VERSION,WINDOW_VERSION,VERSION,CHANNEL_VERSION,INDEX_VERSION) and op in ("concat","stack","sum","mean"):
            from unified_capacity_routes import vectors as route_vectors,suffix_spec,work_estimate
            # Preserve the already verified whole-tensor reduction path.
            shape=check["shapes"][refs[0]]
            if op in ("sum","mean") and {i%len(shape) for i in node["attrs"]["axes"]}==set(range(len(shape))):
                continue
            if any(ref in constants for ref in refs):
                # Public joins retain their pre-v4 registry and remain legal
                # requests; this cipher-routing baseline is not input authority.
                other_refs.update(ref for ref in refs if ref in constants);continue
            if version in (CHANNEL_VERSION,INDEX_VERSION) and op in ("sum","mean"):
                from unified_capacity_routes import aligned_suffix_plan
                window=aligned_suffix_plan(node,check["shapes"],period)
                if window is not None:
                    tensor_strategies[node["id"]]="aligned_suffix"
                    suffix_windows[node["id"]]=dict(width=window["width"],rows=window["rows"])
                    for (operand,source,target,step),vector in sorted(window["routes"].items()):
                        baby=step%min(16,period);giant=step-baby
                        rotated=[vector[(i-giant)%period] for i in range(period)]
                        bind(rotated,node,"suffix_gather",refs[0],operand=operand,
                             input_chunk=source,output_chunk=target,step=step,baby_step=baby,giant_step=giant)
                    continue
            routes=route_vectors(node,check["shapes"],period)
            if op=="stack" and node["attrs"]["axis"]%len(check["shapes"][node["outputs"][0]])==len(shape):
                from unified_capacity_tiling import plan as tile_plan
                batches=math.prod(shape);arity=len(refs);plans=[]
                for operand in range(arity):
                    weights=[[float(o==operand)] for o in range(arity)]
                    tiled=tile_plan(weights,batches,period,1024)
                    if tiled is None:plans=[];break
                    plans.append(tiled)
                if plans and sum(p[0]["estimated_work"] for p in plans)<work_estimate(routes,period):
                    tensor_strategies[node["id"]]="tiled_stack";children=[]
                    for operand,tiled in enumerate(plans):
                        identity=node["id"]+"_operand"+str(operand)
                        while identity in occupied:identity+="_"
                        occupied.add(identity)
                        child=dict(node,id=identity,
                                   inputs=[refs[operand]]+[r for i,r in enumerate(refs) if i!=operand])
                        children.append(dict(operand=operand,node_id=child["id"]))
                        bind_tile(child,tiled,refs[operand])
                    stack_operands[node["id"]]=children;continue
            spec=suffix_spec(node,check["shapes"]);tiled=None
            if spec is not None:
                from unified_capacity_tiling import plan as tile_plan
                alternate=work_estimate(routes,period)
                if len({tuple(v) for v in routes.values()})>=256:alternate=1024
                tiled=tile_plan(spec[0],spec[1],period,alternate)
            tensor_strategies[node["id"]]="tiled_suffix" if tiled is not None else "direct"
            if tiled is not None:bind_tile(node,tiled,refs[0])
            else:
                for (operand,source,target,step),vector in sorted(routes.items()):
                    bind(vector,node,"tensor_route",refs[operand],operand=operand,
                         input_chunk=source,output_chunk=target,step=step)
        else:
            other_refs.update(ref for ref in refs if ref in constants)
    # A tensor also used as polynomial coefficients or another unsupported
    # public operation retains its scalar entries for existing expression rules.
    excluded=packed_refs-other_refs
    from unified_capacity_layout import WIDE_ABI
    explicit_wide=plan.get("execution_abi")==WIDE_ABI and version in (CHANNEL_VERSION,INDEX_VERSION)
    descriptor=dict(version=version,bindings=bindings) if bindings or (version!=LEGACY_VERSION and strategies) or tensor_strategies or explicit_wide or version==INDEX_VERSION else None
    if descriptor is not None and version!=LEGACY_VERSION:descriptor["linear_strategies"]=strategies
    if descriptor is not None and tiles:descriptor["linear_tiles"]=tiles
    if descriptor is not None and tensor_strategies:descriptor["tensor_strategies"]=tensor_strategies
    if descriptor is not None and stack_operands:descriptor["stack_operands"]=stack_operands
    if descriptor is not None and spatial_windows:descriptor["spatial_windows"]=spatial_windows
    if descriptor is not None and spatial_streams:descriptor["spatial_streams"]=spatial_streams
    if descriptor is not None and suffix_windows:descriptor["suffix_windows"]=suffix_windows
    return dict(vectors=vectors,bindings=bindings,excluded_scalars=excluded,
                linear_strategies=strategies,linear_tiles=tiles,tensor_strategies=tensor_strategies,stack_operands=stack_operands,spatial_windows=spatial_windows,spatial_streams=spatial_streams,suffix_windows=suffix_windows,descriptor=descriptor)

def key_for(packed,node_id,role,**indices):
    matches=[r["key"] for r in packed["bindings"]
             if r["node_id"]==node_id and r["role"]==role and all(r.get(k)==v for k,v in indices.items())]
    require(len(matches)==1,"Capacity public vector binding")
    return matches[0]


def rules(version):
    text="""
Public packing capacity-public-tensors-v6: bindings name immutable length-P
vectors, not extra model parameters. Linear strategies are public layout plans.
For tensor_route and pool_gather with baby_step/giant_step, the registered mask
is pre-rotated: mask[j] is the destination coefficient at (j-giant_step) mod P.
The complete route reads input_chunk at the declared step and places its values
in output_chunk; shifts spanning chunks use their declared source/destination.
spatial_window factors an unpadded average window into flat sums, then selects
only valid spatial anchors. spatial_windows.axes gives each count and flat
stride; pool_shift offset is a zero-filled flat displacement before pooling
gather, not an independent tensor input. pool_gather includes the average
divisor. Intermediate windows across row/channel boundaries are never selected.
These plans change physical routing only; the declared model mathematics,
weights, layouts, resource limits and numerical reference remain fixed.
"""
    if version==WINDOW_VERSION:return text
    if version not in (VERSION,CHANNEL_VERSION,INDEX_VERSION):return ""
    result=text.replace("capacity-public-tensors-v6","capacity-public-tensors-v7")+"""
spatial_stream sums a public list of offsets within one source chunk before
gathering valid anchors. spatial_streams lists input_chunk and offsets; stream
in pool_stream_gather selects that list entry, while input_chunk records the
original physical input. Gather masks include the average divisor.
tiled_row_sum Linear first places independent rows at a power-of-two stride,
performs a row-local weighted sum, selects row starts, then packs output rows.
linear_tiles.method, public row weights and row masks bind this layout.
"""

    if version in (CHANNEL_VERSION,INDEX_VERSION):
        result=result.replace("capacity-public-tensors-v7","capacity-public-tensors-v8")+"""
channel_affine preserves inference BatchNorm: bn_gain and bn_offset are public
length-P vectors derived from the frozen statistics, gain, bias and epsilon.
Each output chunk equals input_chunk * bn_gain + bn_offset. Tail positions
remain zero and are outside the logical output.
aligned_suffix sums complete disjoint power-of-two suffix rows. suffix_windows
binds the row width and count; suffix_gather observes only row starts after
rotations/additions at offsets 1,2,...,width/2. Masks include the mean divisor
when applicable. Unobserved rolling-window positions are not tensor outputs.
"""
    if version==INDEX_VERSION:
        result=result.replace("capacity-public-tensors-v8","capacity-public-tensors-v9")+"""
Static slice routes preserve C-order tensor indexing: starts, stops and positive
steps are model attributes. tensor_route masks bind each source chunk and output
chunk, including zero-filled tails. Removing or inserting singleton dimensions
uses reshape after selection and does not rotate ciphertext slots by itself.
"""
    return result
