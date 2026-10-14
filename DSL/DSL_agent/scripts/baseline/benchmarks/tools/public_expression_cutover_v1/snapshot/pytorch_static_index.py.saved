"""Static public tensor selections lowered to existing graph operations."""
import math
from benchmark_graph import require,digest
from model_resources import resolve
VERSION="poseidon-pytorch-static-index-v1"
POLICY="signature-static-index-v1"

def encode(value):
    if type(value) is int:return dict(kind="integer",value=value)
    if isinstance(value,slice):
        require(all(v is None or type(v) is int for v in (value.start,value.stop,value.step)),
                "Static integer slice bounds")
        return dict(kind="slice",start=value.start,stop=value.stop,step=value.step)
    if value is None:return dict(kind="newaxis")
    if value is Ellipsis:return dict(kind="ellipsis")
    if type(value) is list:
        require(all(type(v) is int for v in value),"Public integer index vector")
        return dict(kind="vector",values=list(value))
    raise ValueError("Public static tensor index required")

def decode(value):
    require(type(value) is dict and "kind" in value,"Static index description")
    kind=value["kind"]
    if kind=="integer" and set(value)=={"kind","value"} and type(value["value"]) is int:return value["value"]
    if kind=="slice" and set(value)=={"kind","start","stop","step"}:
        require(all(v is None or type(v) is int for v in (value["start"],value["stop"],value["step"])),"Static integer slice bounds")
        return slice(value["start"],value["stop"],value["step"])
    if kind=="newaxis" and set(value)=={"kind"}:return None
    if kind=="ellipsis" and set(value)=={"kind"}:return Ellipsis
    if kind=="vector" and set(value)=={"kind","values"} and type(value["values"]) is list:
        require(all(type(v) is int for v in value["values"]),"Public integer index vector")
        return list(value["values"])
    raise ValueError("Static index description")

def plan(source_shape,index,*,resource_profile):
    limits=resolve(resource_profile);dims=tuple(source_shape)
    require(len(dims)<=limits.rank and all(type(n) is int and 0<n<=limits.axis_size for n in dims)
            and math.prod(dims)<=limits.tensor_elements,"Static index source shape budget")
    keys=list(index) if type(index) is tuple else [index]
    descriptions=[encode(k) for k in keys]
    require(len(keys)<=2*limits.rank+1,"Static index selector budget")
    require(sum(k is Ellipsis for k in keys)<=1,"At most one ellipsis")
    consumed=sum(k is not None and k is not Ellipsis for k in keys)
    require(consumed<=len(dims),"Too many tensor indices")
    vectors=[k for k in keys if type(k) is list]
    require(len(vectors)<=1,"Only one public advanced index axis")
    require(not vectors or not any(type(k) is int for k in keys),
            "Mixed advanced and integer indexing is not registered")
    normalized=[]
    for key in keys:
        if key is Ellipsis:normalized.extend([slice(None)]*(len(dims)-consumed))
        else:normalized.append(key)
    if not any(k is Ellipsis for k in keys):normalized.extend([slice(None)]*(len(dims)-consumed))
    steps=[];target=[];axis=0
    for key in normalized:
        if key is None:target.append(1);continue
        width=dims[axis]
        if type(key) is int:
            require(-width<=key<width,"Static tensor index out of bounds")
            start=key%width
            steps.append(dict(op="slice",attrs=dict(axis=axis,start=start,stop=start+1,step=1)))
        elif type(key) is list:
            require(1<=len(key)<=limits.axis_size and all(-width<=k<width for k in key),
                    "Public index vector bounds or budget")
            indices=[k%width for k in key]
            steps.append(dict(op="public_gather",attrs=dict(axis=axis,indices=indices)))
            target.append(len(indices))
        else:
            require(isinstance(key,slice),"Static tensor selector")
            require(key.step is None or key.step>0,"Positive static slice step required")
            start,stop,step=key.indices(width);length=len(range(start,stop,step))
            require(length>0,"Empty tensor selections are not registered")
            if (start,stop,step)!=(0,width,1):
                steps.append(dict(op="slice",attrs=dict(axis=axis,start=start,stop=stop,step=step)))
            target.append(length)
        axis+=1
    require(axis==len(dims) and len(target)<=limits.rank and math.prod(target)<=limits.tensor_elements,
            "Static index result shape budget")
    return dict(selectors=descriptions,steps=steps,result_shape=target)

def axis_selection(source_shape,axis,index,*,kind,resource_profile):
    require(type(axis) is int and -len(source_shape)<=axis<len(source_shape),"Static tensor index axis")
    selectors=[slice(None)]*len(source_shape);selectors[axis%len(source_shape)]=index
    if kind=="index_select":
        require(type(index) is list and all(type(i) is int and 0<=i<source_shape[axis] for i in index),
                "Index_select needs nonnegative public indices")
    return plan(source_shape,tuple(selectors),resource_profile=resource_profile)

def check_record(record,original,*,resource_profile):
    require(type(record) is dict and set(record)=={"version","graph_sha256","entries"}
            and record["version"]==VERSION and record["graph_sha256"]==digest(original),
            "Static index graph binding")
    entries=record["entries"];limits=resolve(resource_profile)
    require(type(entries) is list and len(entries)<=limits.original_nodes,"Static index record budget")
    from model_decomposition import Builder
    b=Builder(original,resource_profile=resource_profile)
    shapes={name:b.shapes[value] for name,value in b.refs.items()}
    nodes={n["id"]:n for n in original["nodes"]}
    require(len(nodes)==len(original["nodes"]),"Static index graph ids")
    for node in original["nodes"]:
        b.origin=node["id"];results=b.call(node["op"],[b.refs[r] for r in node["inputs"]],node["attrs"])
        require(len(results)==len(node["outputs"]),"Static index graph arity")
        for name,value in zip(node["outputs"],results):
            b.refs[name]=value;shapes[name]=b.shapes[value]
    seen=set()
    for entry in entries:
        require(type(entry) is dict and set(entry)=={"operation","source","source_shape","plan","node_ids"},
                "Static index record fields")
        require(entry["operation"] in ("getitem","select","narrow","index_select"),"Static index operation")
        dims=entry["source_shape"];require(type(dims) is list and entry["source"] in shapes,"Static index source")
        require(tuple(dims or [1])==shapes[entry["source"]],"Static index source shape binding")
        declared=entry["plan"];require(type(declared) is dict and "selectors" in declared and type(declared["selectors"]) is list,
                                       "Static index plan fields")
        actual=plan(dims,tuple(decode(v) for v in declared["selectors"]),resource_profile=resource_profile)
        require(actual==declared,"Static index plan binding")
        ids=entry["node_ids"];steps=actual["steps"]+[dict(op="reshape",attrs=dict(shape=actual["result_shape"] or [1]))]
        require(type(ids) is list and len(ids)==len(steps) and not seen.intersection(ids),"Static index node binding")
        previous=entry["source"]
        for name,step in zip(ids,steps):
            require(name in nodes,"Static index node exists")
            node=nodes[name]
            require(node["op"]==step["op"] and node["attrs"]==step["attrs"] and node["inputs"]==[previous]
                    and len(node["outputs"])==1,"Static index graph operation binding")
            previous=node["outputs"][0];seen.add(name)
    return record
