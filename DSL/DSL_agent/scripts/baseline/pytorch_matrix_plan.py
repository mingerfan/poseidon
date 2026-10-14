"""Versioned static PyTorch matrix normalization, independent of DSL/backend code."""
import math
from benchmark_graph import require, broadcast, dimensions, digest, canonical
from model_resources import resolve

LEGACY_POLICY = "rank-two-three-v1"
POLICY = "static-tensor-matmul-v1"

def plan(first, second, *, operation="matmul", public_side=None, resource_profile="legacy-v1"):
    limits = resolve(resource_profile)
    require(operation in ("matmul","mm","bmm","dot") and public_side in (None,"left","right"),
            "Static matrix operation/public side")
    shapes = [first, second]
    for i,s in enumerate(shapes):
        bound = limits.constant_elements if public_side == ("left","right")[i] else limits.tensor_elements
        require(type(s) is list and 1 <= len(s) <= limits.rank and
                all(type(n) is int and 1 <= n <= limits.axis_size for n in s) and math.prod(s) <= bound,
                "Static matrix operand shape budget")
    if operation == "mm":
        require(len(first) == len(second) == 2,"mm requires rank-2 tensors")
    elif operation == "bmm":
        require(len(first) == len(second) == 3 and first[0] == second[0],
                "bmm requires matching rank-3 batches")
    elif operation == "dot":
        require(len(first) == len(second) == 1 and first == second,"dot requires matching vectors")
    left = [1,*first] if len(first) == 1 else list(first)
    right = [*second,1] if len(second) == 1 else list(second)
    require(left[-1] == right[-2],"Static matrix contraction mismatch")
    batch = list(broadcast(tuple(left[:-2]),tuple(right[:-2])))
    m,k,n = left[-2],left[-1],right[-1]
    result = batch + ([] if len(first) == 1 else [m]) + ([] if len(second) == 1 else [n])
    core = result or [1]
    dimensions(core,resource_profile=resource_profile)
    # Named-output capacity is checked by graph admission; intermediates use tensor_elements.
    # This is a logical intermediate bound; it does not predict runtime memory.
    strategy = ("plain_right_linear" if public_side == "right" and len(second) <= 2 else
                "plain_left_linear" if public_side == "left" and len(first) <= 2 else
                "plain_tensor_product" if public_side else "cipher_matmul")
    steps=[]
    def add(op,values,attrs=None):
        name="step_"+str(len(steps))
        steps.append(dict(op=op,inputs=values,attrs=attrs or {}))
        return name
    def reshape(value,old,new):
        dimensions(new,resource_profile=resource_profile)
        return value if list(old) == list(new) else add("reshape",[value],dict(shape=new))
    constant=None
    product_elements=None
    B=math.prod(batch)
    if strategy == "plain_right_linear":
        constant=dict(source="right",transform="linear_right",shape=[n,k])
        value=add("linear",["left","constant"])
    elif strategy == "plain_left_linear":
        constant=dict(source="left",transform="linear_left",shape=[m,k])
        value=reshape("right",second,right)
        value=add("transpose",[value],dict(dim0=-2,dim1=-1))
        value=add("linear",[value,"constant"])
        value=add("transpose",[value],dict(dim0=-2,dim1=-1))
    else:
        product_elements=B*m*k*n
        require(product_elements <= limits.tensor_elements,"Static matrix product tensor budget")
        common_left=batch+[m,k];common_right=batch+[k,n]
        dimensions(common_left,resource_profile=resource_profile)
        dimensions(common_right,resource_profile=resource_profile)
        def operand(which,original,promoted,common,flat):
            value=reshape(which,original,promoted)
            if promoted != common:value=add("broadcast",[value],dict(shape=common))
            return reshape(value,common,flat)
        if strategy == "cipher_matmul":
            flat_left=([B] if batch else [])+[m,k]
            flat_right=([B] if batch else [])+[k,n]
            l=operand("left",first,left,common_left,flat_left)
            r=operand("right",second,right,common_right,flat_right)
            value=add("matmul",[l,r])
        else:
            if public_side == "right":
                value=operand("left",first,left,common_left,[B,m,k,1])
                constant=dict(source="right",transform="product_right",shape=[B,1,k,n])
            else:
                value=operand("right",second,right,common_right,[B,1,k,n])
                constant=dict(source="left",transform="product_left",shape=[B,m,k,1])
            dimensions([B,m,k,n],resource_profile=resource_profile)
            value=add("broadcast",[value],dict(shape=[B,m,k,n]))
            value=add("multiply",[value,"constant"])
            value=add("sum",[value],dict(axes=[2],keepdims=False))
    add("reshape",[value],dict(shape=core))
    require(len(steps) <= limits.original_nodes,"Static matrix normalization node budget")
    return dict(version=POLICY,operation=operation,source_shapes=shapes,public_side=public_side,
                promoted_shapes=[left,right],batch_shape=batch,result_shape=result,core_shape=core,
                strategy=strategy,product_tensor_elements=product_elements,constant=constant,steps=steps)

def constant_value(value,planned):
    """Only reshape/transposition/broadcast of the immutable public operand."""
    import numpy as np
    descriptor=planned["constant"]
    require(descriptor is not None,"Static matrix constant plan")
    array=np.asarray(value,dtype="<f8")
    index=0 if descriptor["source"] == "left" else 1
    require(list(array.shape) == planned["source_shapes"][index],"Static matrix public source shape")
    transform=descriptor["transform"]
    if transform == "linear_right":
        result=array.T if array.ndim == 2 else array.reshape(descriptor["shape"])
    elif transform == "linear_left":
        result=array.reshape(descriptor["shape"])
    else:
        promoted=planned["promoted_shapes"][index]
        batch=planned["batch_shape"]
        result=np.broadcast_to(array.reshape(promoted),batch+promoted[-2:]).reshape(descriptor["shape"])
    require(list(result.shape) == descriptor["shape"] and np.isfinite(result).all(),
            "Static matrix derived constant shape")
    return result.tolist()

def check_record(record,original,*,resource_profile="legacy-v1"):
    from model_decomposition import Builder
    require(type(record) is dict and set(record) == {"version","graph_sha256","entries"} and
            record["version"] == POLICY and record["graph_sha256"] == digest(original),
            "Static matrix record identity")
    entries=record["entries"];limits=resolve(resource_profile)
    require(type(entries) is list and len(entries) <= limits.original_nodes,"Static matrix record budget")
    builder=Builder(original,resource_profile=resource_profile)
    shapes={name:builder.shapes[value] for name,value in builder.refs.items()}
    kinds={name:builder.kinds[value] for name,value in builder.refs.items()}
    nodes={v["id"]:v for v in original["nodes"]}
    for node in original["nodes"]:
        builder.origin=node["id"]
        values=builder.call(node["op"],[builder.refs[r] for r in node["inputs"]],node["attrs"])
        for name,value in zip(node["outputs"],values):
            builder.refs[name]=value;shapes[name]=builder.shapes[value];kinds[name]=builder.kinds[value]
    seen=set()
    for entry in entries:
        require(type(entry) is dict and set(entry) == {"sources","plan","node_ids","constant"},
                "Static matrix record fields")
        sources=entry["sources"];declared=entry["plan"]
        require(type(sources) is list and len(sources) == 2 and all(v in shapes for v in sources),
                "Static matrix record sources")
        require(type(declared) is dict and "operation" in declared and "public_side" in declared,
                "Static matrix plan fields")
        plain=[kinds[v] == "plain" for v in sources]
        require(not all(plain),"Static matrix encrypted dependency")
        side="left" if plain[0] else "right" if plain[1] else None
        actual=plan([*shapes[sources[0]]],[*shapes[sources[1]]],
                    operation=declared["operation"],public_side=side,resource_profile=resource_profile)
        require(actual == declared,"Static matrix plan binding")
        refs=dict(zip(("left","right"),sources));constant=entry["constant"]
        if actual["constant"] is None:
            require(constant is None,"Static matrix unexpected constant")
        else:
            require(type(constant) is str and constant in original["constants"] and
                    sources[0 if side == "left" else 1] in original["constants"],
                    "Static matrix constant source")
            expected=constant_value(original["constants"][sources[0 if side == "left" else 1]],actual)
            require(canonical(original["constants"][constant]) == canonical(expected),
                    "Static matrix derived constant binding")
            refs["constant"]=constant
        ids=entry["node_ids"]
        require(type(ids) is list and len(ids) == len(actual["steps"]) and
                len(set(ids)) == len(ids) and not seen.intersection(ids),"Static matrix node binding")
        for i,(name,step) in enumerate(zip(ids,actual["steps"])):
            require(name in nodes,"Static matrix node exists")
            node=nodes[name]
            require(node["op"] == step["op"] and node["attrs"] == step["attrs"] and
                    node["inputs"] == [refs[v] for v in step["inputs"]] and len(node["outputs"]) == 1,
                    "Static matrix graph operation binding")
            refs["step_"+str(i)]=node["outputs"][0];seen.add(name)
    return record
