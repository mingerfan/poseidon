"""Static chunk/unbind views over trusted FX graphs; no generated DSL or backend."""
import copy,hashlib,math,operator
from pathlib import Path
from benchmark_graph import require,digest,canonical
from model_resources import resolve
POLICY="poseidon-pytorch-static-partition-v1"
OPERATIONS=("chunk","unbind")
def arguments(args,kwargs,operation,*,method=False):
    require(operation in OPERATIONS,"Registered partition operation")
    names=("input","chunks","dim")if operation=="chunk"else("input","dim")
    require(len(args)<=len(names)and not(set(kwargs)-set(names))and
            (not method or "input"not in kwargs),"Static partition arguments")
    values=dict(zip(names,args))
    for n,v in kwargs.items():
        require(n not in values,"Duplicate static partition argument");values[n]=v
    require("input"in values and values["input"]is not None and
            (operation!="chunk"or"chunks"in values),"Static partition operands")
    values.setdefault("dim",0)
    return dict(input=values["input"],axis=values["dim"],chunks=values.get("chunks"))
def plan(shape,operation,*,axis=0,chunks=None,resource_profile="general-tensors-v1"):
    limits=resolve(resource_profile)
    require(operation in OPERATIONS,"Registered partition operation")
    require(type(shape)in(list,tuple)and 1<=len(shape)<=limits.rank and
            all(type(n)is int and 0<n<=limits.axis_size for n in shape)and
            math.prod(shape)<=limits.tensor_elements,"Static partition source shape budget")
    require(type(axis)is int and -len(shape)<=axis<len(shape),"Static partition axis")
    axis%=len(shape);width=shape[axis]
    if operation=="chunk":
        require(type(chunks)is int and 1<=chunks<=2**53,"Static chunk count")
        size=(width+chunks-1)//chunks
        count=(width+size-1)//size
        require(count<=limits.original_nodes,"Static partition expansion budget")
        sections=[min(size,width-i*size)for i in range(count)]
    else:
        require(chunks is None,"Unbind has no chunk count")
        # The ordered alias core currently represents nonempty tensor views.
        # Scalar views need a separate alias contract rather than relabelling.
        require(len(shape)>=2,"Scalar unbind views are not registered")
        count=width;require(count<=limits.original_nodes,"Static partition expansion budget")
        sections=[1]*width
    require(count<=limits.original_nodes,"Static partition expansion budget")
    offset=0;ports=[]
    for i,n in enumerate(sections):
        result=[*shape[:axis],n,*shape[axis+1:]]if operation=="chunk"else list(shape[:axis])+list(shape[axis+1:])
        ports.append(dict(port=i,start=offset,stop=offset+n,result_shape=result));offset+=n
    return dict(policy=POLICY,operation=operation,source_shape=list(shape),axis=axis,
        requested_chunks=chunks,sections=sections,ports=ports,resource_profile=resource_profile,
        view_semantics="shared source storage",lowering="registered_basic_index")
def index_for(p,port):
    v=p["ports"][port];index=[slice(None)]*len(p["source_shape"])
    index[p["axis"]]=slice(v["start"],v["stop"],1)if p["operation"]=="chunk"else port
    return tuple(index)
def _index_literal(index):
    def value(v):
        if v is None:return dict(kind="none")
        if type(v)is int:return dict(kind="int",value=v)
        require(type(v)is slice,"Static partition index")
        return dict(kind="slice",items=[value(x)for x in(v.start,v.stop,v.step)])
    return dict(kind="tuple",items=[value(v)for v in index])
def normalize(traced,inputs,*,resource_profile="general-tensors-v1"):
    import torch
    from pytorch_static_trace import ALIAS_POLICY,policy_record
    from pytorch_arithmetic_normalization import clone_public_state
    from pytorch_fused_matrix_normalization import inventory
    require(isinstance(traced,torch.fx.GraphModule)and traced.meta.get("partition_source_policy")==POLICY,
            "Explicit partition trace policy")
    require(traced.meta.get("ordered_alias_capture_policy")==policy_record(inputs,resource_profile,policy=ALIAS_POLICY),
            "Partition ordered trace policy")
    limits=resolve(resource_profile);result=clone_public_state(traced);entries=[]
    for node in list(result.graph.nodes):
        operation=next((n for n in OPERATIONS if
            node.op=="call_function"and node.target is getattr(torch,n)or
            node.op=="call_method"and node.target==n),None)
        if operation is None:continue
        values=arguments(node.args,node.kwargs,operation,method=node.op=="call_method")
        source=values["input"];require(isinstance(source,torch.fx.Node),"Partition FX source")
        metadata=source.meta.get("poseidon_static_tensor")
        require(type(metadata)is dict and metadata.get("policy")==ALIAS_POLICY,"Partition static tensor metadata")
        selected=plan(metadata["shape"],operation,axis=values["axis"],chunks=values["chunks"],
                      resource_profile=resource_profile)
        users=list(node.users);ports=[]
        for user in users:
            require(user.op=="call_function"and user.target is operator.getitem and
                    len(user.args)==2 and user.args[0]is node and type(user.args[1])is int and not user.kwargs,
                    "Static partition tuple consumer")
            port=user.args[1]
            require(0<=port<len(selected["ports"]),"Static partition port")
            require(user.meta.get("poseidon_static_tensor")==dict(policy=ALIAS_POLICY,
                    shape=selected["ports"][port]["result_shape"],dtype=metadata["dtype"]),"Partition port metadata")
            ports.append(dict(name=user.name,port=port))
        require(sorted(v["port"]for v in ports)==list(range(len(selected["ports"]))),"Partition tuple inventory")
        for user in users:user.args=(source,index_for(selected,user.args[1]))
        result.graph.erase_node(node)
        entries.append(dict(source_node=node.name,input=source.name,plan=selected,ports=ports))
    result.graph.lint();result.recompile()
    require(sum(n.op in("call_function","call_method","call_module")for n in result.graph.nodes)<=limits.original_nodes,
            "Normalized partition FX budget")
    text=str(traced.graph);normalized=str(result.graph)
    record=dict(format=POLICY,input_signature=copy.deepcopy(inputs),resource_profile=resource_profile,
        source_fx_text=text,source_fx_sha256=hashlib.sha256(text.encode()).hexdigest(),
        normalized_fx_text=normalized,normalized_fx_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
        source_inventory=inventory(traced),normalized_inventory=inventory(result),entries=entries,
        source_order_preserved=True,public_state_preserved=True,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record["binding"]=digest(record);check_transport(record,inputs=inputs,resource_profile=resource_profile)
    return result,record
def check_transport(record,*,inputs,resource_profile):
    """Check the complete registered rewrite as data; never run source text."""
    from pytorch_static_trace import ALIAS_POLICY
    fields={"format","input_signature","resource_profile","source_fx_text","source_fx_sha256",
        "normalized_fx_text","normalized_fx_sha256","source_inventory","normalized_inventory","entries",
        "source_order_preserved","public_state_preserved","implementation_sha256","binding"}
    limits=resolve(resource_profile)
    require(type(record)is dict and set(record)==fields and record["format"]==POLICY and
            record["binding"]==digest({k:v for k,v in record.items()if k!="binding"}),"Partition record binding")
    require(record["input_signature"]==inputs and record["resource_profile"]==resource_profile and
            record["source_order_preserved"]is True and record["public_state_preserved"]is True and
            record["implementation_sha256"]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "Partition source identity")
    require(len(canonical(record))<=limits.json_bytes,"Partition record budget")
    for t,h in(("source_fx_text","source_fx_sha256"),("normalized_fx_text","normalized_fx_sha256")):
        require(type(record[t])is str and hashlib.sha256(record[t].encode()).hexdigest()==record[h],
                "Partition FX text binding")
    source=record["source_inventory"];normalized=record["normalized_inventory"]
    cap=limits.original_nodes+limits.constants+limits.inputs+limits.outputs
    require(type(source)is list and type(normalized)is list and 1<=len(source)<=cap and
            1<=len(normalized)<=cap,"Partition FX inventory budget")
    require(all(type(n)is dict and set(n)=={"name","op","target","args","kwargs","tensor"}and
            type(n["name"])is str for n in source+normalized),"Partition FX inventory fields")
    nodes={n["name"]:n for n in source};require(len(nodes)==len(source),"Partition FX names")
    require(type(record["entries"])is list and len(record["entries"])<=limits.original_nodes,"Partition entries budget")
    removed=set();changed={}
    def literal(v):
        require(type(v)is dict,"Partition argument literal")
        if v==dict(kind="none"):return None
        if v.get("kind")=="int"and set(v)=={"kind","value"}and type(v["value"])is int:return v["value"]
        if v.get("kind")=="fx_value"and set(v)=={"kind","name"}and v["name"]in nodes:return v
        raise ValueError("Unregistered partition argument")
    for entry in record["entries"]:
        require(type(entry)is dict and set(entry)=={"source_node","input","plan","ports"}and
                entry["source_node"]in nodes and entry["input"]in nodes and
                entry["source_node"]not in removed,"Partition entry source")
        node=nodes[entry["source_node"]];p=entry["plan"]
        operation=next((n for n in OPERATIONS if(node["op"],node["target"])in
                       (("call_function","torch."+n),("call_method",n))),None)
        require(operation is not None,"Partition registered call")
        a=node["args"];k=node["kwargs"]
        require(a.get("kind")=="tuple"and set(a)=={"kind","items"}and
                k.get("kind")=="dict"and set(k)=={"kind","items"},"Partition source arguments")
        kwargs={}
        for pair in k["items"]:
            require(type(pair)is list and len(pair)==2 and type(pair[0])is str and pair[0]not in kwargs,
                    "Partition source keywords");kwargs[pair[0]]=literal(pair[1])
        args=tuple(literal(v)for v in a["items"])
        values=arguments(args,kwargs,operation,method=node["op"]=="call_method")
        require(values["input"]==dict(kind="fx_value",name=entry["input"]),"Partition source operand")
        meta=nodes[entry["input"]]["tensor"]
        require(type(meta)is dict and meta.get("policy")==ALIAS_POLICY,"Partition source tensor binding")
        expected=plan(meta["shape"],operation,axis=values["axis"],chunks=values["chunks"],resource_profile=resource_profile)
        require(p==expected,"Partition plan recomputation")
        ports=entry["ports"];require(type(ports)is list and
            sorted(v.get("port",-1)for v in ports if type(v)is dict)==list(range(len(p["ports"]))),"Partition ports")
        removed.add(entry["source_node"])
        for v in ports:
            require(set(v)=={"name","port"}and v["name"]in nodes and v["name"]not in changed,"Partition port identity")
            port=nodes[v["name"]];i=v["port"]
            require(type(i)is int and port["op"]=="call_function"and port["target"]=="_operator.getitem"and
                port["args"]==dict(kind="tuple",items=[dict(kind="fx_value",name=node["name"]),dict(kind="int",value=i)])and
                port["kwargs"]==dict(kind="dict",items=[])and port["tensor"]==dict(policy=ALIAS_POLICY,
                    shape=p["ports"][i]["result_shape"],dtype=meta["dtype"]),"Partition original port binding")
            changed[v["name"]]=dict(port,args=dict(kind="tuple",items=[
                dict(kind="fx_value",name=entry["input"]),_index_literal(index_for(p,i))]))
    # No opaque tuple producer or unlisted consumer can survive this projection.
    require(not removed.intersection(changed),"Partition overlapping rewrite")
    expected=[changed.get(n["name"],n)for n in source if n["name"]not in removed]
    require(expected==normalized,"Partition normalized graph differs")
    declared={v["source_node"]for v in record["entries"]}
    actual={n["name"]for n in source if any((n["op"],n["target"])in
        (("call_function","torch."+op),("call_method",op))for op in OPERATIONS)}
    require(declared==actual,"Partition missing rewrite")
    return record
