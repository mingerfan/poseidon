"""Registered fused matrix calls over a trusted, metadata-only FX graph."""
import copy, hashlib, math, operator
from pathlib import Path
from benchmark_graph import require, digest, broadcast, dimensions
from model_resources import resolve

POLICY = "poseidon-fx-fused-matrix-draft-v1"
OPERATIONS = ("addmm", "baddbmm", "addbmm")
SCALAR_LIMIT = 1024.

def arguments(args, kwargs, operation, *, method=False):
    require(operation in OPERATIONS, "Registered fused matrix operation")
    names = ("input", "mat1", "mat2") if operation == "addmm" else ("input", "batch1", "batch2")
    allowed = {*names, "alpha", "beta"} | (set() if method else {"out"})
    require(len(args) <= 3 and not (set(kwargs) - allowed), "Fused matrix argument fields")
    values = dict(zip(names, args))
    for name in names:
        if name in kwargs:
            require(name not in values, "Duplicate fused matrix argument")
            values[name] = kwargs[name]
    require(set(values) == set(names), "Fused matrix required operands")
    require(kwargs.get("out") is None, "Fused matrix out mutation is unsupported")
    for name in ("alpha", "beta"):
        scalar = kwargs.get(name, 1)
        require(type(scalar) in (int, float) and math.isfinite(scalar) and abs(scalar) <= SCALAR_LIMIT,
                "Fused matrix finite public scalar budget")
        values[name] = scalar
    return values, names

def plan(operation, shapes, *, alpha=1, beta=1, resource_profile="general-tensors-v1"):
    limits = resolve(resource_profile)
    arguments((0, 0, 0), dict(alpha=alpha, beta=beta), operation)
    require(type(shapes) is list and len(shapes) == 3, "Fused matrix shape inventory")
    for s in shapes:
        require(type(s) is list and len(s) <= limits.rank and
                all(type(n) is int and 0 < n <= limits.axis_size for n in s) and
                math.prod(s) <= limits.tensor_elements, "Fused matrix operand shape budget")
    bias, left, right = shapes
    rank = 2 if operation == "addmm" else 3
    require(len(left) == len(right) == rank, "Fused matrix operand rank")
    require(left[-1] == right[-2], "Fused matrix contraction mismatch")
    if rank == 3: require(left[0] == right[0], "Fused matrix batch mismatch")
    product = ([left[0]] if rank == 3 else []) + [left[-2], right[-1]]
    result = product[-2:] if operation == "addbmm" else product
    dimensions(product, resource_profile=resource_profile)
    dimensions(result, resource_profile=resource_profile)
    try:
        combined = list(broadcast(bias, result))
    except ValueError as error:
        raise ValueError("Fused matrix bias broadcast") from error
    require(combined == result, "Fused matrix bias broadcast")
    return dict(version=POLICY, operation=operation, source_shapes=shapes,
                product_shape=product, result_shape=result, alpha=alpha, beta=beta,
                bias_policy="ignored" if beta == 0 else "scaled_broadcast",
                product_operator="mm" if operation == "addmm" else "bmm",
                reduction_axis=0 if operation == "addbmm" else None,
                resource_profile=resource_profile)

def _state(module):
    import torch
    records = {}
    for name, value in module.state_dict().items():
        require(isinstance(value, torch.Tensor), "Fused matrix public state tensor")
        raw = value.detach().cpu().contiguous().numpy()
        records[name] = dict(shape=list(raw.shape), dtype=raw.dtype.str,
                             sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                             normalized_float64_sha256=hashlib.sha256(raw.astype("<f8").tobytes()).hexdigest())
    return records

def normalize(traced, inputs, *, resource_profile="general-tensors-v1"):
    import torch
    from pytorch_static_trace import ALIAS_POLICY, policy_record
    from pytorch_arithmetic_normalization import clone_public_state
    require(isinstance(traced, torch.fx.GraphModule), "Trusted fused matrix FX graph")
    limits = resolve(resource_profile)
    require(traced.meta.get("fused_matrix_policy") == POLICY, "Explicit fused matrix trace policy")
    require(traced.meta.get("ordered_alias_capture_policy") ==
            policy_record(inputs, resource_profile, policy=ALIAS_POLICY),
            "Ordered fused matrix trace policy")
    source_text = str(traced.graph); state = _state(traced); result = clone_public_state(traced)
    static = {n.name: dict(shape=list(i["shape"]), dtype=i["dtype"]) for n in result.graph.nodes
              if (i := n.meta.get("poseidon_static_tensor")) is not None}
    functions = {torch.addmm:"addmm", torch.baddbmm:"baddbmm", torch.addbmm:"addbmm"}
    roles = {}; entries = []
    def refs(value):
        if isinstance(value, torch.fx.Node):return [value]
        if type(value) in (list, tuple):return [n for v in value for n in refs(v)]
        if type(value) is dict:return [n for v in value.values() for n in refs(v)]
        return []
    def metadata(node, shape):
        node.meta["poseidon_static_tensor"] = dict(policy=ALIAS_POLICY, shape=list(shape), dtype="float64")
        roles[node] = "cipher"
        return node
    for node in list(result.graph.nodes):
        if node.op == "placeholder":roles[node] = "cipher"; continue
        if node.op == "get_attr":roles[node] = "plain"; continue
        op = functions.get(node.target) if node.op == "call_function" else (
             node.target if node.op == "call_method" and node.target in OPERATIONS else None)
        if op is None:
            roles[node] = "cipher" if any(roles.get(n) == "cipher" for n in refs((node.args,node.kwargs))) else "plain"
            continue
        values, names = arguments(node.args, node.kwargs, op, method=node.op == "call_method")
        tensors = [values[n] for n in names]
        require(all(isinstance(v, torch.fx.Node) and v.name in static and static[v.name]["dtype"] == "float64"
                    for v in tensors), "Fused matrix float64 tensor metadata")
        shapes = [static[v.name]["shape"] for v in tensors]
        p = plan(op, shapes, alpha=values["alpha"], beta=values["beta"], resource_profile=resource_profile)
        require(roles.get(tensors[1]) == "cipher" or roles.get(tensors[2]) == "cipher",
                "Fused matrix product requires an encrypted operand")
        require(static.get(node.name) == dict(shape=p["result_shape"], dtype="float64"),
                "Fused matrix result metadata")
        before = dict(fx_node=node.name, call_kind=node.op, operation=op,
                      operands=[v.name for v in tensors], input_kinds=[roles[v] for v in tensors], plan=p)
        emitted = []
        with result.graph.inserting_before(node):
            product = result.graph.call_function(torch.mm if op == "addmm" else torch.bmm,
                                                   args=tuple(tensors[1:]), kwargs={})
            metadata(product, p["product_shape"]); emitted.append(product.name)
            if op == "addbmm":
                product = result.graph.call_function(torch.sum, args=(product,), kwargs=dict(dim=0,keepdim=False))
                metadata(product, p["result_shape"]); emitted.append(product.name)
            if values["alpha"] != 1:
                product = result.graph.call_function(operator.mul, args=(product,values["alpha"]), kwargs={})
                metadata(product, p["result_shape"]); emitted.append(product.name)
        # beta=0 ignores input values in pinned Torch, rather than multiplying
        # them by zero. Earlier source writes remain in their original order.
        node.op = "call_function"; node.target = torch.add
        node.args = (product, 0. if values["beta"] == 0 else tensors[0])
        node.kwargs = {} if values["beta"] == 0 else dict(alpha=values["beta"])
        roles[node] = "cipher"; before["emitted_nodes"] = emitted; entries.append(before)
    count = sum(n.op in ("call_module","call_function","call_method") for n in result.graph.nodes)
    require(count <= limits.original_nodes, "Fused matrix normalization expansion budget")
    result.graph.lint(); result.recompile()
    require(_state(traced) == state == _state(result) and str(traced.graph) == source_text,
            "Fused matrix normalization changed source or public state")
    record = dict(format=POLICY, source_fx_text=source_text,
                  source_fx_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
                  normalized_fx_text=str(result.graph),
                  normalized_fx_sha256=hashlib.sha256(str(result.graph).encode()).hexdigest(),
                  public_state=state, input_signature=copy.deepcopy(inputs), resource_profile=resource_profile,
                  entries=entries, dead_code_policy="retain-source-effects-and-order",
                  source_graph_preserved=True, public_state_preserved=True,
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record["source_inventory"] = inventory(traced)
    record["normalized_inventory"] = inventory(result)
    record["binding_sha256"] = digest(record)
    check_transport(record, inputs=inputs, resource_profile=resource_profile)
    return result, record

def check_record(record, source, normalized):
    require(type(record) is dict and record.get("format") == POLICY and record.get("binding_sha256") ==
            digest({k:v for k,v in record.items() if k != "binding_sha256"}), "Fused matrix record binding")
    expected, evidence = normalize(source, record["input_signature"], resource_profile=record["resource_profile"])
    require(record == evidence and str(normalized.graph) == str(expected.graph) and
            _state(normalized) == _state(expected), "Fused matrix normalization provenance")
    return record


def inventory(module):
    """Data-only FX identity. This text is never imported or evaluated."""
    import re,torch
    from pytorch_static_trace import ALIAS_POLICY
    functions={getattr(torch,n):"torch."+n for n in (*OPERATIONS,"mm","bmm","sum","add")}
    def encode(value):
        if isinstance(value,torch.fx.Node):return dict(kind="fx_value",name=value.name)
        if value is Ellipsis:return dict(kind="ellipsis")
        if value is None:return dict(kind="none")
        if type(value) is bool:return dict(kind="bool",value=value)
        if type(value) in (int,float):
            require(math.isfinite(value),"Finite FX inventory scalar")
            return dict(kind="int" if type(value) is int else "float",value=value)
        if type(value) is str:
            require(len(value)<=256,"FX inventory string budget")
            return dict(kind="str",value=value)
        if isinstance(value,tuple):return dict(kind="tuple",items=[encode(v) for v in value])
        if isinstance(value,list):return dict(kind="list",items=[encode(v) for v in value])
        if isinstance(value,dict):
            require(all(type(k) is str and len(k)<=256 for k in value),"FX inventory dictionary keys")
            return dict(kind="dict",items=[[k,encode(v)] for k,v in value.items()])
        if type(value) is slice:return dict(kind="slice",items=[encode(v) for v in (value.start,value.stop,value.step)])
        if isinstance(value,torch.dtype):return dict(kind="dtype",value=str(value))
        if isinstance(value,torch.device):
            require(value.type in ("cpu","meta") and value.index is None,"FX inventory static device")
            return dict(kind="device",value=value.type)
        raise ValueError("Unregistered FX inventory argument")
    def target(node):
        if node.op!="call_function":return str(node.target)
        if node.target in functions:return functions[node.target]
        value=node.target.__module__+"."+node.target.__name__
        require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*",value) is not None,"FX function identity")
        return value
    return [dict(name=n.name,op=n.op,target=target(n),args=encode(n.args),kwargs=encode(n.kwargs),
                 tensor=copy.deepcopy(n.meta.get("poseidon_static_tensor"))) for n in module.graph.nodes]

def check_transport(record, *, inputs, resource_profile):
    """Verify registered rewrites from JSON; model Python is never executed."""
    from benchmark_graph import canonical
    from pytorch_static_trace import ALIAS_POLICY
    limits=resolve(resource_profile)
    fields={"format","source_fx_text","source_fx_sha256","normalized_fx_text","normalized_fx_sha256",
            "public_state","input_signature","resource_profile","entries","dead_code_policy",
            "source_graph_preserved","public_state_preserved","implementation_sha256",
            "source_inventory","normalized_inventory","binding_sha256"}
    require(type(record) is dict and set(record)==fields and record["format"]==POLICY and
            digest({k:v for k,v in record.items() if k!="binding_sha256"})==record["binding_sha256"],
            "Fused matrix transport binding")
    require(record["input_signature"]==inputs and record["resource_profile"]==resource_profile and
            record["source_graph_preserved"] is True and record["public_state_preserved"] is True and
            record["dead_code_policy"]=="retain-source-effects-and-order" and
            record["implementation_sha256"]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "Fused matrix transport identity")
    require(len(canonical(record))<=limits.json_bytes,"Fused matrix transport budget")
    for text_key,hash_key in (("source_fx_text","source_fx_sha256"),("normalized_fx_text","normalized_fx_sha256")):
        require(type(record[text_key]) is str and hashlib.sha256(record[text_key].encode()).hexdigest()==record[hash_key],
                "Fused matrix transport FX text")
    source,normalized=record["source_inventory"],record["normalized_inventory"]
    cap=limits.expanded_nodes+limits.constants+limits.inputs+limits.outputs
    def check_value(value,seen,depth=0):
        require(type(value) is dict and depth<=16,"Fused matrix FX value")
        kind=value.get("kind")
        if kind=="fx_value":
            require(set(value)=={"kind","name"} and value["name"] in seen,"Fused matrix FX dependency")
        elif kind in ("none","ellipsis"):
            require(set(value)=={"kind"},"Fused matrix FX singleton")
        elif kind in ("bool","int","float","str","dtype","device"):
            require(set(value)=={"kind","value"},"Fused matrix FX literal fields")
            v=value["value"]
            if kind=="bool":require(type(v) is bool,"Fused matrix FX boolean")
            elif kind=="int":require(type(v) is int and abs(v)<=2**53,"Fused matrix FX integer")
            elif kind=="float":require(type(v) is float and math.isfinite(v),"Fused matrix FX floating value")
            elif kind=="device":require(v in ("cpu","meta"),"Fused matrix FX device")
            elif kind=="dtype":require(v in ("torch.float32","torch.float64","torch.int32","torch.int64","torch.bool"),"Fused matrix FX dtype")
            else:require(type(v) is str and len(v)<=256,"Fused matrix FX string")
        else:
            require(kind in ("tuple","list","dict","slice") and set(value)=={"kind","items"} and
                    type(value["items"]) is list and len(value["items"])<=cap,"Fused matrix FX container")
            if kind=="dict":
                keys=[]
                for pair in value["items"]:
                    require(type(pair) is list and len(pair)==2 and type(pair[0]) is str,"Fused matrix FX mapping")
                    keys.append(pair[0]);check_value(pair[1],seen,depth+1)
                require(len(set(keys))==len(keys),"Fused matrix FX duplicate key")
            else:
                if kind=="slice":require(len(value["items"])==3,"Fused matrix FX slice")
                for v in value["items"]:check_value(v,seen,depth+1)
    def check_graph(nodes):
        require(type(nodes) is list and len(nodes)<=cap,"Fused matrix FX node budget")
        seen=set()
        for node in nodes:
            require(type(node) is dict and set(node)=={"name","op","target","args","kwargs","tensor"} and
                    type(node["name"]) is str and 0<len(node["name"])<=256 and node["name"] not in seen and
                    node["op"] in ("placeholder","get_attr","call_function","call_method","call_module","output") and
                    type(node["target"]) is str and 0<len(node["target"])<=256,"Fused matrix FX node")
            check_value(node["args"],seen);check_value(node["kwargs"],seen)
            t=node["tensor"]
            if t is not None:
                require(type(t) is dict and set(t)=={"policy","shape","dtype"} and t["policy"]==ALIAS_POLICY and
                        type(t["shape"]) is list and len(t["shape"])<=limits.rank and
                        all(type(n) is int and 0<n<=limits.axis_size for n in t["shape"]) and
                        math.prod(t["shape"])<=(limits.constant_elements if node["op"]=="get_attr" else limits.tensor_elements) and
                        t["dtype"] in ("float64","float32","int32","int64","bool"),"Fused matrix FX tensor metadata")
            seen.add(node["name"])
    check_graph(source);check_graph(normalized)
    placeholders=[n for n in source if n["op"]=="placeholder"]
    require(len(placeholders)==len(inputs) and all(n["tensor"]==dict(policy=ALIAS_POLICY,shape=s["shape"],dtype="float64")
            for n,s in zip(placeholders,inputs)),"Fused matrix FX input signatures")
    entries=record["entries"]
    require(type(entries) is list and len(entries)<=limits.original_nodes,"Fused matrix transport entries")
    table={};new_names=set()
    for e in entries:
        require(type(e) is dict and set(e)=={"fx_node","call_kind","operation","operands","input_kinds","plan","emitted_nodes"} and
                type(e["fx_node"]) is str and e["fx_node"] not in table and
                type(e["emitted_nodes"]) is list and all(type(n) is str for n in e["emitted_nodes"]),
                "Fused matrix transport entry")
        table[e["fx_node"]]=e
        require(not new_names.intersection(e["emitted_nodes"]) and len(set(e["emitted_nodes"]))==len(e["emitted_nodes"]),
                "Fused matrix generated node inventory")
        new_names.update(e["emitted_nodes"])
    require(not new_names.intersection(n["name"] for n in source),"Fused matrix generated name collision")
    def ref(name):return dict(kind="fx_value",name=name)
    def scalar(value):return dict(kind="float" if type(value) is float else "int",value=value)
    def args(values):return dict(kind="tuple",items=values)
    def kwargs(values):return dict(kind="dict",items=[[k,v] for k,v in values.items()])
    def dependencies(value):
        if value["kind"]=="fx_value":return [value["name"]]
        if value["kind"]=="dict":return [n for _,v in value["items"] for n in dependencies(v)]
        if value["kind"] in ("tuple","list","slice"):return [n for v in value["items"] for n in dependencies(v)]
        return []
    def decode_literal(value):
        require(value["kind"] in ("int","float","none"),"Fused matrix static scalar or null")
        return None if value["kind"]=="none" else value["value"]
    source_by_name={n["name"]:n for n in source};roles={};expected=[];visited=set()
    for n in source:
        if n["op"]=="placeholder":roles[n["name"]]="cipher"
        elif n["op"]=="get_attr":roles[n["name"]]="plain"
        else:roles[n["name"]]="cipher" if any(roles[r]=="cipher" for r in dependencies(n["args"])+dependencies(n["kwargs"])) else "plain"
        operation=n["target"].removeprefix("torch.") if n["op"]=="call_function" else n["target"] if n["op"]=="call_method" else None
        fused=operation in OPERATIONS and (n["op"]=="call_method" or n["target"]=="torch."+operation)
        if not fused:expected.append(copy.deepcopy(n));continue
        require(n["name"] in table,"Fused matrix missing source task")
        e=table[n["name"]];visited.add(n["name"])
        require(n["args"]["kind"]=="tuple" and n["kwargs"]["kind"]=="dict","Fused matrix call arguments")
        positional=[v["name"] if v["kind"]=="fx_value" else decode_literal(v) for v in n["args"]["items"]]
        options={k:v["name"] if v["kind"]=="fx_value" else decode_literal(v) for k,v in n["kwargs"]["items"]}
        v,names=arguments(positional,options,operation,method=n["op"]=="call_method")
        operand=[v[k] for k in names]
        require(all(type(r) is str and r in source_by_name and source_by_name[r]["tensor"] is not None and
                    source_by_name[r]["tensor"]["dtype"]=="float64" for r in operand),"Fused matrix transport operands")
        p=plan(operation,[source_by_name[r]["tensor"]["shape"] for r in operand],
               alpha=v["alpha"],beta=v["beta"],resource_profile=resource_profile)
        require(e["call_kind"]==n["op"] and e["operation"]==operation and e["operands"]==operand and
                e["input_kinds"]==[roles[r] for r in operand] and e["plan"]==p and
                "cipher" in e["input_kinds"][1:] and
                n["tensor"]==dict(policy=ALIAS_POLICY,shape=p["result_shape"],dtype="float64"),
                "Fused matrix transport plan")
        iterator=iter(e["emitted_nodes"])
        def emit(target,values,options,shape):
            name=next(iterator,None)
            require(name is not None,"Fused matrix generated operation missing")
            expected.append(dict(name=name,op="call_function",target=target,args=args(values),kwargs=kwargs(options),
                                 tensor=dict(policy=ALIAS_POLICY,shape=shape,dtype="float64")))
            return name
        product=emit("torch."+p["product_operator"],[ref(r) for r in operand[1:]],{},p["product_shape"])
        if operation=="addbmm":product=emit("torch.sum",[ref(product)],dict(dim=scalar(0),keepdim=dict(kind="bool",value=False)),p["result_shape"])
        if v["alpha"]!=1:product=emit("_operator.mul",[ref(product),scalar(v["alpha"])],{},p["result_shape"])
        require(next(iterator,None) is None,"Fused matrix extra generated operation")
        changed=copy.deepcopy(n);changed.update(op="call_function",target="torch.add",
          args=args([ref(product),scalar(0.) if v["beta"]==0 else ref(operand[0])]),
          kwargs=kwargs({} if v["beta"]==0 else dict(alpha=scalar(v["beta"]))))
        expected.append(changed);roles[n["name"]]="cipher"
    require(visited==set(table) and canonical(expected)==canonical(normalized),"Fused matrix transported rewrite differs")
    require(sum(n["op"] in ("call_function","call_method","call_module") for n in normalized)<=limits.original_nodes,
            "Fused matrix transported expansion budget")
    return record
