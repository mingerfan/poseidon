"""Static FX tensordot normalization with reproducible public-state provenance."""
import copy,hashlib
from pathlib import Path
from benchmark_graph import require,digest,canonical
from model_resources import resolve
from tensor_contraction_plan import plan
from gather_fx_normalization import public_arrays,_public_state
POLICY="poseidon-pytorch-static-tensor-contraction-v1"
TARGETS=("torch.functional.tensordot","torch.tensordot")

def operation(kind,target):
    import torch
    return kind=="call_function"and target is torch.tensordot

def arguments(args,kwargs):
    names=("a","b","dims","out")
    require(type(args)in(tuple,list)and len(args)<=4 and type(kwargs)is dict and
        not(set(kwargs)-set(names)),"Contraction call fields")
    values=dict(dims=2,out=None)
    for name,value in zip(names,args):
        require(name not in kwargs,"Duplicate contraction call field");values[name]=value
    values.update(kwargs)
    require("a"in values and "b"in values and values["out"]is None,"Contraction operands and immutable output")
    return values["a"],values["b"],values["dims"]

def public_dimensions(value,*,resource_profile):
    import numpy as np
    limits=resolve(resource_profile)
    if type(value)in(int,list,tuple):return value
    require(isinstance(value,np.ndarray)and value.dtype in(np.dtype("int32"),np.dtype("int64"))and
        0<value.size<=2*limits.rank and value.ndim<=limits.rank,"Contraction fixed integer dimensions")
    if value.size==1:return int(value.reshape(-1)[0])
    require(value.ndim==2 and value.shape[0]==2,"Contraction public axis tensor shape")
    return value.tolist()

def metadata(kind,target,args,kwargs,*,resolve_value,resource_profile,resolve_public):
    if not operation(kind,target):return None
    import torch
    a,b,dims=arguments(args,kwargs);a,b=resolve_value(a),resolve_value(b)
    require(isinstance(a,torch.Tensor)and isinstance(b,torch.Tensor)and
        a.dtype==b.dtype and a.dtype in(torch.float32,torch.float64),"Contraction metadata dtype")
    if type(dims)not in(int,list,tuple):
        raw=resolve_public(dims)
        require(isinstance(raw,torch.Tensor)and raw.device.type=="cpu"and
            raw.dtype in(torch.int32,torch.int64),"Contraction immutable public dimensions")
        dims=public_dimensions(raw.detach().contiguous().numpy(),resource_profile=resource_profile)
    p=plan(list(a.shape),list(b.shape),dims,resource_profile=resource_profile)
    return torch.empty(p["output_shape"],device="meta",dtype=a.dtype)

def implementation():
    base=Path(__file__).parent
    own={name:hashlib.sha256((base/name).read_bytes()).hexdigest()for name in
        ("contraction_fx_normalization.py","tensor_contraction_plan.py")}
    shared=Path(__import__("gather_fx_normalization").__file__)
    own["shared:"+shared.name]=hashlib.sha256(shared.read_bytes()).hexdigest()
    return own

def normalized_inventory(source,arrays,*,resource_profile):
    import numpy as np
    from pytorch_public_expression_flow import _decode,Ref
    from pytorch_static_trace import ALIAS_POLICY
    limits=resolve(resource_profile);capacity=limits.expanded_nodes+limits.constants+limits.inputs+limits.outputs
    require(type(source)is list and len(source)<=capacity,"Contraction FX capacity")
    names={n["name"]:Ref(n["name"])for n in source};used=set(names);rows={};public={};result=[];entries=[]
    def enc(value):
        if type(value)is Ref:return dict(kind="fx_value",name=value.name)
        if type(value)is tuple:return dict(kind="tuple",items=[enc(x)for x in value])
        if type(value)is int:return dict(kind="int",value=value)
        require(value is None,"Contraction normalized literal");return dict(kind="none")
    def fresh(base):
        prefix=base[:36]+"_contract_";i=0
        while prefix+str(i)in used:i+=1
        name=prefix+str(i);used.add(name);return name
    def emit(name,kind,target,args,shape):
        node=dict(name=name,op=kind,target=target,args=enc(args),kwargs=dict(kind="dict",items=[]),
            tensor=dict(policy=ALIAS_POLICY,shape=list(shape),dtype="float64"))
        result.append(node);rows[name]=node;return Ref(name)
    for node in source:
        require(type(node)is dict and node["name"]not in rows,"Contraction unique FX node")
        args=_decode(node["args"],names);kwargs=_decode(node["kwargs"],names)
        if node["op"]=="get_attr"and node["target"]in arrays:public[node["name"]]=np.asarray(arrays[node["target"]])
        selected=node["op"]=="call_function"and node["target"]in TARGETS
        if not selected:
            result.append(copy.deepcopy(node));rows[node["name"]]=node;continue
        a,b,dims=arguments(args,kwargs)
        require(type(a)is Ref and type(b)is Ref and a.name in rows and b.name in rows,"Contraction bound operands")
        dimension_ref=None
        if type(dims)is Ref:
            require(dims.name in public,"Contraction dimensions must be immutable public data")
            dimension_ref=dims.name;dims=public_dimensions(public[dims.name],resource_profile=resource_profile)
        am,bm=rows[a.name]["tensor"],rows[b.name]["tensor"]
        require(type(am)is dict and type(bm)is dict and am["dtype"]==bm["dtype"]=="float64",
            "Contraction normalized float64 operands")
        p=plan(am["shape"],bm["shape"],dims,resource_profile=resource_profile)
        require(canonical(node["tensor"])==canonical(dict(policy=ALIAS_POLICY,shape=p["output_shape"],dtype="float64")),
            "Contraction FX result shape")
        begin=len(result);values={"a":a,"b":b}
        for step in p["steps"]:
            op=step["op"]
            if op=="matmul":
                values["result"]=emit(fresh(node["name"]),"call_function","torch.matmul",
                    (values["a"],values["b"]),[p["matrix_shapes"][0][0],p["matrix_shapes"][1][1]])
                continue
            name=step["operand"]
            if op=="clone":
                values[name]=emit(node["name"],"call_method","clone",(values[name],),p["output_shape"]);continue
            attrs=step["attrs"]
            shape=(list(step["attrs"]["shape"])if op!="permute"else
                [rows[values[name].name]["tensor"]["shape"][i]for i in attrs["axes"]])
            if op=="expand":values[name]=emit(fresh(node["name"]),"call_method","expand",(values[name],tuple(attrs["shape"])),shape)
            else:
                target="torch.permute"if op=="permute"else"torch.reshape"
                values[name]=emit(fresh(node["name"]),"call_function",target,
                    (values[name],tuple(attrs["axes"]if op=="permute"else attrs["shape"])),shape)
        entries.append(dict(source_node=node["name"],a=a.name,b=b.name,dimension_ref=dimension_ref,
            plan=p,generated_nodes=[n["name"]for n in result[begin:]]))
        require(len(result)<=capacity,"Contraction FX expansion budget")
    require(len(result)<=capacity,"Contraction normalized FX capacity")
    return result,entries

def check(record,inputs,arrays,*,resource_profile,expected_source_inventory=None):
    fields={"format","input_signature","resource_profile","source_inventory","normalized_inventory",
        "entries","public_state","implementation","source_order_preserved","public_state_preserved","binding"}
    require(type(record)is dict and set(record)==fields and record["format"]==POLICY and
        record["binding"]==digest({k:v for k,v in record.items()if k!="binding"}),"Contraction provenance binding")
    require(canonical(record["input_signature"])==canonical(inputs)and record["resource_profile"]==resource_profile and
        canonical(record["implementation"])==canonical(implementation())and
        canonical(record["public_state"])==canonical(_public_state(arrays))and
        record["source_order_preserved"]is True and record["public_state_preserved"]is True,
        "Contraction provenance identity")
    if expected_source_inventory is not None:
        require(canonical(record["source_inventory"])==canonical(expected_source_inventory),"Contraction source inventory bridge")
    expected,entries=normalized_inventory(record["source_inventory"],arrays,resource_profile=resource_profile)
    require(canonical(record["normalized_inventory"])==canonical(expected)and
        canonical(record["entries"])==canonical(entries),"Contraction mathematical normalization")
    return expected

def normalize(traced,inputs,*,resource_profile):
    import torch
    from pytorch_fused_matrix_normalization import inventory
    from pytorch_arithmetic_normalization import clone_public_state
    require(isinstance(traced,torch.fx.GraphModule)and traced.meta.get("tensor_contraction_source_policy")==POLICY,
        "Contraction explicit normalization policy")
    source=inventory(traced);arrays=public_arrays(traced)
    expected,entries=normalized_inventory(source,arrays,resource_profile=resource_profile)
    result=clone_public_state(traced);by_name={n.name:n for n in result.graph.nodes}
    expected_by={n["name"]:n for n in expected}
    for entry in entries:
        original=by_name[entry["source_node"]];generated=iter(entry["generated_nodes"]);values={"a":by_name[entry["a"]],"b":by_name[entry["b"]]}
        p=entry["plan"]
        with result.graph.inserting_before(original):
            for step in p["steps"]:
                op=step["op"]
                if op=="clone":
                    n=result.graph.call_method("clone",(values["result"],));n.meta=copy.deepcopy(original.meta)
                    name=next(generated);cloned=n;clone_name=name;continue
                if op=="matmul":kind,target,args="call_function",torch.matmul,(values["a"],values["b"])
                else:
                    operand=values[step["operand"]];attrs=step["attrs"]
                    if op=="expand":kind,target,args="call_method","expand",(operand,tuple(attrs["shape"]))
                    else:
                        kind,target,args="call_function",torch.permute if op=="permute"else torch.reshape,(operand,tuple(attrs["axes"]if op=="permute"else attrs["shape"]))
                name=next(generated);n=result.graph.create_node(kind,target,args,{},name=name)
                n.meta=copy.deepcopy(original.meta);n.meta["poseidon_static_tensor"]=copy.deepcopy(expected_by[name]["tensor"])
                values["result"if op=="matmul"else step["operand"]]=n;by_name[name]=n
        original.replace_all_uses_with(cloned);result.graph.erase_node(original);cloned.name=clone_name;by_name[clone_name]=cloned
    result.graph.lint();result.recompile()
    require(canonical(inventory(result))==canonical(expected),"Contraction generated FX inventory")
    require(canonical(inventory(traced))==canonical(source),"Contraction original graph changed")
    require(canonical(_public_state(public_arrays(result)))==canonical(_public_state(arrays)),"Contraction original public state changed")
    record=dict(format=POLICY,input_signature=copy.deepcopy(inputs),resource_profile=resource_profile,
        source_inventory=source,normalized_inventory=expected,entries=entries,public_state=_public_state(arrays),
        implementation=implementation(),source_order_preserved=True,public_state_preserved=True)
    record["binding"]=digest(record);check(record,inputs,arrays,resource_profile=resource_profile)
    return result,record
