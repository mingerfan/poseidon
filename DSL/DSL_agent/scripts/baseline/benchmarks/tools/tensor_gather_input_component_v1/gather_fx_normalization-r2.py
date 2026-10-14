"""Registered public Tensor.gather FX normalization and replayable provenance."""
import copy
import hashlib
import math
from pathlib import Path
from benchmark_graph import require,digest
from model_resources import resolve
from tensor_gather_plan import plan

POLICY = "poseidon-pytorch-public-tensor-gather-v1"

def operation(kind,target):
    import torch
    return (kind=="call_function" and target is torch.gather
            or kind=="call_method" and target=="gather")

def arguments(args,kwargs):
    names=("input","dim","index","sparse_grad","out")
    require(len(args)<=3 and not(set(kwargs)-set(names)),"Gather arguments")
    values=dict(zip(names,(None,None,None,False,None)))
    for name,value in zip(names,args):
        require(name not in kwargs,"Duplicate gather argument")
        values[name]=value
    values.update(kwargs)
    require(type(values["dim"]) is int,"Gather static axis")
    require(values["sparse_grad"] is False and values["out"] is None,"Gather inference and immutable output")
    return values["input"],values["dim"],values["index"]

def metadata(kind,target,args,kwargs,*,resolve_value,resource_profile):
    if not operation(kind,target):return None
    import torch
    x,axis,index=arguments(args,kwargs);x=resolve_value(x);index=resolve_value(index)
    limits=resolve(resource_profile)
    require(isinstance(x,torch.Tensor)and x.dtype in(torch.float32,torch.float64)
            and isinstance(index,torch.Tensor)and index.dtype==torch.int64,"Gather tensor metadata dtype")
    shape=list(x.shape);target_shape=list(index.shape)
    require(1<=len(shape)<=limits.rank and len(target_shape)==len(shape)
            and -len(shape)<=axis<len(shape)
            and all(0<v<=limits.axis_size for v in shape+target_shape)
            and math.prod(shape)<=limits.tensor_elements and math.prod(target_shape)<=limits.tensor_elements,
            "Gather metadata shape budget")
    axis%=len(shape)
    require(all(i==axis or target_shape[i]<=shape[i]for i in range(len(shape))),"Gather nonindexed dimension")
    return torch.empty(target_shape,device="meta",dtype=x.dtype)

def _public_state(arrays):
    import numpy as np
    return {name:dict(shape=list(np.asarray(a).shape),dtype=np.asarray(a).dtype.str,
       sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
       for name,a in sorted(arrays.items())}

def public_arrays(module):
    return {name:value.detach().cpu().contiguous().numpy().copy()
            for name,value in module.state_dict().items()}

def implementation():
    base=Path(__file__).parent
    return {name:hashlib.sha256((base/name).read_bytes()).hexdigest()
            for name in ("gather_fx_normalization.py","tensor_gather_plan.py")}

def normalized_inventory(source,arrays,*,resource_profile):
    import numpy as np
    from pytorch_public_expression_flow import _decode,Ref
    from pytorch_static_trace import ALIAS_POLICY
    limits=resolve(resource_profile);capacity=limits.expanded_nodes+limits.constants+limits.inputs+limits.outputs
    require(type(source)is list and len(source)<=capacity,"Gather FX inventory capacity")
    names={n["name"]:Ref(n["name"])for n in source};rows={};public={};result=[];entries=[]
    used=set(names)
    def enc(value):
        if type(value)is Ref:return dict(kind="fx_value",name=value.name)
        if type(value)is tuple:return dict(kind="tuple",items=[enc(x)for x in value])
        if type(value)is int:return dict(kind="int",value=value)
        require(value is None,"Gather normalized literal");return dict(kind="none")
    def fresh(base):
        prefix=base[:36]+"_gather_";i=0
        while prefix+str(i)in used:i+=1
        name=prefix+str(i);used.add(name);return name
    def emit(name,kind,target,args,shape):
        node=dict(name=name,op=kind,target=target,args=enc(args),kwargs=dict(kind="dict",items=[]),
         tensor=dict(policy=ALIAS_POLICY,shape=list(shape),dtype="float64"))
        result.append(node);rows[name]=node;return Ref(name)
    for node in source:
        require(type(node)is dict and node["name"]not in rows,"Gather unique FX node")
        args=_decode(node["args"],names);kwargs=_decode(node["kwargs"],names)
        if node["op"]=="get_attr"and node["target"]in arrays:
            public[node["name"]]=np.asarray(arrays[node["target"]])
        if node["op"]=="call_function"and node["target"] in("torch.tensor","torch._VariableFunctionsClass.tensor"):
            if args and kwargs.get("dtype")=="torch.int64" and not kwargs.get("requires_grad",False):
                try:
                    _,data=__import__("tensor_gather_plan").index_tensor(args[0],
                        max_rank=limits.rank,max_elements=limits.tensor_elements)
                    value=np.asarray(args[0],dtype=np.int64);public[node["name"]]=value
                except (ValueError,TypeError,OverflowError):pass
        selected=node["op"]=="call_function"and node["target"] in("torch.gather","torch._VariableFunctionsClass.gather")or node["op"]=="call_method"and node["target"]=="gather"
        if not selected:
            result.append(copy.deepcopy(node));rows[node["name"]]=node;continue
        x,axis,index=arguments(args,kwargs)
        require(type(x)is Ref and x.name in rows and type(index)is Ref and index.name in public,
                "Gather index must be immutable public data")
        data=public[index.name];xmeta=rows[x.name]["tensor"]
        require(data.dtype==np.int64 and type(xmeta)is dict and xmeta["dtype"]=="float64",
                "Gather public int64 and float64 computation")
        planned=plan(list(xmeta["shape"]),data.tolist(),axis,resource_profile=resource_profile)
        require(node["tensor"]==dict(policy=ALIAS_POLICY,shape=planned["result_shape"],dtype="float64"),
                "Gather FX result shape")
        begin=len(result);flat=emit(fresh(node["name"]),"call_function","torch.reshape",
                                 (x,(math.prod(planned["source_shape"]),)),[math.prod(planned["source_shape"])])
        parts=[emit(fresh(node["name"]),"call_function","torch.narrow",(flat,0,offset,1),[1])
               for offset in planned["steps"][1]["attrs"]["indices"]]
        joined=parts[0]if len(parts)==1 else emit(fresh(node["name"]),"call_function","torch.cat",
                                               (tuple(parts),0),[len(parts)])
        shaped=emit(fresh(node["name"]),"call_function","torch.reshape",
                    (joined,tuple(planned["result_shape"])),planned["result_shape"])
        emit(node["name"],"call_method","clone",(shaped,),planned["result_shape"])
        require(len(result)<=capacity,"Gather normalized FX expansion budget")
        entries.append(dict(source_node=node["name"],input=x.name,index=index.name,
          plan=planned,generated_nodes=[n["name"]for n in result[begin:]]))
    require(len(result)<=capacity,"Gather normalized FX capacity")
    return result,entries

def check(record,inputs,arrays,*,resource_profile,expected_source_inventory=None):
    fields={"format","input_signature","resource_profile","source_inventory","normalized_inventory",
            "entries","public_state","implementation","source_order_preserved","public_state_preserved","binding"}
    require(type(record)is dict and set(record)==fields and record["format"]==POLICY
            and record["binding"]==digest({k:v for k,v in record.items()if k!="binding"}),"Gather provenance binding")
    require(record["input_signature"]==inputs and record["resource_profile"]==resource_profile
            and record["implementation"]==implementation()and record["public_state"]==_public_state(arrays)
            and record["source_order_preserved"]is True and record["public_state_preserved"]is True,
            "Gather provenance identity")
    if expected_source_inventory is not None:
        require(record["source_inventory"]==expected_source_inventory,"Gather source inventory bridge")
    expected,entries=normalized_inventory(record["source_inventory"],arrays,resource_profile=resource_profile)
    require(record["normalized_inventory"]==expected and record["entries"]==entries,"Gather mathematical normalization")
    return expected

def normalize(traced,inputs,*,resource_profile):
    import torch
    from pytorch_fused_matrix_normalization import inventory
    from pytorch_arithmetic_normalization import clone_public_state
    require(isinstance(traced,torch.fx.GraphModule)and traced.meta.get("tensor_gather_source_policy")==POLICY,
            "Gather explicit normalization policy")
    source=inventory(traced);arrays=public_arrays(traced)
    expected,entries=normalized_inventory(source,arrays,resource_profile=resource_profile)
    result=clone_public_state(traced);by_name={n.name:n for n in result.graph.nodes}
    expected_by_name={n["name"]:n for n in expected}
    for entry in entries:
        original=by_name[entry["source_node"]];mapping={entry["input"]:by_name[entry["input"]]}
        planned=entry["plan"];offsets=planned["steps"][1]["attrs"]["indices"];generated=iter(entry["generated_nodes"])
        with result.graph.inserting_before(original):
            def make(kind,target,args,shape):
                name=next(generated);n=result.graph.create_node(kind,target,args,{},name=name)
                n.meta=copy.deepcopy(original.meta);n.meta["poseidon_static_tensor"]=copy.deepcopy(expected_by_name[name]["tensor"])
                by_name[name]=n;return n
            flat=make("call_function",torch.reshape,(mapping[entry["input"]],(math.prod(planned["source_shape"]),)),[math.prod(planned["source_shape"])])
            parts=[make("call_function",torch.narrow,(flat,0,offset,1),[1])for offset in offsets]
            joined=parts[0]if len(parts)==1 else make("call_function",torch.cat,(tuple(parts),0),[len(parts)])
            shaped=make("call_function",torch.reshape,(joined,tuple(planned["result_shape"])),planned["result_shape"])
            # The original name remains reserved until its node is erased.
            cloned=result.graph.call_method("clone",(shaped,));cloned.meta=copy.deepcopy(original.meta)
            clone_name=next(generated)
        original.replace_all_uses_with(cloned);result.graph.erase_node(original);cloned.name=clone_name
        by_name[clone_name]=cloned
    result.graph.lint();result.recompile()
    require(inventory(result)==expected and inventory(traced)==source
            and _public_state(public_arrays(result))==_public_state(arrays),"Gather pure FX normalization")
    record=dict(format=POLICY,input_signature=copy.deepcopy(inputs),resource_profile=resource_profile,
      source_inventory=source,normalized_inventory=expected,entries=entries,public_state=_public_state(arrays),
      implementation=implementation(),source_order_preserved=True,public_state_preserved=True)
    record["binding"]=digest(record);check(record,inputs,arrays,resource_profile=resource_profile)
    return result,record
