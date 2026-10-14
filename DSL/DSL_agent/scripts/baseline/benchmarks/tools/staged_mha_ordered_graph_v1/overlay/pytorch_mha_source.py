"""Registered native MHA nodes inside the unified ordered capture stream.

The frozen mathematical planner is reused; no model/factory dispatch exists.
Source records reconstruct every emitted MHA node/constant and original-state
slice before exposing approximation source locations to the host checker.
"""
import copy, hashlib, math
from benchmark_graph import NAME
from benchmark_graph import canonical, digest, require
from model_resources import resolve

POLICY="poseidon-mha-ordered-source-v1"
FORMAT="poseidon-pytorch-ordered-mha-source-v1"
IMPLEMENTATION=("pytorch_mha_source.py","pytorch_mha_math.py")
MASK_DIRECTION="native MHA bool True blocks; allow is logical-not blocked"
NAMES=("query","key","value","key_padding_mask","need_weights","attn_mask","average_attn_weights","is_causal")
ENTRY_FIELDS={"source_node","module_path","module_paths","source_request","source_request_sha256",
              "constant_bindings","node_bindings","value_bindings","state_bindings","mask_bindings","mask_direction","call_inputs","projections","outputs"}

def arguments(args,kwargs):
    require(len(args)<=len(NAMES) and not(set(kwargs)-set(NAMES)) and
            not(set(NAMES[:len(args)])&set(kwargs)),"Fixed native MHA call signature")
    result=dict(zip(NAMES[3:],(None,True,None,True,False)))
    result.update(zip(NAMES,args));result.update(kwargs)
    require(set(result)==set(NAMES) and all(type(result[n]) is bool for n in
            ("need_weights","average_attn_weights","is_causal")),"Static native MHA call options")
    return result

def metadata(module,args,kwargs,value,*,resource_profile):
    """Shape/type validation on metadata tensors only, never private values."""
    import torch
    require(type(module) is torch.nn.MultiheadAttention and module.training is False and
            not module._forward_hooks and not module._forward_pre_hooks,"Native eval MHA metadata only")
    options=arguments(args,kwargs);limits=resolve(resource_profile)
    xs=[value(options[name]) for name in NAMES[:3]]
    require(all(isinstance(x,torch.Tensor) and x.dtype is torch.float64 and x.ndim in (2,3)
                for x in xs),"Declared float64 MHA cipher metadata")
    q,k,v=[tuple(x.shape) for x in xs];e=module.embed_dim;h=module.num_heads
    require(type(e) is int and type(h) is int and e>0 and h>0 and e%h==0 and
            q[-1]==e and k[-1]==module.kdim and v[-1]==module.vdim and
            len(q)==len(k)==len(v),"Native MHA dimensions")
    unbatched=len(q)==2;batch_axis=0 if module.batch_first else 1;seq_axis=1 if module.batch_first else 0
    n=1 if unbatched else q[batch_axis];l=q[0] if unbatched else q[seq_axis]
    s=k[0] if unbatched else k[seq_axis]
    require((v[0]==s if unbatched else k[batch_axis]==v[batch_axis]==n and v[seq_axis]==s),
            "Native MHA batch/sequence binding")
    extra=int(module.bias_k is not None)+int(module.add_zero_attn);total=s+extra
    for name,allowed in (("attn_mask",((l,s),(n*h,l,s))),
                         ("key_padding_mask",((s,),) if unbatched else ((n,s),))):
        raw=options[name]
        if raw is not None:
            mask=value(raw)
            require(isinstance(mask,torch.Tensor) and mask.dtype in (torch.bool,torch.float64) and
                    tuple(mask.shape) in allowed,"Public MHA mask metadata")
    require(not options["is_causal"] or options["attn_mask"] is not None,"Explicit MHA causal mask required")
    dims=([l,total] if options["average_attn_weights"] else [h,l,total]) if unbatched else (
         [n,l,total] if options["average_attn_weights"] else [n,h,l,total])
    require(n*h*l*total<=limits.tensor_elements and all(
            math.prod(x)<=limits.tensor_elements for x in (q,k,v)),"MHA metadata resource budget")
    return (torch.empty(q,device="meta",dtype=torch.float64),
            torch.empty(dims,device="meta",dtype=torch.float64) if options["need_weights"] else None)

def is_mask_attribute(node,traced):
    """Only original tensor attributes consumed exclusively as native MHA masks."""
    import torch
    if node.op!="get_attr" or not node.users:return False
    for user in node.users:
        if user.op!="call_module" or type(traced.get_submodule(user.target)) is not torch.nn.MultiheadAttention:
            return False
        values=arguments(user.args,user.kwargs)
        if node not in (values["attn_mask"],values["key_padding_mask"]):return False
    return True

def _original_parameters(snapshot,target):
    return {target+"."+name:value for name,value in snapshot["parameters"].items() if value is not None}

def _projection_specs(snapshot,graph):
    c=snapshot["config"];e=c["embed_dim"]
    linears=[n for n in graph["nodes"] if n["op"]=="linear"]
    require(len(linears)==4,"Four native MHA projections")
    rows=[]
    for i,role in enumerate(("query","key","value","output")):
        weight="in_proj_weight" if i<3 and c["same_projection"] else (
               ("q_proj_weight","k_proj_weight","v_proj_weight")[i] if i<3 else "out_proj.weight")
        for operand,param in ((1,weight),(2,"in_proj_bias" if i<3 else "out_proj.bias")):
            if operand>=len(linears[i]["inputs"]):
                require(snapshot["parameters"][param] is None,"MHA projection bias presence");continue
            packed=i<3 and param in ("in_proj_weight","in_proj_bias")
            rows.append(dict(role=role,parameter=param,constant=linears[i]["inputs"][operand],
                slice=dict(axis=0,start=i*e,stop=(i+1)*e) if packed else None))
    return rows

def _module_paths(paths,target):
    from pytorch_approximation import _paths
    _paths(paths)
    require(target in paths and all(p==target or target.startswith(p+".") for p in paths),
            "Native MHA original module paths")

def expand(node,module,traced,*,cipher_ref,cipher_shape,public_values,
           constant,emit,graph,public_constant_views,source_origins,selection,approximations,resource_profile):
    """Called only by the registered MHA branch of the common capture worker."""
    import numpy as np, torch
    from pytorch_mha_math import snapshot_module,public_mask,plan
    from pytorch_approximation import Resolver,module_paths
    values=arguments(node.args,node.kwargs);paths=module_paths(node);_module_paths(paths,str(node.target))
    operands={}
    for role in NAMES[:3]:
        x=values[role]
        require(isinstance(x,torch.fx.Node),"Native MHA cipher operand")
        operands[role]=dict(kind="cipher",name=cipher_ref(x),shape=list(cipher_shape(x)),dtype="float64")
    masks={name:None for name in ("attn_mask","key_padding_mask")};options={k:values[k] for k in NAMES[3:]}
    for name in ("attn_mask","key_padding_mask"):
        x=values[name]
        if x is None:options[name]=None;continue
        require(isinstance(x,torch.fx.Node) and x.op=="get_attr" and x in public_values,
                "MHA masks must bind original public state")
        raw=public_values[x]
        require(raw.device.type=="cpu" and raw.dtype in (torch.bool,torch.float64) and
                torch.isfinite(raw).all().item(),"Finite original MHA mask state")
        masks[name]=str(x.target);options[name]=public_mask(raw)
    snapshot=snapshot_module(module)
    if selection is not None:
        temporary=Resolver(selection.configuration)
        profiles={name:temporary.select(node.name,name,name,paths) for name in ("exp","reciprocal")}
    else:
        require(type(approximations) is dict and all(n in approximations for n in ("exp","reciprocal")),
                "Explicit native MHA approximation")
        profiles={name:copy.deepcopy(approximations[name]) for name in ("exp","reciprocal")}
    mini=plan(snapshot,operands,options,profiles,resource_profile=resource_profile)
    request=mini["source_request"];g=mini["mathematical_graph"]
    state={}
    for path,array in _original_parameters(snapshot,str(node.target)).items():
        raw=traced
        for part in path.split("."):raw=getattr(raw,part)
        require(raw.device.type=="cpu" and raw.dtype is torch.float64,"Normalized original MHA parameter")
        require(np.array_equal(raw.detach().numpy(),np.asarray(array,dtype="<f8")),"MHA original parameter snapshot")
        state[path]=constant(raw,path)
    for name,path in masks.items():
        if path is None:continue
        raw=public_values[values[name]]
        state[path]=constant(raw.double(),path)
    projections=_projection_specs(snapshot,g);projected={p["constant"]:p for p in projections}
    remap={x["name"]:x["name"] for x in g["inputs"]}
    constant_bindings={}
    for name,array in g["constants"].items():
        if name in projected:
            p=projected[name];path=str(node.target)+"."+p["parameter"]
            raw=np.asarray(snapshot["parameters"][p["parameter"]],dtype="<f8")
            indexes=np.arange(raw.size,dtype=np.int64).reshape(raw.shape)
            expected=raw
            if p["slice"] is not None:
                s=p["slice"];expected=raw[s["start"]:s["stop"]];indexes=indexes[s["start"]:s["stop"]]
            require(np.array_equal(np.asarray(array,dtype="<f8"),expected),"MHA projection slice bytes")
            if p["slice"] is None:
                actual=state[path]
            else:
                actual=constant(torch.tensor(array,dtype=torch.float64))
                public_constant_views[actual]=dict(source=state[path],source_shape=list(raw.shape),
                    result_shape=list(expected.shape),indices=indexes.tolist())
        else:
            actual=constant(torch.tensor(array,dtype=torch.float64))
        remap[name]=actual;constant_bindings[name]=actual
    call_inputs={name:dict(fx_node=values[name].name,value=operands[name]["name"]) for name in NAMES[:3]}
    annotation=dict(policy=POLICY,source_node=node.name,module_path=str(node.target),module_paths=paths,
                    call_inputs=call_inputs,mask_bindings=masks)
    node_bindings={};value_bindings={}
    for i,item in enumerate(g["nodes"]):
        if item["op"] in ("exp","reciprocal") and selection is not None:
            chosen=selection.select(node.name,item["op"],item["op"],paths)
            require(chosen==item["attrs"]["approximation"],"MHA emitted approximation binding")
        emit(node,item["op"],[remap[x] for x in item["inputs"]],copy.deepcopy(item["attrs"]),
             suffix="_mha_"+str(i))
        actual=graph["nodes"][-1]
        source_origins[actual["id"]]["mha"]=copy.deepcopy(annotation)
        node_bindings[item["id"]]=actual["id"]
        remap[item["outputs"][0]]=actual["outputs"][0]
        value_bindings[item["outputs"][0]]=actual["outputs"][0]
    outs={o["name"]:remap[o["value"]] for o in g["outputs"]}
    ports=(outs["attn_output"],outs.get("attn_weights"))
    entry=dict(source_node=node.name,module_path=str(node.target),module_paths=paths,
        source_request=request,source_request_sha256=mini["source_request_sha256"],
        constant_bindings=constant_bindings,node_bindings=node_bindings,value_bindings=value_bindings,
        state_bindings=state,mask_bindings=masks,mask_direction=MASK_DIRECTION,
        call_inputs=call_inputs,projections=projections,outputs=outs)
    return ports,entry

def _mask_states(options,bindings):
    require(type(bindings) is dict and set(bindings)=={"attn_mask","key_padding_mask"},
            "MHA original mask source inventory")
    states={}
    for name,path in bindings.items():
        mask=options[name]
        require((mask is None)==(path is None),"MHA mask source presence")
        if mask is None:continue
        require(type(path) is str and 0<len(path)<=256 and
                type(mask) is dict and set(mask)=={"dtype","values","negative_infinity"} and
                mask["dtype"] in ("bool","float64") and mask["negative_infinity"] is None,
                "Finite typed original MHA mask source")
        require(path not in states or canonical(states[path])==canonical(mask),
                "Shared original MHA mask options differ")
        states[path]=copy.deepcopy(mask)
    return states

def _check_entry(entry,emissions,*,resource_profile,source_origins,public_constant_views,
                 records=None,expected_weights=None):
    import numpy as np
    from pytorch_mha_math import plan
    require(type(entry) is dict and set(entry)==ENTRY_FIELDS,"MHA source entry fields")
    require(type(entry["source_node"]) is str and NAME.fullmatch(entry["source_node"]) and
            type(entry["module_path"]) is str,"Native MHA FX/module identity")
    _module_paths(entry["module_paths"],entry["module_path"])
    require(entry["mask_direction"]==MASK_DIRECTION and type(entry["mask_bindings"]) is dict and
            set(entry["mask_bindings"])=={"attn_mask","key_padding_mask"} and
            type(entry["call_inputs"]) is dict and set(entry["call_inputs"])==set(NAMES[:3]),
            "Native MHA call/mask inventory")
    request=entry["source_request"]
    require(type(request) is dict and set(request)=={"module","operands","options","approximation","resource_profile"}
            and request["resource_profile"]==resource_profile and digest(request)==entry["source_request_sha256"],
            "MHA original call/request binding")
    mini=plan(**request);g=mini["mathematical_graph"]
    require(entry["projections"]==_projection_specs(request["module"],g),"MHA original projection records")
    nodes={n["id"]:n for n in emissions["nodes"]}
    require(len(nodes)==len(emissions["nodes"]) and
            set(entry["constant_bindings"])==set(g["constants"]) and
            set(entry["node_bindings"])=={n["id"] for n in g["nodes"]} and
            set(entry["value_bindings"])=={v for n in g["nodes"] for v in n["outputs"]},
            "Complete MHA emission inventory")
    remap={s["name"]:s["name"] for s in g["inputs"]}
    for name,array in g["constants"].items():
        actual=entry["constant_bindings"][name]
        require(actual in emissions["constants"] and
                canonical(emissions["constants"][actual])==canonical(array),"MHA emitted constant bytes")
        remap[name]=actual
    annotation=dict(policy=POLICY,source_node=entry["source_node"],module_path=entry["module_path"],
                    module_paths=entry["module_paths"],call_inputs=entry["call_inputs"],
                    mask_bindings=entry["mask_bindings"])
    for role,call in entry["call_inputs"].items():
        require(type(call) is dict and set(call)=={"fx_node","value"} and
                type(call["fx_node"]) is str and NAME.fullmatch(call["fx_node"]) and
                call["value"]==request["operands"][role]["name"],"Native MHA input call binding")
    locations={};used=[]
    for item in g["nodes"]:
        actual_id=entry["node_bindings"][item["id"]];require(actual_id in nodes,"MHA emission node missing")
        actual=nodes[actual_id]
        require(actual_id in source_origins and source_origins[actual_id].get("fx_node")==entry["source_node"] and
                source_origins[actual_id].get("fx_kind")=="call_module" and
                source_origins[actual_id].get("operation")==actual["op"] and
                source_origins[actual_id].get("mha")==annotation,"Native MHA call/emission origin binding")
        out=entry["value_bindings"][item["outputs"][0]]
        require(actual==dict(id=actual_id,op=item["op"],inputs=[remap[x] for x in item["inputs"]],
                            attrs=item["attrs"],outputs=[out]),"MHA emission differs from mathematical plan")
        remap[item["outputs"][0]]=out;used.append(actual_id)
        if item["op"] in ("exp","reciprocal"):
            locations[actual_id]=dict(source_node=entry["source_node"],module_paths=copy.deepcopy(entry["module_paths"]))
    require(len(set(used))==len(used),"MHA node remapping collision")
    require(entry["outputs"]=={o["name"]:remap[o["value"]] for o in g["outputs"]},"MHA tuple output binding")
    original=_original_parameters(request["module"],entry["module_path"])
    mask_states=_mask_states(request["options"],entry["mask_bindings"])
    mask_paths=set(mask_states)
    require(not mask_paths.intersection(original) and
            set(entry["state_bindings"])==set(original)|mask_paths,"Exact original MHA state inventory")
    expected_state=dict(original)
    expected_state.update({path:mask["values"] for path,mask in mask_states.items()})
    for path,array in expected_state.items():
        constant_name=entry["state_bindings"][path]
        require(constant_name in emissions["constants"],"MHA original state constant missing")
        raw=np.asarray(emissions["constants"][constant_name],dtype="<f8")
        expected=np.asarray(array,dtype="<f8")
        require(raw.shape==expected.shape and np.array_equal(raw,expected) and np.isfinite(raw).all(),
                "MHA original state differs from native request")
        if records is not None:
            require(path in records and list(raw.shape)==records[path]["shape"] and
                    hashlib.sha256(raw.tobytes(order="C")).hexdigest()==
                    expected_weights["normalized_float64_sha256"][path],"MHA original state differs from loaded weights")
            for name,mask_path in entry["mask_bindings"].items():
                if path==mask_path:
                    dtype="|b1" if request["options"][name]["dtype"]=="bool" else None
                    require(records[path]["dtype"]==dtype if dtype is not None else
                            records[path]["dtype"] in ("<f4","<f8"),"MHA original mask dtype")
    for projection in entry["projections"]:
        path=entry["module_path"]+"."+projection["parameter"]
        actual=entry["constant_bindings"][projection["constant"]]
        original_name=entry["state_bindings"][path]
        raw=np.asarray(original[path],dtype="<f8")
        if projection["slice"] is None:
            require(actual==original_name,"MHA direct projection source")
        else:
            selected=projection["slice"];indices=np.arange(raw.size,dtype=np.int64).reshape(raw.shape)
            indices=indices[selected["start"]:selected["stop"]]
            expected=dict(source=original_name,source_shape=list(raw.shape),
                          result_shape=list(indices.shape),indices=indices.tolist())
            require(public_constant_views.get(actual)==expected,"MHA packed projection slice provenance")
    return locations,used

def _locations(graph,emissions,locations,normalized_origins):
    if normalized_origins is None:
        require(graph==emissions,"MHA normalized origins required")
        return locations
    require(type(normalized_origins) is dict and set(normalized_origins)=={n["id"] for n in graph["nodes"]},
            "Complete MHA normalized provenance")
    result={};seen=[]
    raw={n["id"]:n for n in emissions["nodes"]}
    for node in graph["nodes"]:
        origin=normalized_origins[node["id"]]
        require(type(origin) is dict,"MHA normalized origin fields")
        source=origin.get("source_id")
        if source in locations and node["op"] in ("exp","reciprocal"):
            require(node["op"]==raw[source]["op"] and node["attrs"]==raw[source]["attrs"],
                    "MHA normalized nonlinear component differs")
            result[node["id"]]=copy.deepcopy(locations[source]);seen.append(source)
    require(set(seen)==set(locations) and len(seen)==len(locations),"MHA nonlinear normalization inventory")
    return result

def record(graph,emissions,entries,*,normalized_origins=None,source_origins,
           public_constant_views,resource_profile):
    require(type(entries) is list and len(entries)<=resolve(resource_profile).original_nodes and
            type(source_origins) is dict and type(public_constant_views) is dict and
            set(source_origins)=={n["id"] for n in emissions["nodes"]},"MHA source inventory budget/provenance")
    seen=set();source=set();locations={}
    for entry in entries:
        selected,used=_check_entry(entry,emissions,resource_profile=resource_profile,
                        source_origins=source_origins,public_constant_views=public_constant_views)
        require(not seen.intersection(used) and entry["source_node"] not in source,"Unique MHA source segments")
        seen.update(used);source.add(entry["source_node"]);locations.update(selected)
    require(seen=={identifier for identifier,origin in source_origins.items() if "mha" in origin},
            "Complete native MHA source segments")
    _locations(graph,emissions,locations,normalized_origins)
    value=dict(format=FORMAT,policy=POLICY,graph_sha256=digest(graph),emission_graph_sha256=digest(emissions),
               entries=copy.deepcopy(entries))
    value["binding"]=digest(value);return value

def check_record(value,graph,*,emissions,normalized_origins=None,source_origins,
                 public_constant_views,state_constant_sources=None,declared_records=None,
                 expected_weight_binding=None,resource_profile):
    require((state_constant_sources is None)==(declared_records is None)==(expected_weight_binding is None),
            "Complete loaded MHA state binding required")
    if declared_records is not None:
        from pytorch_weight_package import check_binding
        check_binding(expected_weight_binding,declared_records)
    require(type(value) is dict and set(value)=={"format","policy","graph_sha256","emission_graph_sha256","entries","binding"}
            and value["format"]==FORMAT and value["policy"]==POLICY and value["binding"]==
            digest({k:v for k,v in value.items() if k!="binding"}) and
            value["graph_sha256"]==digest(graph) and value["emission_graph_sha256"]==digest(emissions),
            "MHA source record identity")
    require(value==record(graph,emissions,value["entries"],normalized_origins=normalized_origins,
                          source_origins=source_origins,public_constant_views=public_constant_views,
                          resource_profile=resource_profile),"MHA source record reconstruction")
    result={}
    for entry in value["entries"]:
        locations,_=_check_entry(entry,emissions,resource_profile=resource_profile,
                                source_origins=source_origins,public_constant_views=public_constant_views,
                                records=declared_records,expected_weights=expected_weight_binding)
        if state_constant_sources is not None:
            for path,name in entry["state_bindings"].items():
                require(state_constant_sources.get(name)==path,"MHA original emission state source")
        require(not set(result).intersection(locations),"Unique MHA approximation source locations")
        result.update(locations)
    _locations(graph,emissions,result,normalized_origins)
    return result
