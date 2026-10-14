"""Deterministic FX arithmetic normalization for an explicitly trusted capture worker.

This operates on an already captured FX GraphModule. It does not import model
source, load weights, execute a candidate, or bypass the existing capture rules.
Public V1-V7 intake does not call it; integration requires a new capture policy.
"""
import copy
import hashlib
import math
import operator
from benchmark_graph import digest,require
from model_resources import resolve

POLICY="poseidon-fx-public-arithmetic-v1"

def normalize(traced,*,resource_profile="legacy-v1",preserve_unused_state=False):
    import torch
    require(isinstance(traced,torch.fx.GraphModule),"An explicitly trusted FX GraphModule is required")
    limits=resolve(resource_profile)
    nodes=list(traced.graph.nodes)
    require(len(nodes)<=limits.expanded_nodes+limits.constants+limits.inputs+limits.outputs,
            "FX arithmetic graph budget")
    source_graph=str(traced.graph)
    before={name:value.detach().clone() for name,value in traced.state_dict().items()}
    public={}
    for node in nodes:
        if node.op!="get_attr":continue
        value=traced
        for segment in str(node.target).split("."):value=getattr(value,segment)
        if not isinstance(value,torch.Tensor):continue
        array=value.detach().cpu().double().numpy()
        public[node.name]=dict(target=str(node.target),shape=list(array.shape),
                              float64_sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())
    require(type(preserve_unused_state) is bool,"Public state copy policy")
    result=clone_public_state(traced) if preserve_unused_state else copy.deepcopy(traced)
    roles={};entries=[]
    operators={operator.add:"add",operator.sub:"subtract",operator.mul:"multiply",
               torch.add:"add",torch.sub:"subtract",torch.subtract:"subtract",
               torch.mul:"multiply",torch.multiply:"multiply"}
    methods={"add":"add","sub":"subtract","subtract":"subtract","mul":"multiply","multiply":"multiply"}
    targets={"add":operator.add,"subtract":operator.sub,"multiply":operator.mul}
    def dependencies(value):
        if isinstance(value,torch.fx.Node):return [value]
        if isinstance(value,(tuple,list)):return [n for v in value for n in dependencies(v)]
        if isinstance(value,dict):return [n for v in value.values() for n in dependencies(v)]
        return []
    def encoded(value):
        if isinstance(value,torch.fx.Node):return dict(kind="fx_value",name=value.name)
        require(type(value) in (int,float) and math.isfinite(value),"Static finite arithmetic scalar")
        return dict(kind="public_scalar",value=value)
    def operands(node,op):
        if node.op=="call_function" and node.target in (operator.add,operator.sub,operator.mul):
            require(len(node.args)==2 and not node.kwargs,"Binary arity/kwargs")
            return list(node.args)
        allowed={"other","alpha","out"} if node.op=="call_method" else {"input","other","alpha","out"}
        if op=="multiply":allowed-={"alpha"}
        require(len(node.args)<=2 and not(set(node.kwargs)-allowed),"Arithmetic argument fields")
        names=("self","other") if node.op=="call_method" else ("input","other")
        values={}
        for name,value in zip(names,node.args):values[name]=value
        for name in names:
            if name in node.kwargs:
                require(name not in values,"Duplicate arithmetic argument")
                values[name]=node.kwargs[name]
        require(set(values)==set(names),"Arithmetic input/other required")
        alpha=node.kwargs.get("alpha",1)
        require(type(alpha) in (int,float) and math.isfinite(alpha) and alpha==1,
                "Arithmetic alpha=1 required")
        require(node.kwargs.get("out") is None,"Arithmetic out mutation is unsupported")
        return [values[name] for name in names]
    for node in list(result.graph.nodes):
        if node.op=="placeholder":roles[node]="cipher";continue
        if node.op=="get_attr":roles[node]="plain";continue
        refs=dependencies((node.args,node.kwargs))
        role="cipher" if any(roles.get(ref)=="cipher" for ref in refs) else "plain"
        if node.op=="call_method" and node.target in ("add_","sub_","subtract_","mul_","multiply_"):
            raise ValueError("In-place arithmetic is unsupported")
        op=operators.get(node.target) if node.op=="call_function" else methods.get(node.target) if node.op=="call_method" else None
        if op is None:roles[node]=role;continue
        args=operands(node,op)
        encoded_args=[encoded(a) for a in args]
        kinds=[roles[a] if isinstance(a,torch.fx.Node) else "plain" for a in args]
        require("cipher" in kinds,"Arithmetic requires an encrypted operand")
        entry=dict(fx_node=node.name,source_fx_node=node.name,source_operands=copy.deepcopy(encoded_args),source_call_kind=node.op,operator=op,
                   operands=encoded_args,input_kinds=kinds,action="canonical_call",added_node=None)
        if kinds[0]=="plain":
            if op=="subtract":
                with result.graph.inserting_before(node):
                    negative=result.graph.call_function(operator.neg,args=(args[1],),kwargs={})
                roles[negative]="cipher";args=[negative,args[0]];target=operator.add
                entry.update(action="negate_cipher_then_add_public",added_node=negative.name)
            else:
                args.reverse();target=targets[op];entry["action"]="swap_commutative_operands"
        else:target=targets[op]
        node.op="call_function";node.target=target;node.args=tuple(args);node.kwargs={}
        roles[node]="cipher";entry["normalized_operands"]=[encoded(a) for a in args]
        entries.append(entry)
    result.graph.lint();result.recompile()
    require(before.keys()==traced.state_dict().keys()==result.state_dict().keys() and
            all(torch.equal(value,traced.state_dict()[name]) and torch.equal(value,result.state_dict()[name])
                for name,value in before.items()),"Arithmetic normalization changed public state")
    require(str(traced.graph)==source_graph,"Arithmetic normalization changed source graph")
    record=dict(format=POLICY,source_fx_sha256=hashlib.sha256(source_graph.encode()).hexdigest(),
                normalized_fx_sha256=hashlib.sha256(str(result.graph).encode()).hexdigest(),
                source_fx_text=source_graph,normalized_fx_text=str(result.graph),
                fx_inputs=[node.name for node in nodes if node.op=="placeholder"],
                public_attributes=public,
                entries=entries,resource_profile=resource_profile,
                capture_name_map={n.name:n.name for n in result.graph.nodes},
                source_graph_preserved=True,
                public_state_preserved=True,requires_trusted_capture_worker=True,
                existing_capture_rules_retained=True)
    record["binding_sha256"]=digest(record)
    return result,record

def bind_record(record,graph,*,captured_fx=None):
    """Bind the trusted FX change record to the actual captured high graph."""
    result=copy.deepcopy(record)
    result.pop("binding_sha256",None)
    result["captured_graph_sha256"]=digest(graph)
    aliases={}
    if captured_fx is not None:
        import torch
        require(isinstance(captured_fx,torch.fx.GraphModule),"Captured FX graph required")
        for node in captured_fx.graph.nodes:
            if (node.op=="call_function" and node.target is operator.getitem and
                len(node.args)==2 and not node.kwargs and isinstance(node.args[0],torch.fx.Node) and
                type(node.args[1]) is int):
                source=node.args[0]
                if ((source.op=="call_function" and source.target is torch.split) or
                    (source.op=="call_method" and source.target=="split")):
                    aliases[node.name]=dict(source=source.name,port=node.args[1])
    result["split_aliases"]=aliases
    result["binding_sha256"]=digest(result)
    check_record(result,graph)
    return result

def check_record(record,graph):
    import numpy as np
    require(type(record) is dict and record.get("format")==POLICY and
            digest({k:v for k,v in record.items() if k!="binding_sha256"})==record.get("binding_sha256"),
            "Arithmetic normalization record binding")
    require(record.get("captured_graph_sha256")==digest(graph) and
            record.get("source_graph_preserved") is True and record.get("public_state_preserved") is True and
            record.get("requires_trusted_capture_worker") is True and
            record.get("existing_capture_rules_retained") is True,"Arithmetic capture scope/graph binding")
    for text_key,sha_key in (("source_fx_text","source_fx_sha256"),("normalized_fx_text","normalized_fx_sha256")):
        text=record.get(text_key)
        require(type(text) is str and len(text.encode())<=resolve(record["resource_profile"]).json_bytes and
                hashlib.sha256(text.encode()).hexdigest()==record.get(sha_key),"Arithmetic FX source binding")
    inputs=record.get("fx_inputs");name_map=record.get("capture_name_map")
    limits=resolve(record["resource_profile"])
    require(type(inputs) is list and len(inputs)==len(graph["inputs"]) and
            all(type(name) is str for name in inputs) and len(set(inputs))==len(inputs),
            "Arithmetic input port binding")
    require(type(name_map) is dict and len(name_map)<=limits.expanded_nodes+limits.constants+limits.inputs+limits.outputs and
            all(type(a) is str and type(b) is str for a,b in name_map.items()) and
            len(set(name_map.values()))==len(name_map),"Arithmetic source name mapping")
    aliases=record.get("split_aliases")
    require(type(aliases) is dict and len(aliases)<=limits.expanded_nodes and
            all(type(name) is str and type(value) is dict and set(value)=={"source","port"} and
                type(value["source"]) is str and type(value["port"]) is int
                for name,value in aliases.items()),"Arithmetic split alias binding")
    entries=record.get("entries")
    require(type(entries) is list and len(entries)<=resolve(record["resource_profile"]).expanded_nodes,
            "Arithmetic normalization entries")
    kinds={s["name"]:"cipher" for s in graph["inputs"]}
    kinds.update({name:"plain" for name in graph["constants"]})
    nodes={node["id"]:node for node in graph["nodes"]}
    for node in graph["nodes"]:
        for name in node["outputs"]:kinds[name]="cipher"
    input_refs=dict(zip(inputs,(s["name"] for s in graph["inputs"])))
    def cipher_ref(name):
        if name in input_refs:return input_refs[name]
        producer=nodes.get("fx_"+name)
        if producer is not None and len(producer["outputs"])==1:return producer["outputs"][0]
        alias=aliases.get(name)
        require(alias is not None,"Arithmetic cipher operand has no captured producer")
        source=nodes.get("fx_"+alias["source"])
        require(source is not None and source["op"]=="split","Arithmetic split operand producer")
        port=alias["port"]
        require(-len(source["outputs"])<=port<len(source["outputs"]),"Arithmetic split operand port")
        return source["outputs"][port]
    def source_operand(value):
        require(type(value) is dict,"Arithmetic source operand")
        if value.get("kind")=="fx_value":
            require(set(value)=={"kind","name"} and type(value["name"]) is str and value["name"] in name_map,"Arithmetic source operand name")
            return dict(kind="fx_value",name=name_map[value["name"]])
        require(set(value)=={"kind","value"} and value["kind"]=="public_scalar" and
                type(value["value"]) in (int,float) and math.isfinite(value["value"]),"Arithmetic source scalar")
        return value
    for entry in entries:
        require(type(entry) is dict and set(entry)=={"fx_node","source_call_kind","operator","operands",
                "input_kinds","action","added_node","normalized_operands","source_fx_node","source_operands"},"Arithmetic entry fields")
        require(type(entry["source_fx_node"]) is str and entry["source_fx_node"] in name_map and
                entry["fx_node"]==name_map[entry["source_fx_node"]] and
                type(entry["source_operands"]) is list and
                entry["operands"]==[source_operand(v) for v in entry["source_operands"]],
                "Arithmetic source operand mapping")
        op=entry["operator"];roles=entry["input_kinds"]
        require(op in ("add","subtract","multiply") and roles in
                (["cipher","cipher"],["cipher","plain"],["plain","cipher"]) and
                entry["source_call_kind"] in ("call_function","call_method"),"Arithmetic entry operation/roles")
        expected="negate_cipher_then_add_public" if roles[0]=="plain" and op=="subtract" else \
                 "swap_commutative_operands" if roles[0]=="plain" else "canonical_call"
        require(entry["action"]==expected,"Arithmetic entry action")
        operands=entry["operands"];normalized=entry["normalized_operands"]
        require(type(operands) is list and len(operands)==2 and type(normalized) is list and len(normalized)==2,
                "Arithmetic entry arity")
        if expected=="swap_commutative_operands":
            require(normalized==operands[::-1] and entry["added_node"] is None,"Arithmetic swap")
        elif expected=="negate_cipher_then_add_public":
            require(type(entry["added_node"]) is str and
                    normalized==[dict(kind="fx_value",name=entry["added_node"]),operands[0]],"Arithmetic subtraction")
        else:require(normalized==operands and entry["added_node"] is None,"Canonical arithmetic call")
        captured=nodes.get("fx_"+entry["fx_node"])
        require(captured is not None and captured["op"]==("add" if expected=="negate_cipher_then_add_public" else op)
                and captured["attrs"]=={} and len(captured["inputs"])==2,"Captured arithmetic operation")
        actual_roles=[kinds.get(name) for name in captured["inputs"]]
        require(actual_roles==(["cipher","plain"] if roles[0]=="plain" else roles),
                "Captured arithmetic operand roles")
        if expected=="negate_cipher_then_add_public":
            negative=nodes.get("fx_"+entry["added_node"])
            require(negative is not None and negative["op"]=="negate" and negative["attrs"]=={} and
                    negative["outputs"]==[captured["inputs"][0]] and
                    negative["inputs"]==[cipher_ref(operands[1]["name"])],
                    "Captured arithmetic negation")
        for index,kind in enumerate(actual_roles):
            if kind=="cipher":
                value=normalized[index]
                require(value.get("kind")=="fx_value" and
                        captured["inputs"][index]==cipher_ref(value["name"]),
                        "Captured arithmetic cipher operand binding")
                continue
            require(kind=="plain","Captured arithmetic operand kind")
            encoded=normalized[index];actual=np.asarray(graph["constants"][captured["inputs"][index]],dtype="<f8")
            if encoded.get("kind")=="public_scalar":
                require(actual.ndim==0 and float(actual)==encoded["value"],"Captured public scalar binding")
            else:
                descriptor=record["public_attributes"].get(encoded.get("name"))
                require(descriptor is not None and list(actual.shape)==descriptor["shape"] and
                        hashlib.sha256(actual.tobytes(order="C")).hexdigest()==descriptor["float64_sha256"],
                        "Captured public tensor binding")
    return record

def rename_record(record,normalized,canonical):
    """Record stable source-to-capture names after the existing static retracer."""
    import torch
    require(isinstance(normalized,torch.fx.GraphModule) and isinstance(canonical,torch.fx.GraphModule),
            "Arithmetic retrace requires FX GraphModules")
    old=list(normalized.graph.nodes);new=list(canonical.graph.nodes)
    require(len(old)==len(new),"Arithmetic retrace graph arity changed")
    # Generated FX code may read a public attribute at its first use rather than
    # its earlier position in the source graph. Computations keep their order.
    # Match attribute targets separately; do not accept reordered computations.
    old_compute=[n for n in old if n.op!="get_attr"]
    new_compute=[n for n in new if n.op!="get_attr"]
    require(len(old_compute)==len(new_compute),"Arithmetic retrace compute arity changed")
    mapping=dict(zip(old_compute,new_compute))
    def attribute_groups(nodes):
        groups={}
        for node in nodes:
            if node.op=="get_attr":groups.setdefault(node.target,[]).append(node)
        return groups
    a_attrs,b_attrs=attribute_groups(old),attribute_groups(new)
    require(a_attrs.keys()==b_attrs.keys(),"Arithmetic retrace public attributes changed")
    def attribute(module,target):
        value=module
        for segment in str(target).split("."):value=getattr(value,segment)
        return value
    for target,previous in a_attrs.items():
        current=b_attrs[target]
        require(len(previous)==len(current),"Arithmetic retrace attribute arity changed")
        a_value,b_value=attribute(normalized,target),attribute(canonical,target)
        if isinstance(a_value,torch.Tensor):
            require(isinstance(b_value,torch.Tensor) and a_value.dtype==b_value.dtype and
                    a_value.device==b_value.device and a_value.shape==b_value.shape and
                    torch.equal(a_value,b_value),"Arithmetic retrace public tensor changed")
        else:
            require(type(a_value) is type(b_value) and a_value==b_value,
                    "Arithmetic retrace public attribute changed")
        mapping.update(zip(previous,current))
    def mapped(value):
        if isinstance(value,torch.fx.Node):return mapping[value]
        if isinstance(value,tuple):return tuple(mapped(v) for v in value)
        if isinstance(value,list):return [mapped(v) for v in value]
        if isinstance(value,dict):return {k:mapped(v) for k,v in value.items()}
        return value
    for a,b in mapping.items():
        require(a.op==b.op and a.target==b.target and mapped(a.args)==b.args and
                mapped(a.kwargs)==b.kwargs,"Arithmetic retrace semantics changed")
    names={a.name:b.name for a,b in mapping.items()}
    def renamed(value):
        if value["kind"]=="fx_value":return dict(kind="fx_value",name=names[value["name"]])
        return copy.deepcopy(value)
    result=copy.deepcopy(record);result.pop("binding_sha256",None)
    for entry in result["entries"]:
        entry["fx_node"]=names[entry["fx_node"]]
        entry["operands"]=[renamed(v) for v in entry["operands"]]
        entry["normalized_operands"]=[renamed(v) for v in entry["normalized_operands"]]
        if entry["added_node"] is not None:entry["added_node"]=names[entry["added_node"]]
    result["capture_name_map"]={a:names[b] for a,b in result["capture_name_map"].items()}
    result["public_attributes"]={names[k]:v for k,v in result["public_attributes"].items()}
    result["fx_inputs"]=[names[k] for k in result["fx_inputs"]]
    result["normalized_fx_text"]=str(canonical.graph)
    result["normalized_fx_sha256"]=hashlib.sha256(str(canonical.graph).encode()).hexdigest()
    result["binding_sha256"]=digest(result)
    return result


COMMON_POLICY="poseidon-fx-common-calls-v1"


def clone_public_state(module):
    """FX deepcopy may discard unused buffers; retain them for the V11 state audit."""
    import torch
    result=copy.deepcopy(module)
    for table,is_parameter in ((module.named_parameters(remove_duplicate=False),True),
                               (module.named_buffers(remove_duplicate=False),False)):
        for path,value in table:
            owner=result;source_owner=module
            parts=path.split(".")
            for part in parts[:-1]:
                source_owner=source_owner.get_submodule(part)
                if part not in owner._modules:
                    owner.add_module(part,copy.deepcopy(source_owner))
                owner=owner.get_submodule(part)
            leaf=parts[-1]
            registered=owner._parameters if is_parameter else owner._buffers
            if leaf in registered:continue
            require(not hasattr(owner,leaf),"Public state copy collision")
            if is_parameter:owner.register_parameter(leaf,copy.deepcopy(value))
            else:owner.register_buffer(leaf,copy.deepcopy(value),
                                      persistent=leaf not in source_owner._non_persistent_buffers_set)
    return result

def common_arguments(args,kwargs,names,defaults):
    require(len(args)<=len(names) and not(set(kwargs)-set(names)),"Registered common call arguments")
    values=dict(zip(names,args))
    for name,value in kwargs.items():
        require(name not in values,"Duplicate common call argument")
        values[name]=value
    for name,value in defaults.items():values.setdefault(name,value)
    require(set(values)==set(names),"Common call required arguments")
    return values

def common_scalar(value):
    require(type(value) in (int,float) and math.isfinite(value),"Finite public scalar required")
    return value

def sdpa_shape(query,key,value,*,resource_profile):
    import torch
    limits=resolve(resource_profile)
    shapes=(query,key,value)
    require(all(type(s) in (list,tuple) and 2<=len(s)<=limits.rank and
                all(type(d) is int and 0<d<=limits.axis_size for d in s) for s in shapes),
            "Static SDPA tensor ranks")
    require(query[-1]==key[-1] and key[-2]==value[-2],"SDPA contracting dimensions")
    batch=list(torch.broadcast_shapes(query[:-2],key[:-2],value[:-2]))
    scores=batch+[query[-2],key[-2]];output=batch+[query[-2],value[-1]]
    require(math.prod(scores)<=limits.tensor_elements and math.prod(output)<=limits.tensor_elements,
            "SDPA tensor budget")
    return scores,output

def normalize_common(traced,inputs,*,resource_profile="general-tensors-v1"):
    """Canonicalize registered calls on an already isolated, trusted FX graph."""
    import torch
    from torch.nn import functional as F
    from pytorch_static_trace import COMMON_POLICY as SHAPE_POLICY
    require(isinstance(traced,torch.fx.GraphModule),"Explicit FX capture required")
    limits=resolve(resource_profile)
    source=str(traced.graph)
    before={k:v.detach().clone() for k,v in traced.state_dict().items()}
    result=clone_public_state(traced);entries=[];generated=[]
    ports=[n for n in result.graph.nodes if n.op=="placeholder"]
    require(len(ports)==len(inputs),"Common normalization input binding")
    signatures=dict(zip(ports,(item["shape"] for item in inputs)))
    def shape(node):
        require(isinstance(node,torch.fx.Node),"Static common tensor required")
        if node in signatures:return list(signatures[node])
        metadata=node.meta.get("poseidon_static_tensor")
        require(type(metadata) is dict and set(metadata)=={"policy","shape","dtype"} and
                metadata["policy"]==SHAPE_POLICY,"Registered static tensor metadata required")
        dims=metadata["shape"]
        require(type(dims) is list and len(dims)<=limits.rank and
                all(type(d) is int and 0<d<=limits.axis_size for d in dims) and
                math.prod(dims)<=limits.tensor_elements,"Common tensor shape capacity")
        return list(dims)
    def attribute(target):
        value=result
        for part in str(target).split("."):
            require(type(part) is str and part.isidentifier(),"Public attribute path")
            value=getattr(value,part)
        return value
    def public_mask(node,score_shape):
        require(isinstance(node,torch.fx.Node) and node.op=="get_attr","SDPA mask must be public state")
        mask=attribute(node.target)
        require(isinstance(mask,torch.Tensor) and mask.dtype in (torch.bool,torch.float32,torch.float64),
                "Public SDPA mask dtype")
        dims=list(mask.shape)
        require(list(torch.broadcast_shapes(dims,score_shape))==score_shape,"SDPA mask broadcast")
        if mask.dtype is not torch.bool:
            require(bool(torch.isfinite(mask).all()),"Finite additive SDPA mask required")
        return mask
    def generated_constant(node,tensor,kind):
        value=tensor.detach().cpu().double().contiguous()
        require(bool(torch.isfinite(value).all()),"Finite generated common constant required")
        require(value.numel()<=limits.constant_elements,"Generated common constant capacity")
        name="_poseidon_common_"+node.name+"_"+kind
        require(not hasattr(result,name),"Generated common constant name collision")
        result.register_buffer(name,value)
        with result.graph.inserting_before(node):
            ref=result.graph.get_attr(name)
        generated.append(dict(target=name,shape=list(value.shape),dtype="float64",kind=kind,
                              float64_sha256=hashlib.sha256(value.numpy().astype("<f8",copy=False).tobytes()).hexdigest()))
        return ref
    dropout_functions={F.dropout,F.dropout1d,F.dropout2d,F.dropout3d,F.alpha_dropout,F.feature_alpha_dropout}
    dropout_modules=(torch.nn.Dropout,torch.nn.Dropout1d,torch.nn.Dropout2d,torch.nn.Dropout3d,
                     torch.nn.AlphaDropout,torch.nn.FeatureAlphaDropout)
    for node in list(result.graph.nodes):
        kind=node.op;target=node.target
        arithmetic=(kind=="call_function" and target in (torch.add,torch.sub,torch.subtract)) or (
            kind=="call_method" and target in ("add","sub","subtract"))
        if arithmetic:
            names=("self","other","alpha","out") if kind=="call_method" else ("input","other","alpha","out")
            values=common_arguments(node.args,node.kwargs,names,{"alpha":1,"out":None})
            require(len(node.args)<=2,"Arithmetic alpha is keyword-only")
            alpha=common_scalar(values["alpha"])
            require(values["out"] is None,"Arithmetic out mutation is unsupported")
            if alpha==1:continue
            other=values["other"]
            if type(other) in (int,float):
                weighted=common_scalar(common_scalar(other)*alpha);added=[]
            elif isinstance(other,torch.fx.Node) and other.op=="get_attr":
                tensor=attribute(other.target)
                require(isinstance(tensor,torch.Tensor) and tensor.dtype in
                        (torch.float32,torch.float64,torch.int32,torch.int64),"Public alpha tensor dtype")
                if tensor.dtype in (torch.int32,torch.int64):
                    require(bool(((tensor>=-(2**53))&(tensor<=2**53)).all()),"Public alpha integer precision")
                weighted=generated_constant(node,tensor.double()*alpha,"alpha_constant");added=[weighted.name]
            else:
                with result.graph.inserting_before(node):
                    weighted=result.graph.call_function(operator.mul,(other,alpha),{})
                added=[weighted.name]
            operation="add" if target in (torch.add,"add") else "subtract"
            node.op="call_function";node.target=operator.add if operation=="add" else operator.sub
            node.args=(values[names[0]],weighted);node.kwargs={}
            entries.append(dict(source_node=node.name,source_kind=kind,rule="static_alpha",
                                attrs=dict(alpha=alpha,operation=operation),
                                normalized_nodes=added+[node.name]))
        elif (kind=="call_function" and target in dropout_functions) or kind=="call_module":
            child=result.get_submodule(target) if kind=="call_module" else None
            if child is not None and type(child) not in dropout_modules:continue
            if child is not None:
                require(len(node.args)==1 and not node.kwargs,"Dropout module arity")
                data=node.args[0];p=child.p;training=child.training;inplace=child.inplace
            else:
                values=common_arguments(node.args,node.kwargs,("input","p","training","inplace"),
                                        {"p":0.5,"training":True,"inplace":False})
                data=values["input"];p=values["p"];training=values["training"];inplace=values["inplace"]
            require(isinstance(data,torch.fx.Node),"Dropout tensor required")
            require(0<=common_scalar(p)<=1,"Dropout probability")
            require(training is False,"Only inference Dropout is supported")
            require(inplace is False,"In-place Dropout is unsupported")
            spatial=(1 if target is F.dropout1d or child is not None and type(child) is torch.nn.Dropout1d else
                     2 if target is F.dropout2d or child is not None and type(child) is torch.nn.Dropout2d else
                     3 if target is F.dropout3d or child is not None and type(child) is torch.nn.Dropout3d else 0)
            if spatial:require(len(shape(data)) in (spatial+1,spatial+2),"Spatial Dropout input rank")
            entries.append(dict(source_node=node.name,source_kind=kind,rule="inference_dropout",
                                attrs=dict(p=p,spatial=spatial,training=False,inplace=False),
                                normalized_nodes=[data.name]))
            node.replace_all_uses_with(data);result.graph.erase_node(node)
        elif kind=="call_function" and target is F.scaled_dot_product_attention:
            values=common_arguments(node.args,node.kwargs,("query","key","value","attn_mask","dropout_p","is_causal"),
                                    {"attn_mask":None,"dropout_p":0.,"is_causal":False})
            require(common_scalar(values["dropout_p"])==0,"SDPA dropout must be zero")
            require(type(values["is_causal"]) is bool,"SDPA causal flag must be static")
            mask=values["attn_mask"];causal=values["is_causal"]
            require(not causal or mask is None,"SDPA causal and explicit mask cannot be combined")
            query,key,value=(values[name] for name in ("query","key","value"))
            qs,ks,vs=(shape(n) for n in (query,key,value))
            score_shape,output_shape=sdpa_shape(qs,ks,vs,resource_profile=resource_profile)
            mask_kind="none";mask_value=None
            if mask is not None:
                mask_value=public_mask(mask,score_shape)
                mask_kind="boolean" if mask_value.dtype is torch.bool else "additive"
            elif causal:
                mask_value=torch.arange(qs[-2])[:,None]>=torch.arange(ks[-2])[None,:]
                mask_kind="causal"
            if mask_kind in ("boolean","causal"):
                require(bool(mask_value.expand(score_shape).any(dim=-1).all()),"SDPA fully masked row unsupported")
                mask=generated_constant(node,mask_value,"mask")
            added=[]
            with result.graph.inserting_before(node):
                kt=result.graph.call_method("transpose",(key,-1,-2),{});added.append(kt)
                logits=result.graph.call_function(operator.matmul,(query,kt),{});added.append(logits)
                scaled=result.graph.call_function(operator.mul,(logits,1/math.sqrt(qs[-1])),{});added.append(scaled)
                if mask_kind=="additive":
                    scaled=result.graph.call_function(operator.add,(scaled,mask),{});added.append(scaled)
                if mask_kind in ("boolean","causal"):
                    exponential=result.graph.call_function(torch.exp,(scaled,),{});added.append(exponential)
                    numerator=result.graph.call_function(operator.mul,(exponential,mask),{});added.append(numerator)
                    total=result.graph.call_function(torch.sum,(numerator,),{"dim":-1,"keepdim":True});added.append(total)
                    reciprocal=result.graph.call_function(torch.reciprocal,(total,),{});added.append(reciprocal)
                    probabilities=result.graph.call_function(operator.mul,(numerator,reciprocal),{});added.append(probabilities)
                else:
                    probabilities=result.graph.call_function(torch.softmax,(scaled,),{"dim":-1});added.append(probabilities)
            node.op="call_function";node.target=operator.matmul;node.args=(probabilities,value);node.kwargs={}
            entries.append(dict(source_node=node.name,source_kind=kind,rule="sdpa",
                attrs=dict(query_shape=qs,key_shape=ks,value_shape=vs,score_shape=score_shape,
                           output_shape=output_shape,scale=1/math.sqrt(qs[-1]),mask_kind=mask_kind),
                normalized_nodes=[n.name for n in added]+[node.name]))
    result.graph.eliminate_dead_code()
    used_attributes={str(n.target) for n in result.graph.nodes if n.op=="get_attr"}
    for item in generated:
        if item["target"] not in used_attributes:delattr(result,item["target"])
    generated=[item for item in generated if item["target"] in used_attributes]
    require(sum(n.op in ("call_function","call_method","call_module") for n in result.graph.nodes)<=limits.original_nodes,
            "Common FX expansion budget")
    result.graph.lint();result.recompile()
    after=result.state_dict();extra={row["target"] for row in generated}
    require(set(after)==set(before)|extra and str(traced.graph)==source and
            all(torch.equal(v,after[k]) and torch.equal(v,traced.state_dict()[k]) for k,v in before.items()),
            "Common normalization changed source or public state")
    live={n.name for n in result.graph.nodes}
    # A dead source operation must not create a false executed-construction claim.
    for entry in entries:
        entry["normalized_nodes"]=[name for name in entry["normalized_nodes"] if name in live]
    record=dict(format=COMMON_POLICY,source_fx_text=source,
                source_fx_sha256=hashlib.sha256(source.encode()).hexdigest(),
                normalized_fx_text=str(result.graph),
                normalized_fx_sha256=hashlib.sha256(str(result.graph).encode()).hexdigest(),
                entries=entries,generated_constants=generated,input_signature=copy.deepcopy(inputs),
                resource_profile=resource_profile,source_preserved=True,public_state_preserved=True)
    record["binding_sha256"]=digest(record)
    return result,record

def bind_common_record(record,graph,arithmetic,constant_sources):
    result=copy.deepcopy(record);result.pop("binding_sha256",None)
    result["captured_graph_sha256"]=digest(graph)
    result["capture_name_map"]=copy.deepcopy(arithmetic["capture_name_map"])
    result["arithmetic_source_fx_sha256"]=arithmetic["source_fx_sha256"]
    for item in result["generated_constants"]:
        names=[name for name,target in constant_sources.items() if target==item["target"]]
        require(len(names)==1,"Generated mask must have one captured public binding")
        item["captured_constant"]=names[0]
    result["binding_sha256"]=digest(result)
    check_common_record(result,graph,arithmetic)
    return result

def check_common_record(record,graph,arithmetic):
    import numpy as np
    require(type(record) is dict and set(record)=={"format","source_fx_text","source_fx_sha256",
        "normalized_fx_text","normalized_fx_sha256","entries","generated_constants","input_signature",
        "resource_profile","source_preserved","public_state_preserved","captured_graph_sha256",
        "capture_name_map","arithmetic_source_fx_sha256","binding_sha256"},"Common capture fields")
    require(record["format"]==COMMON_POLICY and
            digest({k:v for k,v in record.items() if k!="binding_sha256"})==record["binding_sha256"],
            "Common capture binding")
    limits=resolve(record["resource_profile"])
    require(record["source_preserved"] is True and record["public_state_preserved"] is True and
            record["input_signature"]==graph["inputs"] and record["captured_graph_sha256"]==digest(graph),
            "Common source and graph binding")
    require(record["capture_name_map"]==arithmetic["capture_name_map"] and
            record["arithmetic_source_fx_sha256"]==record["normalized_fx_sha256"]==arithmetic["source_fx_sha256"],
            "Common to arithmetic graph binding")
    for text_key,hash_key in (("source_fx_text","source_fx_sha256"),("normalized_fx_text","normalized_fx_sha256")):
        text=record[text_key]
        require(type(text) is str and len(text.encode())<=limits.json_bytes and
                hashlib.sha256(text.encode()).hexdigest()==record[hash_key],"Common FX text binding")
    require(type(record["entries"]) is list and len(record["entries"])<=limits.original_nodes,"Common entry capacity")
    mapping=record["capture_name_map"]
    for entry in record["entries"]:
        require(type(entry) is dict and set(entry)=={"source_node","source_kind","rule","attrs","normalized_nodes"}
                and type(entry["source_node"]) is str and entry["source_kind"] in ("call_function","call_method","call_module")
                and entry["rule"] in ("static_alpha","inference_dropout","sdpa") and type(entry["attrs"]) is dict,
                "Common entry fields")
        require(type(entry["normalized_nodes"]) is list and all(type(n) is str and n in mapping for n in entry["normalized_nodes"]),
                "Common normalized dependency mapping")
        attrs=entry["attrs"]
        if entry["rule"]=="static_alpha":
            require(set(attrs)=={"alpha","operation"} and attrs["operation"] in ("add","subtract"),
                    "Common alpha operation")
            common_scalar(attrs["alpha"])
        elif entry["rule"]=="inference_dropout":
            require(set(attrs)=={"p","spatial","training","inplace"} and attrs["training"] is False and
                    attrs["inplace"] is False and type(attrs["spatial"]) is int and attrs["spatial"] in (0,1,2,3) and
                    0<=common_scalar(attrs["p"])<=1,"Common inference Dropout")
        else:
            require(set(attrs)=={"query_shape","key_shape","value_shape","score_shape","output_shape","scale","mask_kind"}
                    and attrs["mask_kind"] in ("none","boolean","additive","causal"),"Common SDPA attributes")
            scores,output=sdpa_shape(attrs["query_shape"],attrs["key_shape"],attrs["value_shape"],
                                    resource_profile=record["resource_profile"])
            require(attrs["score_shape"]==scores and attrs["output_shape"]==output and
                    attrs["scale"]==1/math.sqrt(attrs["query_shape"][-1]),"Common SDPA shape and scale")
    require(type(record["generated_constants"]) is list and len(record["generated_constants"])<=limits.constants,
            "Common generated constant capacity")
    for item in record["generated_constants"]:
        require(type(item) is dict and set(item)=={"target","shape","dtype","kind","float64_sha256","captured_constant"} and
                type(item["target"]) is str and item["target"].startswith("_poseidon_common_") and
                item["dtype"]=="float64" and item["kind"] in ("mask","alpha_constant") and item["captured_constant"] in graph["constants"],
                "Common generated constant fields")
        array=np.asarray(graph["constants"][item["captured_constant"]],dtype="<f8")
        require(list(array.shape)==item["shape"] and array.size<=limits.constant_elements and
                np.isfinite(array).all() and (item["kind"]!="mask" or np.isin(array,(0.,1.)).all()) and
                hashlib.sha256(array.tobytes(order="C")).hexdigest()==item["float64_sha256"],
                "Common generated mask binding")
    return record
