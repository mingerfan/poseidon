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

def normalize(traced,*,resource_profile="legacy-v1"):
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
    result=copy.deepcopy(traced);roles={};entries=[]
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
