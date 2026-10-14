"""Provenance-checked public FX folding; uploaded code is never imported here."""
import copy,hashlib
from dataclasses import dataclass
import numpy as np
from benchmark_graph import require,digest
from model_resources import resolve,identity as resource_identity
from pytorch_public_value_ops import POLICY,checked_array,signature,plan

METHODS=frozenset(("reshape","view","permute","transpose","t","unsqueeze","squeeze","flatten","contiguous","float","double","to"))
FUNCTIONS=frozenset(("torch.reshape","torch.permute","torch.transpose","torch.t","torch.unsqueeze","torch.squeeze","torch.flatten","operator.getitem","_operator.getitem","builtins.getattr"))
FORMAT="poseidon-public-expression-flow-v1"
@dataclass(frozen=True)
class Ref:
    name:str

def _decode(value,names,depth=0,budget=None):
    if budget is None:budget=[4096]
    budget[0]-=1
    require(depth<=64 and budget[0]>=0,"public_expression_argument_capacity")
    require(type(value)is dict and type(value.get("kind"))is str,"public_expression_inventory_argument")
    kind=value["kind"]
    if kind=="fx_value":
        require(set(value)=={"kind","name"}and type(value["name"])is str and value["name"]in names,"public_expression_inventory_dependency")
        return names[value["name"]]
    if kind in("none","ellipsis"):
        require(set(value)=={"kind"},"public_expression_inventory_argument")
        return None if kind=="none"else Ellipsis
    if kind in("int","float","bool","str","dtype","device"):
        require(set(value)=={"kind","value"},"public_expression_inventory_argument")
        actual=value["value"]
        expected=int if kind=="int"else float if kind=="float"else bool if kind=="bool"else str
        require(type(actual)is expected,"public_expression_inventory_argument")
        return actual
    if kind in("tuple","list"):
        require(set(value)=={"kind","items"}and type(value["items"])is list,"public_expression_inventory_argument")
        values=[_decode(v,names,depth+1,budget)for v in value["items"]]
        return tuple(values)if kind=="tuple"else values
    if kind=="dict":
        require(set(value)=={"kind","items"}and type(value["items"])is list and
                all(type(i)is list and len(i)==2 and type(i[0])is str for i in value["items"]),"public_expression_inventory_argument")
        require(len({i[0]for i in value["items"]})==len(value["items"]),"public_expression_inventory_duplicate_key")
        return {key:_decode(item,names,depth+1,budget)for key,item in value["items"]}
    if kind=="slice":
        require(set(value)=={"kind","items"}and type(value["items"])is list and len(value["items"])==3,"public_expression_inventory_argument")
        items=[_decode(v,names,depth+1,budget)for v in value["items"]]
        require(all(v is None or type(v)is int for v in items),"public_expression_static_slice")
        return slice(*items)
    raise ValueError("public_expression_inventory_argument")

def _refs(value):
    if type(value)is Ref:return [value]
    if type(value)in(tuple,list):return [r for v in value for r in _refs(v)]
    if type(value)is dict:return [r for v in value.values()for r in _refs(v)]
    return []

def state_identity(values,*,resource_profile):
    require(type(values)is dict and all(type(n)is str and n for n in values),"public_expression_state")
    return {n:signature(v,resource_profile=resource_profile)for n,v in values.items()}

def expected_flow(inventory,original,*,resource_profile="general-tensors-v1"):
    limits=resolve(resource_profile)
    require(type(inventory)is list and len(inventory)<=limits.expanded_nodes+limits.constants+limits.inputs+limits.outputs,"public_expression_graph_capacity")
    require(type(original)is dict and len(original)<=limits.constants and
            all(type(n)is str and n for n in original),"public_expression_state_capacity")
    states={n:checked_array(v,resource_profile=resource_profile)for n,v in original.items()}
    require(sum(v.size for v in states.values())<=limits.total_constant_elements,"public_expression_state_capacity")
    initial=state_identity(states,resource_profile=resource_profile)
    names={};roles={};public={};rewritten=[];entries=[];new_elements=0
    for source in inventory:
        require(type(source)is dict and set(source)=={"name","op","target","args","kwargs","tensor"}and
                type(source["name"])is str and source["name"]not in names and type(source["op"])is str and
                type(source["target"])is str,"public_expression_inventory_node")
        name=source["name"];kind=source["op"];target=source["target"]
        args=_decode(source["args"],names);kwargs=_decode(source["kwargs"],names)
        require(type(args)is tuple and type(kwargs)is dict,"public_expression_inventory_call")
        references=_refs((args,kwargs));row=copy.deepcopy(source);names[name]=Ref(name)
        if kind=="placeholder":
            require(not args and not kwargs,"public_expression_placeholder");roles[name]="cipher";rewritten.append(row);continue
        if kind=="get_attr":
            require(not args and not kwargs and target in states,"public_expression_attribute_not_declared")
            roles[name]="public";public[name]=dict(logical=states[target],physical=states[target],roots=[target],origin=name)
            rewritten.append(row);continue
        require(kind in("call_method","call_function","call_module","output"),"public_expression_inventory_operation")
        registered=(kind=="call_method"and target in METHODS or kind=="call_function"and target in FUNCTIONS)
        first=args[0]if args else kwargs.get("input")
        if registered and type(first)is Ref and roles[first.name]=="public":
            require(all(roles[r.name]=="public"for r in references),"public_expression_dynamic_metadata")
        if kind=="output"or any(roles[r.name]=="cipher"for r in references):
            roles[name]="cipher";rewritten.append(row);continue
        require(references and all(roles[r.name]=="public"for r in references),"public_expression_public_dependency")
        require(kind=="call_method"and target in METHODS or kind=="call_function"and target in FUNCTIONS,"public_expression_operation_not_registered")
        # Torch functions also permit a keyword input. Bind it without executing a callable.
        bound_args=list(args);bound_kwargs=dict(kwargs)
        if kind=="call_function"and target.startswith("torch.")and not bound_args:
            require("input"in bound_kwargs,"public_expression_input")
            bound_args=[bound_kwargs.pop("input")]
        require(bound_args and type(bound_args[0])is Ref and
                all(r is bound_args[0]for r in references),"public_expression_dynamic_metadata")
        parent=public[bound_args[0].name]
        stable_target="operator.getitem"if target=="_operator.getitem"else target
        logical,physical,proof=plan(parent["logical"],kind,stable_target,tuple(bound_args),bound_kwargs,resource_profile=resource_profile)
        attribute="_poseidon_public_expression_"+str(len(entries))
        require(attribute not in states,"public_expression_materialization_collision")
        new_elements+=physical.size
        require(len(entries)<limits.constants and new_elements<=limits.total_constant_elements,"public_expression_derived_capacity")
        states[attribute]=physical
        tensor=copy.deepcopy(row["tensor"])
        if tensor is not None:
            require(type(tensor)is dict and set(tensor)=={"policy","shape","dtype"},"public_expression_tensor_metadata")
            tensor.update(shape=list(physical.shape),dtype={"<f8":"float64","<f4":"float32","|b1":"bool","<i4":"int32","<i8":"int64"}[physical.dtype.str])
        row.update(op="get_attr",target=attribute,args=dict(kind="tuple",items=[]),kwargs=dict(kind="dict",items=[]),tensor=tensor)
        entries.append(dict(source_fx_node=name,source_kind=kind,source_target=target,source_parent=bound_args[0].name,
                            state_name=attribute,roots=list(parent["roots"]),plan=proof))
        public[name]=dict(logical=logical,physical=physical,roots=list(parent["roots"]),origin=name)
        roles[name]="public";rewritten.append(row)
    # Remove only unused get_attr reads. Keep encrypted operations and their order.
    used={r.name for row in rewritten for r in _refs((_decode(row["args"],names),_decode(row["kwargs"],names)))}
    dropped=[row["name"]for row in rewritten if row["op"]=="get_attr"and row["name"]not in used]
    normalized=[row for row in rewritten if row["name"]not in dropped]
    effective={row["target"]:states[row["target"]]for row in normalized if row["op"]=="get_attr"}
    record=dict(format=FORMAT,policy=POLICY,resource_profile=resource_identity(resource_profile),
                source_inventory=copy.deepcopy(inventory),normalized_inventory=normalized,original_state=initial,
                full_normalized_state=state_identity(states,resource_profile=resource_profile),
                effective_state=state_identity(effective,resource_profile=resource_profile),
                dropped_get_attr_nodes=dropped,entries=entries,derived_elements=int(new_elements),
                original_state_preserved=True,cipher_operations_preserved=True,uploaded_code_executed=False)
    record["binding"]=digest(record)
    return states,effective,record

def _arrays(module):
    import torch
    result={}
    for name,value in module.state_dict().items():
        require(value.device.type=="cpu"and value.layout is torch.strided,"public_expression_cpu_state")
        result[name]=np.array(value.detach().contiguous().numpy(),copy=True)
    return result

def normalize(traced,original,*,expected_state,resource_profile="general-tensors-v1"):
    import torch
    from pytorch_arithmetic_normalization import clone_public_state
    from pytorch_fused_matrix_normalization import inventory
    require(isinstance(traced,torch.fx.GraphModule),"public_expression_trusted_graph")
    require(state_identity(original,resource_profile=resource_profile)==expected_state==
            state_identity(_arrays(traced),resource_profile=resource_profile),"public_expression_original_state_binding")
    source_inventory=inventory(traced);states,effective,record=expected_flow(source_inventory,original,resource_profile=resource_profile)
    result=clone_public_state(traced)
    original_meta={node.name:copy.deepcopy(node.meta)for node in traced.graph.nodes}
    for node in result.graph.nodes:node.meta=copy.deepcopy(original_meta[node.name])
    nodes={node.name:node for node in result.graph.nodes}
    for entry in record["entries"]:
        name=entry["state_name"]
        require(not hasattr(result,name),"public_expression_materialization_collision")
        result.register_buffer(name,torch.from_numpy(np.array(states[name],copy=True)))
        node=nodes[entry["source_fx_node"]];node.op="get_attr";node.target=name;node.args=();node.kwargs={}
        if node.meta.get("poseidon_static_tensor")is not None:
            tensor=copy.deepcopy(node.meta["poseidon_static_tensor"])
            tensor.update(shape=list(states[name].shape),
                          dtype={"<f8":"float64","<f4":"float32","|b1":"bool","<i4":"int32","<i8":"int64"}[states[name].dtype.str])
            node.meta["poseidon_static_tensor"]=tensor
    for name in record["dropped_get_attr_nodes"]:
        require(not nodes[name].users and nodes[name].op=="get_attr","public_expression_unused_attribute")
        result.graph.erase_node(nodes[name])
    result.graph.lint();result.recompile()
    require(inventory(result)==record["normalized_inventory"]and inventory(traced)==source_inventory and
            state_identity(_arrays(result),resource_profile=resource_profile)==record["full_normalized_state"]and
            state_identity(_arrays(traced),resource_profile=resource_profile)==expected_state,"public_expression_source_graph_bridge")
    return result,record

def check(record,original,*,expected_binding,expected_source_inventory,resource_profile="general-tensors-v1"):
    require(type(record)is dict and record.get("binding")==expected_binding==
            digest({k:v for k,v in record.items()if k!="binding"}),"public_expression_record_binding")
    states,effective,expected=expected_flow(expected_source_inventory,original,resource_profile=resource_profile)
    require(record==expected,"public_expression_record_changed")
    return states,effective
