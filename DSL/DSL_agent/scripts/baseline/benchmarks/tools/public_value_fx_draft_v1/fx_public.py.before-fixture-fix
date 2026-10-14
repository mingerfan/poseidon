"""Prototype for registered public-only FX expressions. No production intake hook."""
import copy,hashlib,operator,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(HERE.parent/"public_value_expression_draft_v1"))
from public_values import evaluate,signature,MAX_ELEMENTS
from model_resources import resolve,identity as resource_identity
from benchmark_graph import digest
from pytorch_arithmetic_normalization import clone_public_state
FORMAT="poseidon-public-value-fx-draft-v1"
DTYPES=(np.dtype("float64"),np.dtype("float32"),np.dtype("bool"),np.dtype("int32"),np.dtype("int64"))

def require(ok,code):
    if not ok:raise ValueError(code)

def arrays(module):
    result={}
    for name,value in module.state_dict().items():
        require(value.device.type=="cpu" and value.layout is torch.strided,"public_state_device_layout")
        result[name]=np.array(value.detach().contiguous().numpy(),copy=True)
    return result

def state_identity(values,resource_profile):
    limits=resolve(resource_profile)
    require(type(values)is dict and len(values)<=limits.constants,"public_state_count")
    total=0;result={}
    for name,value in values.items():
        require(type(name)is str and name and type(value)is np.ndarray and value.dtype in DTYPES,"public_state_type")
        require(value.ndim<=limits.rank and 0<value.size<=limits.constant_elements and
                all(0<d<=limits.axis_size for d in value.shape),"public_state_capacity")
        total+=value.size
        require(np.isfinite(value).all(),"public_state_nonfinite")
        result[name]=dict(dtype=value.dtype.str,shape=list(value.shape),
                         sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())
    require(total<=limits.total_constant_elements,"public_state_capacity")
    return result

def graph_hash(module):
    return hashlib.sha256(str(module.graph).encode()).hexdigest()

def refs(item):
    if isinstance(item,torch.fx.Node):return [item]
    if type(item)in(tuple,list):return [node for value in item for node in refs(value)]
    if type(item)is dict:return [node for value in item.values()for node in refs(value)]
    return []

def static_sequence(value):
    require(type(value)in(tuple,list) and all(type(n)is int for n in value),"public_static_sequence")
    return list(value)

def transform(node,parent):
    args=list(node.args);kwargs=dict(node.kwargs)
    require(args and args[0]is parent,"public_expression_input")
    tail=args[1:];target=node.target
    if node.op=="call_method" and target in("reshape","view","permute"):
        require(not kwargs,"public_expression_kwargs")
        values=static_sequence(tail[0])if len(tail)==1 and type(tail[0])in(tuple,list)else static_sequence(tail)
        return ("permute",dict(dims=values))if target=="permute"else("reshape",dict(shape=values))
    if node.op=="call_function" and target in(torch.reshape,torch.permute):
        require(len(tail)==1 and not kwargs,"public_expression_arity")
        values=static_sequence(tail[0])
        return ("permute",dict(dims=values))if target is torch.permute else("reshape",dict(shape=values))
    if node.op=="call_method" and target in("float","double"):
        require(not tail and not kwargs,"public_expression_arity")
        return "bool_cast",dict(dtype="float32"if target=="float"else"float64")
    if node.op=="call_method" and target=="to":
        require((len(tail)==1 and not kwargs)or(not tail and set(kwargs)=={"dtype"}),"public_cast_arguments")
        dtype=tail[0]if tail else kwargs["dtype"]
        require(dtype is torch.float32 or dtype is torch.float64,"public_cast_dtype")
        return "bool_cast",dict(dtype="float32"if dtype is torch.float32 else"float64")
    if node.op=="call_function" and target is operator.getitem:
        require(len(tail)==1 and not kwargs,"public_expression_arity")
        items=tail[0]if type(tail[0])is tuple else(tail[0],)
        selectors=[]
        for item in items:
            if item is None or type(item)is int:selectors.append(item)
            elif type(item)is slice:
                selectors.append(dict(start=item.start,stop=item.stop,step=item.step))
            else:raise ValueError("public_static_slice")
        return "slice",dict(selectors=selectors)
    raise ValueError("public_operation_not_registered")

def normalize(traced,original_state,*,expected_state,source_sha256,resource_profile="general-tensors-v1"):
    require(isinstance(traced,torch.fx.GraphModule),"trusted_fx_graph_required")
    require(type(source_sha256)is str and len(source_sha256)==64 and all(c in"0123456789abcdef"for c in source_sha256),"public_source_binding")
    expected=state_identity(original_state,resource_profile)
    require(type(expected_state)is dict and expected==expected_state and
            state_identity(arrays(traced),resource_profile)==expected,"public_state_binding")
    limits=resolve(resource_profile)
    require(len(list(traced.graph.nodes))<=limits.expanded_nodes+limits.constants+limits.inputs+limits.outputs,"public_graph_capacity")
    before=graph_hash(traced);result=clone_public_state(traced)
    metadata={node.name:copy.deepcopy(node.meta)for node in traced.graph.nodes}
    for node in result.graph.nodes:node.meta=copy.deepcopy(metadata[node.name])
    public={};roles={};entries=[];materialized={}
    for node in list(result.graph.nodes):
        if node.op=="placeholder":roles[node]="cipher";continue
        if node.op=="get_attr":
            require(type(node.target)is str and node.target in original_state,"public_attribute_not_declared")
            public[node]=dict(value=np.array(original_state[node.target],copy=True),origin=node.name,roots=[node.target])
            roles[node]="public";continue
        operands=refs((node.args,node.kwargs))
        if node.op=="output":
            roles[node]="cipher"if any(roles.get(n)=="cipher"for n in operands)else"public";continue
        require(all(n in roles for n in operands),"public_graph_dependency")
        is_transform=(node.op=="call_method"and node.target in("reshape","view","permute","float","double","to")or
                      node.op=="call_function"and node.target in(torch.reshape,torch.permute,operator.getitem))
        if is_transform and node.args and isinstance(node.args[0],torch.fx.Node)and node.args[0]in public:
            require(all(n is node.args[0]for n in operands),"public_expression_dynamic_metadata")
        require(operands,"public_operation_not_registered")
        if any(roles[n]=="cipher"for n in operands):
            roles[node]="cipher";continue
        require(node.op in("call_method","call_function")and operands,"public_operation_not_registered")
        parent=node.args[0]if node.args else None
        require(isinstance(parent,torch.fx.Node)and parent in public and all(n is parent for n in operands),"public_expression_dynamic_metadata")
        op,attrs=transform(node,parent);value=public[parent]["value"]
        # This prototype reuses the separately tested small-expression mathematics.
        # Full declared weights remain bound by the selected existing resource profile.
        require(value.size<=MAX_ELEMENTS,"public_expression_draft_capacity")
        step=dict(name=node.name,op=op,input=public[parent]["origin"],attributes=attrs)
        computed,proof=evaluate({step["input"]:value},[step]);derived=computed[node.name]
        attribute="_poseidon_public_expression_"+str(len(entries))
        require(not hasattr(result,attribute),"public_materialization_collision")
        result.register_buffer(attribute,torch.from_numpy(np.array(derived,copy=True)))
        with result.graph.inserting_before(node):replacement=result.graph.get_attr(attribute)
        replacement.meta=copy.deepcopy(node.meta)
        entry=dict(source_fx_node=node.name,source_kind=node.op,source_target=node.target if node.op=="call_method"else {torch.reshape:"torch.reshape",torch.permute:"torch.permute",operator.getitem:"operator.getitem"}[node.target],
                   step=proof["steps"][0],attribute=attribute,roots=list(public[parent]["roots"]))
        entries.append(entry);materialized[attribute]=signature(derived)
        public[replacement]=dict(value=derived,origin=node.name,roots=list(public[parent]["roots"]))
        roles[replacement]="public";node.replace_all_uses_with(replacement);result.graph.erase_node(node)
    result.graph.lint();result.recompile()
    final=state_identity(arrays(result),resource_profile)
    require(all(final.get(n)==v for n,v in expected.items())and
            graph_hash(traced)==before and state_identity(arrays(traced),resource_profile)==expected,"public_source_mutated")
    record=dict(format=FORMAT,source_sha256=source_sha256,source_fx_sha256=before,
                original_state=expected,normalized_fx_sha256=graph_hash(result),normalized_state=final,
                resource_profile=resource_identity(resource_profile),entries=entries,derived=materialized,
                production_integrated=False,expression_element_limit=MAX_ELEMENTS)
    record["binding"]=digest(record)
    return result,record

def check(normalized,record,original_state,*,expected_binding,source_sha256,resource_profile="general-tensors-v1"):
    require(type(record)is dict and record.get("binding")==expected_binding==
            digest({k:v for k,v in record.items()if k!="binding"}),"public_record_authority")
    require(record.get("format")==FORMAT and record.get("source_sha256")==source_sha256 and
            record.get("resource_profile")==resource_identity(resource_profile)and record.get("production_integrated")is False,"public_record_identity")
    require(state_identity(original_state,resource_profile)==record["original_state"]and
            graph_hash(normalized)==record["normalized_fx_sha256"]and
            state_identity(arrays(normalized),resource_profile)==record["normalized_state"],"public_record_state")
    values={};state=arrays(normalized)
    for entry in record["entries"]:
        step=entry["step"];parent=step["input"]
        if parent not in values:
            require(len(entry["roots"])==1 and entry["roots"][0]in original_state,"public_replay_root")
            values[parent]=np.array(original_state[entry["roots"][0]],copy=True)
        fields={k:step[k]for k in("name","op","input","attributes")}
        derived,proof=evaluate({parent:values[parent]},[fields])
        require(proof["steps"][0]==step and
                signature(state[entry["attribute"]])==signature(derived[step["name"]])==record["derived"][entry["attribute"]],"public_replay_payload")
        values[step["name"]]=derived[step["name"]]
    require(set(record["derived"])=={e["attribute"]for e in record["entries"]},"public_replay_inventory")
    return True
