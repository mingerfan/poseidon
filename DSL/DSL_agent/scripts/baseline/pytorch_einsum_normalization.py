"""Deterministic static einsum normalization to registered tensor primitives.

This module plans shape-only rewrites. It never loads a model, creates DSL, or
changes a backend. Equation text is parsed as data.
"""
import copy
import hashlib
import math
import re
from pathlib import Path

from benchmark_graph import canonical, digest, require
from model_resources import resolve

POLICY = "poseidon-pytorch-static-einsum-v1"
MAX_OPERANDS = 4

def arguments(args, kwargs):
    require(type(args) in (tuple, list) and not kwargs and len(args) >= 2,
            "Static einsum equation and operands")
    equation = args[0]
    operands = list(args[1:])
    if len(operands) == 1 and type(operands[0]) in (tuple, list):
        operands = list(operands[0])
    require(type(equation) is str and 1 <= len(equation) <= 256 and
            1 <= len(operands) <= MAX_OPERANDS, "Static einsum argument budget")
    return equation, operands

def _parse(equation, shapes):
    require(type(equation) is str and 1 <= len(equation) <= 256,
            "Static einsum equation budget")
    text = "".join(c for c in equation if c.isspace() is False)
    require(text.count("->") <= 1, "Static einsum output delimiter")
    parts = text.split("->")
    inputs = parts[0].split(",")
    require(len(inputs) == len(shapes), "Static einsum operand count")
    tokens = []
    ellipsis_ranks = []
    for term, shape in zip(inputs, shapes):
        require(re.fullmatch(r"[A-Za-z]*(?:\.\.\.[A-Za-z]*)?", term) is not None,
                "Static einsum operand labels")
        letters = [c for c in term if c.isalpha()]
        require(len(set(letters)) == len(letters),
                "Diagonal einsum labels are not registered")
        hidden = len(shape) - len(letters)
        require(hidden >= 0 and ("..." in term or hidden == 0),
                "Static einsum operand rank")
        tokens.append(term)
        ellipsis_ranks.append(hidden)
    ellipsis_rank = max(ellipsis_ranks)
    ellipsis_labels = ["@"+str(i) for i in range(ellipsis_rank)]
    labels = []
    for term, rank in zip(tokens, ellipsis_ranks):
        if "..." in term:
            a, b = term.split("...")
            row = list(a) + ellipsis_labels[ellipsis_rank-rank:] + list(b)
        else:
            row = list(term)
        labels.append(row)
    occurrences = {}
    for row in labels:
        for label in row:
            occurrences[label] = occurrences.get(label, 0) + 1
    if len(parts) == 2:
        term = parts[1]
        require(re.fullmatch(r"[A-Za-z]*(?:\.\.\.[A-Za-z]*)?", term) is not None,
                "Static einsum output labels")
        if "..." in term:
            a, b = term.split("...")
            output = list(a) + ellipsis_labels + list(b)
        else:
            output = list(term)
        require(len(output) == len(set(output)) and all(l in occurrences for l in output),
                "Static einsum output label inventory")
    else:
        output = ellipsis_labels + sorted(l for l, n in occurrences.items()
                                          if not l.startswith("@") and n == 1)
    dimensions = {}
    for row, shape in zip(labels, shapes):
        for label, width in zip(row, shape):
            previous = dimensions.get(label, 1)
            require(previous == width or previous == 1 or width == 1,
                    "Static einsum label broadcast")
            dimensions[label] = max(previous, width)
    return text, labels, output, dimensions

def plan(equation, shapes, *, resource_profile="general-tensors-v1"):
    limits = resolve(resource_profile)
    require(type(shapes) is list and 1 <= len(shapes) <= MAX_OPERANDS,
            "Static einsum operand budget")
    def valid_shape(shape):
        return type(shape) is list and len(shape) <= limits.rank and \
            all(type(n) is int and 0 < n <= limits.axis_size for n in shape) and \
            math.prod(shape) <= limits.tensor_elements
    require(all(valid_shape(s) and len(s) >= 1 for s in shapes),
            "Static einsum tensor shape budget")
    text, labels, output, sizes = _parse(equation, shapes)
    result_shape = [sizes[l] for l in output]
    require(valid_shape(result_shape), "Static einsum output shape budget")
    steps = []
    current_shapes = {"operand_"+str(i): list(s) for i, s in enumerate(shapes)}
    def emit(op, inputs, attrs, shape):
        require(valid_shape(shape), "Static einsum intermediate shape budget")
        require(len(steps) < limits.original_nodes, "Static einsum expansion budget")
        name = "step_"+str(len(steps))
        steps.append(dict(op=op, inputs=list(inputs), attrs=copy.deepcopy(attrs), shape=list(shape)))
        current_shapes[name] = list(shape)
        return name
    def permute(value, current, target):
        axes = [current.index(l) for l in target]
        if axes == list(range(len(axes))):
            return value
        return emit("permute", [value], dict(dims=axes),
                    [current_shapes[value][i] for i in axes])
    strategy = "broadcast_product_reduce"
    native = False
    if len(shapes) == 2:
        left, right = labels
        batch = [l for l in output if l in left and l in right]
        contract = sorted(l for l in left if l in right and l not in output)
        left_output = [l for l in output if l in left and l not in right]
        right_output = [l for l in output if l in right and l not in left]
        own_sizes = [dict(zip(row, shape)) for row, shape in zip(labels, shapes)]
        equal_contract = all(own_sizes[0][l] == own_sizes[1][l] for l in contract)
        matrix_shapes = [
            [own_sizes[0][l] for l in batch]+[
                math.prod(own_sizes[0][l] for l in left_output),
                math.prod(own_sizes[0][l] for l in contract)],
            [own_sizes[1][l] for l in batch]+[
                math.prod(own_sizes[1][l] for l in contract),
                math.prod(own_sizes[1][l] for l in right_output)]]
        product_shape = [sizes[l] for l in batch]+[matrix_shapes[0][-2],matrix_shapes[1][-1]]
        native = equal_contract and all(valid_shape(s) for s in matrix_shapes+[product_shape])
        if native:
            ordered = [batch+left_output+contract, batch+contract+right_output]
            values = []
            for i in range(2):
                current = labels[i]
                value = "operand_"+str(i)
                reduce_axes = [j for j, l in enumerate(current) if l not in ordered[i]]
                if reduce_axes:
                    remaining = [l for j,l in enumerate(current) if j not in reduce_axes]
                    value = emit("sum", [value], dict(axes=reduce_axes),
                                 [own_sizes[i][l] for l in remaining])
                    current = remaining
                value = permute(value, current, ordered[i])
                if current_shapes[value] != matrix_shapes[i]:
                    value = emit("reshape", [value], dict(shape=matrix_shapes[i]), matrix_shapes[i])
                values.append(value)
            value = emit("matmul", values, {}, product_shape)
            grouped = batch+left_output+right_output
            expanded = [sizes[l] for l in grouped]
            if current_shapes[value] != expanded:
                value = emit("reshape", [value], dict(shape=expanded), expanded)
            value = permute(value, grouped, output)
            strategy = "grouped_static_matmul"
    if not native:
        union = output+sorted(l for l in sizes if l not in output)
        union_shape = [sizes[l] for l in union]
        require(valid_shape(union_shape), "Static einsum broadcast workspace budget")
        values = []
        for i, current in enumerate(labels):
            value = "operand_"+str(i)
            target = [l for l in union if l in current]
            value = permute(value, current, target)
            own = dict(zip(current, shapes[i]))
            expanded = [own[l] if l in own else 1 for l in union]
            if current_shapes[value] != expanded:
                value = emit("reshape", [value], dict(shape=expanded), expanded)
            values.append(value)
        value = values[0]
        for other in values[1:]:
            a, b = current_shapes[value], current_shapes[other]
            shape = [max(x,y) for x,y in zip(a,b)]
            value = emit("multiply", [value,other], {}, shape)
        if len(union) > len(output):
            value = emit("sum", [value], dict(axes=list(range(len(output),len(union)))),result_shape)
    if not steps:
        value = emit("permute", ["operand_0"], dict(dims=list(range(len(shapes[0])))),shapes[0])
    require(current_shapes[value] == result_shape and value == "step_"+str(len(steps)-1),
            "Static einsum result planning")
    return dict(policy=POLICY,equation=text,source_shapes=copy.deepcopy(shapes),
                operand_labels=labels,output_labels=output,result_shape=result_shape,
                strategy=strategy,steps=steps,resource_profile=resource_profile,
                floating_semantics="finite float64 reference checked; association may differ")

def _literal(value):
    if type(value) is str:return dict(kind="str",value=value)
    if type(value) is int:return dict(kind="int",value=value)
    if type(value) in (list,tuple):return dict(kind="tuple",items=[_literal(v)for v in value])
    raise ValueError("Unregistered einsum plan literal")

def step_inventory(step, references, name):
    value = lambda n: dict(kind="fx_value",name=references[n])
    op = step["op"]; attrs = step["attrs"]
    args = [value(n) for n in step["inputs"]]
    kwargs = {}
    if op in ("permute","reshape"):
        kind,target="call_method",op
        args.append(_literal(attrs["dims" if op=="permute"else "shape"]))
    elif op=="sum":
        kind,target="call_function","torch.sum"
        kwargs={"dim":_literal(attrs["axes"]),"keepdim":dict(kind="bool",value=False)}
    elif op=="matmul":kind,target="call_function","torch.matmul"
    elif op=="multiply":kind,target="call_function","_operator.mul"
    else:raise ValueError("Unregistered einsum primitive")
    return dict(name=name,op=kind,target=target,args=dict(kind="tuple",items=args),
                kwargs=dict(kind="dict",items=[[k,v]for k,v in kwargs.items()]),
                tensor=dict(policy="signature-static-ordered-alias-staging-v1",
                            shape=step["shape"],dtype="float64"))

def normalize(traced, inputs, *, resource_profile="general-tensors-v1"):
    import torch, operator
    from pytorch_static_trace import ALIAS_POLICY, policy_record
    from pytorch_arithmetic_normalization import clone_public_state
    from pytorch_fused_matrix_normalization import inventory, _state
    require(isinstance(traced,torch.fx.GraphModule)and traced.meta.get("einsum_source_policy")==POLICY,
            "Explicit einsum trace policy")
    require(traced.meta.get("ordered_alias_capture_policy")==
            policy_record(inputs,resource_profile,policy=ALIAS_POLICY),"Einsum ordered trace policy")
    original = str(traced.graph);state=_state(traced)
    result = clone_public_state(traced);entries=[]
    for node in list(result.graph.nodes):
        if node.op!="call_function"or node.target is not torch.einsum:continue
        equation,operands=arguments(node.args,node.kwargs)
        require(all(isinstance(v,torch.fx.Node)for v in operands),"Static einsum tensor operands")
        metas=[v.meta.get("poseidon_static_tensor")for v in operands]
        require(all(type(m)is dict and m.get("policy")==ALIAS_POLICY and m.get("dtype")=="float64"
                    for m in metas),"Static einsum float64 metadata")
        selected=plan(equation,[m["shape"]for m in metas],resource_profile=resource_profile)
        require(node.meta.get("poseidon_static_tensor")==
                dict(policy=ALIAS_POLICY,shape=selected["result_shape"],dtype="float64"),
                "Static einsum result metadata")
        references={"operand_"+str(i):v for i,v in enumerate(operands)};emitted=[]
        for i,step in enumerate(selected["steps"]):
            refs={k:v.name for k,v in references.items()}
            proto=step_inventory(step,refs,node.name)
            args=tuple(references[n]for n in step["inputs"]);kwargs={}
            if step["op"]=="permute":args+=(tuple(step["attrs"]["dims"]),)
            elif step["op"]=="reshape":args+=(tuple(step["attrs"]["shape"]),)
            elif step["op"]=="sum":kwargs=dict(dim=tuple(step["attrs"]["axes"]),keepdim=False)
            target={"sum":torch.sum,"matmul":torch.matmul,"multiply":operator.mul}.get(step["op"],step["op"])
            if i==len(selected["steps"])-1:
                created=node;created.op=proto["op"];created.target=target
                created.args=args;created.kwargs=kwargs
            else:
                with result.graph.inserting_before(node):
                    created=result.graph.create_node(proto["op"],target,args,kwargs)
            created.meta["poseidon_static_tensor"]=proto["tensor"]
            references["step_"+str(i)]=created;emitted.append(created.name)
        entries.append(dict(source_node=node.name,operands=[v.name for v in operands],
                            plan=selected,emitted_nodes=emitted))
    result.graph.lint();result.recompile()
    require(sum(n.op in("call_function","call_method","call_module")for n in result.graph.nodes)
            <=resolve(resource_profile).original_nodes,"Einsum normalized FX budget")
    require(original==str(traced.graph)and state==_state(result),"Einsum source or public state changed")
    text=str(result.graph)
    record=dict(format=POLICY,input_signature=copy.deepcopy(inputs),resource_profile=resource_profile,
                source_fx_text=original,source_fx_sha256=hashlib.sha256(original.encode()).hexdigest(),
                normalized_fx_text=text,normalized_fx_sha256=hashlib.sha256(text.encode()).hexdigest(),
                source_inventory=inventory(traced),normalized_inventory=inventory(result),entries=entries,
                source_order_preserved=True,public_state_preserved=True,
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record["binding"]=digest(record);check_transport(record,inputs=inputs,resource_profile=resource_profile)
    return result,record

def check_transport(record, *, inputs, resource_profile):
    fields={"format","input_signature","resource_profile","source_fx_text","source_fx_sha256",
            "normalized_fx_text","normalized_fx_sha256","source_inventory","normalized_inventory",
            "entries","source_order_preserved","public_state_preserved","implementation_sha256","binding"}
    limits=resolve(resource_profile)
    require(type(record)is dict and set(record)==fields and record["format"]==POLICY and
            record["binding"]==digest({k:v for k,v in record.items()if k!="binding"}),"Einsum record binding")
    require(record["input_signature"]==inputs and record["resource_profile"]==resource_profile and
            record["source_order_preserved"]is True and record["public_state_preserved"]is True and
            record["implementation_sha256"]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "Einsum record source identity")
    require(len(canonical(record))<=limits.json_bytes,"Einsum record budget")
    for a,b in[("source_fx_text","source_fx_sha256"),("normalized_fx_text","normalized_fx_sha256")]:
        require(type(record[a])is str and hashlib.sha256(record[a].encode()).hexdigest()==record[b],
                "Einsum FX text binding")
    source=record["source_inventory"];normalized=record["normalized_inventory"]
    cap=limits.original_nodes+limits.constants+limits.inputs+limits.outputs
    keys={"name","op","target","args","kwargs","tensor"}
    require(type(source)is list and type(normalized)is list and 1<=len(source)<=cap and
            1<=len(normalized)<=cap and all(type(n)is dict and set(n)==keys and type(n["name"])is str
            for n in source+normalized),"Einsum inventory fields and budget")
    nodes={n["name"]:n for n in source};require(len(nodes)==len(source),"Einsum source names")
    require(type(record["entries"])is list and len(record["entries"])<=limits.original_nodes,"Einsum entry budget")
    expected=[];entries={};fresh=set()
    def unpack(value):
        require(type(value)is dict,"Einsum inventory argument")
        if value.get("kind")=="str"and set(value)=={"kind","value"}and type(value["value"])is str:return value["value"]
        if value.get("kind")=="fx_value"and set(value)=={"kind","name"}and value["name"]in nodes:return value
        if value.get("kind")in("tuple","list")and set(value)=={"kind","items"}and type(value["items"])is list:
            return tuple(unpack(v)for v in value["items"])
        raise ValueError("Unregistered einsum source argument")
    for entry in record["entries"]:
        require(type(entry)is dict and set(entry)=={"source_node","operands","plan","emitted_nodes"}and
                entry["source_node"]in nodes and entry["source_node"]not in entries,"Einsum entry source")
        node=nodes[entry["source_node"]]
        require(node["op"]=="call_function"and node["target"]=="torch.functional.einsum"and
                node["kwargs"]==dict(kind="dict",items=[]),"Einsum registered source call")
        equation,operands=arguments(unpack(node["args"]),{})
        require(entry["operands"]==[v["name"]for v in operands]and all(type(v)is dict and
                v==dict(kind="fx_value",name=v["name"])for v in operands),"Einsum operand binding")
        metas=[nodes[v["name"]]["tensor"]for v in operands]
        require(all(type(m)is dict and set(m)=={"policy","shape","dtype"}and
                    m["policy"]=="signature-static-ordered-alias-staging-v1"and m["dtype"]=="float64"
                    for m in metas),"Einsum operand metadata")
        selected=plan(equation,[m["shape"]for m in metas],resource_profile=resource_profile)
        require(entry["plan"]==selected and node["tensor"]==
                dict(policy="signature-static-ordered-alias-staging-v1",shape=selected["result_shape"],dtype="float64"),
                "Einsum plan recomputation")
        emitted=entry["emitted_nodes"]
        require(type(emitted)is list and len(emitted)==len(selected["steps"])and
                all(type(n)is str for n in emitted)and len(set(emitted))==len(emitted)and
                emitted[-1]==node["name"]and not set(emitted[:-1]).intersection(set(nodes)|fresh),
                "Einsum emission inventory")
        fresh.update(emitted[:-1]);references={"operand_"+str(i):v["name"]for i,v in enumerate(operands)}
        replacement=[]
        for i,(step,name)in enumerate(zip(selected["steps"],emitted)):
            replacement.append(step_inventory(step,references,name));references["step_"+str(i)]=name
        entries[node["name"]]=replacement
    registered={n["name"]for n in source if n["op"]=="call_function"and n["target"]=="torch.functional.einsum"}
    require(set(entries)==registered,"Einsum source call inventory")
    for node in source:expected.extend(entries[node["name"]]if node["name"]in entries else[copy.deepcopy(node)])
    require(expected==normalized,"Einsum complete normalization projection")
    require(sum(n["op"]in("call_function","call_method","call_module")for n in normalized)
            <=limits.original_nodes,"Einsum normalized expansion budget")
    return record
