from pytorch_provenance_transport_source import SOURCE_FORMAT as TRANSPORT_SOURCE_FORMAT, PACKAGE_FORMAT as TRANSPORT_PACKAGE_FORMAT, REFERENCE_FORMAT as TRANSPORT_REFERENCE_FORMAT
from pytorch_logical_view_source import SOURCE_FORMAT as VIEW_SOURCE_FORMAT, PACKAGE_FORMAT as VIEW_PACKAGE_FORMAT, REFERENCE_FORMAT as VIEW_REFERENCE_FORMAT
from pytorch_public_join_source import SOURCE_FORMAT as JOIN_SOURCE_FORMAT, PACKAGE_FORMAT as JOIN_PACKAGE_FORMAT, REFERENCE_FORMAT as JOIN_REFERENCE_FORMAT
from pytorch_public_mask_source import SOURCE_FORMAT as MASK_SOURCE_FORMAT, PACKAGE_FORMAT as MASK_PACKAGE_FORMAT, REFERENCE_FORMAT as MASK_REFERENCE_FORMAT
from pytorch_public_constant_source import SOURCE_FORMAT as CONSTANT_SOURCE_FORMAT, PACKAGE_FORMAT as CONSTANT_PACKAGE_FORMAT, REFERENCE_FORMAT as CONSTANT_REFERENCE_FORMAT
"""Explicit V30 static repetition/permutation source normalization."""
import copy, hashlib
from pathlib import Path
from benchmark_graph import require, digest, canonical
from model_resources import resolve
from static_tensor_transform_plan import plan, OPERATORS

POLICY = "poseidon-pytorch-static-tensor-transforms-v1"
SOURCE_FORMAT = "poseidon-trusted-pytorch-v30"
PACKAGE_FORMAT = "poseidon-trusted-pytorch-provenance-v30"
REFERENCE_FORMAT = "poseidon-trusted-pytorch-reference-v30"
from pytorch_public_tensor_transform_source import SOURCE_FORMAT as STATE_TRANSFORM_SOURCE_FORMAT, PACKAGE_FORMAT as STATE_TRANSFORM_PACKAGE_FORMAT, REFERENCE_FORMAT as STATE_TRANSFORM_REFERENCE_FORMAT
CAPTURE_FORMAT = "poseidon-public-expression-source-capture-v7"
FUNCTIONS = frozenset("torch."+name for name in ("tile","repeat_interleave","flip","roll"))

def selected(spec):
    return type(spec) is dict and spec.get("format") in (SOURCE_FORMAT,STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT,MASK_SOURCE_FORMAT,JOIN_SOURCE_FORMAT,VIEW_SOURCE_FORMAT,TRANSPORT_SOURCE_FORMAT)

def operation(kind, target):
    if kind=="call_method" and target in OPERATORS: return target
    if kind=="call_function":
        if type(target) is str: return target.removeprefix("torch.") if target in FUNCTIONS else None
        import torch
        for name in ("tile","repeat_interleave","flip","roll"):
            if target is getattr(torch,name): return name
    return None

def arguments(op, args, kwargs, kind):
    require(type(args) in (tuple,list) and type(kwargs) is dict,"static_transform_call")
    args=list(args); kwargs=dict(kwargs)
    if op=="repeat" and kind=="call_method":
        require(args and not(set(kwargs)-{"repeats"}),"static_transform_repeat_signature")
        source=args.pop(0)
        require(not(args and kwargs),"static_transform_duplicate_argument")
        factors=kwargs.get("repeats",args[0] if len(args)==1 and type(args[0]) in (list,tuple) else args)
        require(type(factors) in (tuple,list),"static_transform_static_repeats")
        return source,dict(repeats=list(factors))
    fields = {"tile":("input","dims"),"repeat_interleave":("input","repeats","dim","output_size"),
              "flip":("input","dims"),"roll":("input","shifts","dims")}[op]
    defaults = {"repeat_interleave":dict(dim=None,output_size=None),"roll":dict(dims=None)}.get(op,{})
    require(len(args)<=len(fields) and not(set(kwargs)-set(fields)),"static_transform_signature")
    if kind=="call_method": require("input" not in kwargs,"static_transform_receiver_keyword")
    if op=="repeat_interleave": require(len(args)<=3,"static_transform_output_size_keyword_only")
    bound=dict(zip(fields,args))
    require(not(set(bound)&set(kwargs)),"static_transform_duplicate_argument")
    bound.update(kwargs)
    for name in fields:
        if name not in bound:
            require(name in defaults,"static_transform_missing_argument")
            bound[name]=defaults[name]
    def ints(value):
        return [value] if type(value) is int else list(value) if type(value) in (tuple,list) else value
    if op=="tile":
        require(type(bound["dims"]) in (tuple,list),"static_transform_static_repeats")
        attrs=dict(repeats=list(bound["dims"]))
    elif op=="repeat_interleave":attrs=dict(repeats=bound["repeats"],axis=bound["dim"],output_size=bound["output_size"])
    elif op=="flip":
        require(type(bound["dims"]) in (tuple,list),"static_transform_static_axes")
        attrs=dict(axes=list(bound["dims"]))
    else: attrs=dict(shifts=ints(bound["shifts"]),axes=None if bound["dims"] is None else ints(bound["dims"]))
    return bound["input"],attrs

def metadata(kind,target,args,kwargs,*,resolve_value,resource_profile):
    op=operation(kind,target)
    if op is None:return None
    import torch
    source,attrs=arguments(op,args,kwargs,kind); source=resolve_value(source)
    require(isinstance(source,torch.Tensor) and source.dtype in (torch.float32,torch.float64),
            "static_transform_tensor_metadata")
    selected_plan=plan(op,list(source.shape),attrs,resource_profile=resource_profile)
    return torch.empty(selected_plan["output_shape"],device="meta",dtype=source.dtype)

def implementation():
    base=Path(__file__).parent
    return {n:hashlib.sha256((base/n).read_bytes()).hexdigest()
            for n in ("pytorch_static_tensor_transforms.py","static_tensor_transform_plan.py")}

def normalized_inventory(source,*,resource_profile):
    from pytorch_public_expression_flow import _decode, Ref
    from pytorch_static_trace import ALIAS_POLICY
    limits=resolve(resource_profile); capacity=limits.expanded_nodes+limits.constants+limits.inputs+limits.outputs
    require(type(source) is list and 1<=len(source)<=capacity,"static_transform_inventory_capacity")
    used={n["name"] for n in source};seen={};cipher={};result=[];entries=[]
    def encode(value):
        if type(value) is Ref:return dict(kind="fx_value",name=value.name)
        if type(value) is tuple:return dict(kind="tuple",items=[encode(v) for v in value])
        if type(value) is list:return dict(kind="list",items=[encode(v) for v in value])
        if value is None:return dict(kind="none")
        require(type(value) is int,"static_transform_generated_argument")
        return dict(kind="int",value=value)
    def dependency(value):
        if type(value) is Ref:return cipher[value.name]
        if type(value) in (tuple,list):return any(dependency(v) for v in value)
        if type(value) is dict:return any(dependency(v) for v in value.values())
        return False
    for item in source:
        require(type(item) is dict and set(item)=={"name","op","target","args","kwargs","tensor"} and
                type(item["name"]) is str and item["name"] not in seen,"static_transform_inventory_identity")
        names={n:Ref(n) for n in seen};args=_decode(item["args"],names);kwargs=_decode(item["kwargs"],names)
        require(type(args) is tuple and type(kwargs) is dict,"static_transform_inventory_arguments")
        op=operation(item["op"],item["target"])
        cipher[item["name"]]=item["op"]=="placeholder" or dependency(args) or dependency(kwargs)
        if op is None: result.append(copy.deepcopy(item));seen[item["name"]]=item;continue
        ref,attrs=arguments(op,args,kwargs,item["op"])
        require(type(ref) is Ref and cipher[ref.name],"static_transform_requires_encrypted_tensor")
        tensor=seen[ref.name]["tensor"]
        require(type(tensor) is dict and set(tensor)=={"policy","shape","dtype"} and
                tensor["policy"]==ALIAS_POLICY and tensor["dtype"] in ("float32","float64"),
                "static_transform_input_metadata")
        selected_plan=plan(op,tensor["shape"],attrs,resource_profile=resource_profile)
        require(item["tensor"]==dict(tensor,shape=selected_plan["output_shape"]),"static_transform_output_metadata")
        mapping={"input":ref.name};generated=[]
        for index,step in enumerate(selected_plan["steps"]):
            name=item["name"]+"_tensor_"+str(index)
            require(name not in used,"static_transform_generated_name_collision");used.add(name)
            refs=[Ref(mapping[r]) for r in step["inputs"]]
            if step["op"]=="reshape":target="torch.reshape";arguments_out=(refs[0],tuple(step["attrs"]["shape"]))
            elif step["op"]=="slice":
                a=step["attrs"];target="torch.narrow";arguments_out=(refs[0],a["axis"],a["start"],a["stop"]-a["start"])
            else:target="torch.cat";arguments_out=(tuple(refs),step["attrs"]["axis"])
            out=dict(name=name,op="call_function",target=target,args=encode(arguments_out),
                     kwargs=dict(kind="dict",items=[]),tensor=dict(tensor,shape=step["shape"]))
            result.append(out);generated.append(name);mapping[step["name"]]=name
        result.append(dict(name=item["name"],op="call_method",target="clone",
            args=encode((Ref(mapping[selected_plan["result"]]),)),kwargs=dict(kind="dict",items=[]),
            tensor=copy.deepcopy(item["tensor"])))
        entries.append(dict(source_node=item["name"],source_kind=item["op"],source_target=item["target"],
            input=ref.name,attributes=attrs,plan=selected_plan,generated_nodes=generated))
        seen[item["name"]]=item
        require(len(result)<=capacity,"static_transform_inventory_capacity")
    require(len(result)<=capacity,"static_transform_inventory_capacity")
    return result,entries

def check(record,inputs,*,resource_profile,expected_source_inventory=None):
    fields={"format","input_signature","resource_profile","source_inventory","normalized_inventory","entries",
            "source_fx_text","source_fx_sha256","normalized_fx_text","normalized_fx_sha256","public_state",
            "implementation","source_order_preserved","public_state_preserved","binding"}
    require(type(record) is dict and set(record)==fields and record["format"]==POLICY and
            record["binding"]==digest({k:v for k,v in record.items() if k!="binding"}),
            "static_transform_transport_binding")
    require(record["input_signature"]==inputs and record["resource_profile"]==resource_profile and
            record["implementation"]==implementation() and record["source_order_preserved"] is True and
            record["public_state_preserved"] is True,"static_transform_source_identity")
    require(len(canonical(record))<=resolve(resource_profile).json_bytes,"static_transform_record_capacity")
    if expected_source_inventory is not None:
        require(record["source_inventory"]==expected_source_inventory,"static_transform_source_bridge")
    for name in ("source","normalized"):
        text=record[name+"_fx_text"]
        require(type(text) is str and hashlib.sha256(text.encode()).hexdigest()==record[name+"_fx_sha256"],
                "static_transform_fx_binding")
    expected,entries=normalized_inventory(record["source_inventory"],resource_profile=resource_profile)
    require(record["normalized_inventory"]==expected and record["entries"]==entries,
            "static_transform_normalization_changed")
    return expected

def normalize(traced,inputs,*,resource_profile):
    import torch
    from pytorch_fused_matrix_normalization import inventory,_state
    from pytorch_arithmetic_normalization import clone_public_state
    require(isinstance(traced,torch.fx.GraphModule) and traced.meta.get("tensor_transform_source_policy")==POLICY,
            "static_transform_explicit_policy")
    source=inventory(traced);expected,entries=normalized_inventory(source,resource_profile=resource_profile)
    state=_state(traced);source_text=str(traced.graph);result=clone_public_state(traced)
    by_name={n.name:n for n in result.graph.nodes};expected_by_name={n["name"]:n for n in expected}
    for entry in entries:
        original=by_name[entry["source_node"]];mapping={"input":by_name[entry["input"]]}
        with result.graph.inserting_before(original):
            for step,name in zip(entry["plan"]["steps"],entry["generated_nodes"]):
                refs=[mapping[r] for r in step["inputs"]]
                if step["op"]=="reshape":target=torch.reshape;args=(refs[0],tuple(step["attrs"]["shape"]))
                elif step["op"]=="slice":
                    a=step["attrs"];target=torch.narrow;args=(refs[0],a["axis"],a["start"],a["stop"]-a["start"])
                else:target=torch.cat;args=(tuple(refs),step["attrs"]["axis"])
                node=result.graph.create_node("call_function",target,args,{},name=name)
                node.meta=copy.deepcopy(original.meta);node.meta["poseidon_static_tensor"]=copy.deepcopy(expected_by_name[name]["tensor"])
                mapping[step["name"]]=node;by_name[name]=node
            copy_node=result.graph.call_method("clone",(mapping[entry["plan"]["result"]],))
            copy_node.meta=copy.deepcopy(original.meta)
        original.replace_all_uses_with(copy_node);result.graph.erase_node(original)
        copy_node.name=entry["source_node"];by_name[copy_node.name]=copy_node
    result.graph.lint();result.recompile()
    require(inventory(result)==expected and inventory(traced)==source and _state(result)==_state(traced)==state,
            "static_transform_pure_normalization")
    normalized_text=str(result.graph)
    record=dict(format=POLICY,input_signature=copy.deepcopy(inputs),resource_profile=resource_profile,
        source_inventory=source,normalized_inventory=expected,entries=entries,public_state=state,
        source_fx_text=source_text,source_fx_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
        normalized_fx_text=normalized_text,normalized_fx_sha256=hashlib.sha256(normalized_text.encode()).hexdigest(),
        implementation=implementation(),source_order_preserved=True,public_state_preserved=True)
    record["binding"]=digest(record);check(record,inputs,resource_profile=resource_profile,expected_source_inventory=source)
    return result,record
