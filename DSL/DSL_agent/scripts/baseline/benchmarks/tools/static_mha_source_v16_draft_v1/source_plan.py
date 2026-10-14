"""v16 static trusted-source MHA draft; does not register a production source format.

Only native eval MHA calls, public masks, static tuple selection and output
containers are admitted. No shape propagation on sample inputs, backend or
Provider is used. The frozen r1063 algebra and the current math stack are reused.
"""
import copy
import hashlib
import importlib.util
import json
import math
import operator
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from output_contract import SOURCE_FORMAT, digest, flatten, validate as validate_output

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
ROOT = (Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
OLD = BASE / "benchmarks/tools/static_mha_source_draft_v1"
FORMAT = "poseidon-static-mha-source-v16-draft-v1"
REQUEST_FORMAT = "poseidon-static-mha-source-request-v16-draft-v1"
OWNED = ("source_plan.py", "output_contract.py", "fixtures.py", "test_controls.py",
         "test_math.py", "audit.py")
LEGACY_SHA = {
    "mha_source_plan.py": "f1f845815f671091a71b9b86217c5b68644d704b736ed7a237f638dec6164f49",
    "mha_source_reference.py": "25bdaf114101b57fe3b3f8b78f6fd2eb4e77e97f0e357851aef4e86b849817c5",
}
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
REQUEST_FIELDS = {"format", "source_format", "inputs", "outputs", "approximation",
                  "resource_profile", "source_fx_text", "source_fx_code", "public_state",
                  "calls", "output_values"}

def require(value, message):
    if not value:
        raise ValueError(message)

def source_identity():
    paths = [HERE / n for n in OWNED]
    paths += [OLD / n for n in LEGACY_SHA]
    paths += [BASE / n for n in ("benchmark_graph.py", "model_decomposition.py",
               "model_resources.py", "benchmark_math.py", "benchmark_torch.py",
               "pytorch_model_intake.py", "pytorch_ordered_contract.py",
               "platform_config.py", "platform-profiles.json")]
    result = {}
    for p in paths:
        require(p.is_file() and not p.is_symlink(), "source_identity_file")
        result[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for n, want in LEGACY_SHA.items():
        require(hashlib.sha256((OLD/n).read_bytes()).hexdigest() == want, "frozen_mha_math_changed")
    return result

def _load_legacy(name):
    require(name in LEGACY_SHA, "fixed_math_module")
    p = OLD/name
    require(hashlib.sha256(p.read_bytes()).hexdigest() == LEGACY_SHA[name], "frozen_mha_math_changed")
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))
    spec = importlib.util.spec_from_file_location("_v16_" + p.stem, p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def check_approximation(value):
    require(type(value) is dict and set(value) == {"exp", "reciprocal"} and
            type(value["exp"]) is dict and type(value["reciprocal"]) is dict and
            bool(value["exp"]) and bool(value["reciprocal"]), "explicit_approximation")

def _shape(value):
    require(type(value) is list and 2 <= len(value) <= 3 and
            all(type(x) is int and 0 < x <= 1024 for x in value) and
            math.prod(value) <= 1048576, "static_cipher_signature")
    return tuple(value)

def check_request_header(request):
    require(type(request) is dict and set(request) == REQUEST_FIELDS, "source_request_fields")
    require(request["format"] == REQUEST_FORMAT and request["source_format"] == SOURCE_FORMAT,
            "explicit_v16_source")
    check_approximation(request["approximation"])
    require(request["resource_profile"] == "general-tensors-v1", "draft_resource_profile")
    ins, outs = request["inputs"], request["outputs"]
    require(type(ins) is list and 1 <= len(ins) <= 4 and type(outs) is list and
            1 <= len(outs) <= 4 and len(set(outs)) == len(outs) and
            all(type(x) is str and NAME.fullmatch(x) for x in outs), "source_signature")
    names = []
    for item in ins:
        require(type(item) is dict and set(item) == {"name", "shape"} and
                type(item["name"]) is str and NAME.fullmatch(item["name"]), "source_input")
        _shape(item["shape"])
        names.append(item["name"])
    require(len(set(names)) == len(names), "source_input_unique")
    require(type(request["calls"]) is list and 1 <= len(request["calls"]) <= 32,
            "registered_mha_calls")
    require(type(request["public_state"]) is dict and len(request["public_state"]) <= 128,
            "public_state_inventory")
    require(all(type(request[k]) is str and len(request[k].encode()) <= 65536
                for k in ("source_fx_text", "source_fx_code")), "source_fx_budget")
    require(len(json.dumps(request, allow_nan=False).encode()) <= 4*1024**2,
            "source_request_budget")
    return request

@dataclass(frozen=True)
class TensorRef:
    name: str
    shape: tuple

def _encode(value):
    if isinstance(value, TensorRef):
        return dict(kind="tensor", value=value.name)
    if value is None:
        return dict(kind="none")
    if type(value) in (tuple, list):
        return dict(kind="tuple" if type(value) is tuple else "list", items=[_encode(x) for x in value])
    if type(value) is dict:
        return dict(kind="dict", items=[dict(key=k, value=_encode(v)) for k,v in value.items()])
    raise ValueError("dynamic_output_rejected")

def _decode(value, descriptors, depth=0):
    require(type(value) is dict and depth <= 8, "source_output_node")
    kind = value.get("kind")
    if kind == "tensor":
        require(set(value) == {"kind", "value"} and value["value"] in descriptors, "source_output_tensor")
        return TensorRef(value["value"], descriptors[value["value"]])
    if kind == "none":
        require(set(value) == {"kind"}, "source_output_none")
        return None
    require(kind in ("tuple", "list", "dict") and set(value) == {"kind", "items"} and
            type(value["items"]) is list and 1 <= len(value["items"]) <= 4, "source_output_container")
    if kind == "dict":
        pairs = value["items"]
        require(all(type(p) is dict and set(p) == {"key", "value"} and
                    type(p["key"]) is str and 0 < len(p["key"]) <= 128 for p in pairs) and
                len({p["key"] for p in pairs}) == len(pairs), "source_output_keys")
        return {p["key"]: _decode(p["value"], descriptors, depth+1) for p in pairs}
    items = [_decode(x, descriptors, depth+1) for x in value["items"]]
    return tuple(items) if kind == "tuple" else items

def _state_record(value):
    import numpy as np
    array = np.asarray(value)
    require(array.dtype.str in ("<f8", "|b1") and array.size <= 1048576 and
            np.isfinite(array).all() and (np.abs(array.astype(float)) <= 1024).all(), "finite_public_state")
    return dict(dtype=array.dtype.str, shape=list(array.shape),
                bytes_sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                values=array.tolist())

def _state_array(record):
    import numpy as np
    require(type(record) is dict and set(record) == {"dtype", "shape", "bytes_sha256", "values"} and
            record["dtype"] in ("<f8", "|b1"), "public_state_record")
    value = np.asarray(record["values"], dtype=record["dtype"])
    require(list(value.shape) == record["shape"] and _state_record(value) == record, "public_state_bytes")
    return value

def _module_state(call, public_state):
    import numpy as np
    require(type(call) is dict and set(call) == {"id", "target", "module", "operands",
            "options", "mask_sources"} and type(call["id"]) is str and NAME.fullmatch(call["id"]) and
            type(call["target"]) is str and all(NAME.fullmatch(x) for x in call["target"].split(".")),
            "mha_call_fields")
    params = call["module"]["parameters"]
    for name, value in params.items():
        target = call["target"] + "." + name
        if value is None:
            require(target not in public_state, "absent_parameter_state")
        else:
            require(target in public_state, "original_parameter_state")
            raw = _state_array(public_state[target])
            require(raw.dtype.str == "<f8" and np.array_equal(raw, np.asarray(value, dtype="<f8")),
                    "original_parameter_bytes")
    require(set(call["mask_sources"]) == {"attn_mask", "key_padding_mask"}, "mask_source_fields")
    for option, target in call["mask_sources"].items():
        value = call["options"][option]
        if target is None:
            require(value is None, "mask_source_missing")
        else:
            require(target in public_state, "original_mask_state")
            raw = _state_array(public_state[target])
            expected = dict(dtype="bool" if raw.dtype.str == "|b1" else "float64",
                            values=raw.tolist(), negative_infinity=None)
            require(value == expected, "mask_source_bytes")

def _output_shapes(call):
    c = call["module"]["config"]
    q = call["operands"]["query"]["shape"]
    k = call["operands"]["key"]["shape"]
    unbatched = len(q) == 2
    n = 1 if unbatched else q[0 if c["batch_first"] else 1]
    l = q[0 if unbatched or not c["batch_first"] else 1]
    s = k[0 if unbatched or not c["batch_first"] else 1]
    total = s + int(call["module"]["parameters"]["bias_k"] is not None) + int(c["add_zero_attn"])
    result = {"attn_output": tuple(q)}
    if call["options"]["need_weights"]:
        result["attn_weights"] = tuple(([l,total] if call["options"]["average_attn_weights"] else
                  [c["num_heads"],l,total]) if unbatched else
                 ([n,l,total] if call["options"]["average_attn_weights"] else
                  [n,c["num_heads"],l,total]))
    return result

def _slice_origins(call, mini, remap, public_state):
    import numpy as np
    c = call["module"]["config"]; e = c["embed_dim"]; params = call["module"]["parameters"]
    linears = [n for n in mini["mathematical_graph"]["nodes"] if n["op"] == "linear"]
    require(len(linears) == 4, "mha_projection_nodes")
    records = []
    for index, role in enumerate(("query", "key", "value", "output")):
        node = linears[index]
        weight = "in_proj_weight" if index < 3 and c["same_projection"] else (
            ("q_proj_weight", "k_proj_weight", "v_proj_weight")[index] if index < 3 else "out_proj.weight")
        for operand, parameter in ((1,weight), (2,"in_proj_bias" if index < 3 else "out_proj.bias")):
            if operand >= len(node["inputs"]):
                require(params[parameter] is None, "projection_bias_presence")
                continue
            target = call["target"] + "." + parameter
            raw = _state_array(public_state[target])
            sliced = index < 3 and parameter in ("in_proj_weight", "in_proj_bias")
            start, stop = (index*e, (index+1)*e) if sliced else (0, raw.shape[0])
            expected = raw[start:stop] if sliced else raw
            constant = node["inputs"][operand]
            actual = np.asarray(mini["mathematical_graph"]["constants"][constant], dtype="<f8")
            require(np.array_equal(actual, expected), "projection_slice_bytes")
            records.append(dict(role=role, parameter=parameter, source=target,
                source_shape=list(raw.shape), source_dtype=raw.dtype.str,
                source_bytes_sha256=public_state[target]["bytes_sha256"],
                slice=dict(axis=0,start=start,stop=stop) if sliced else None,
                result_shape=list(expected.shape),
                result_bytes_sha256=hashlib.sha256(expected.tobytes(order="C")).hexdigest(),
                mathematical_constant=remap[constant]))
    return records

def build_request(request):
    request = copy.deepcopy(check_request_header(request))
    legacy = _load_legacy("mha_source_plan.py")
    from model_decomposition import decompose, FORMAT as GRAPH_FORMAT
    from model_resources import identity
    graph = dict(format=GRAPH_FORMAT, id="trusted_mha_v16_draft", inputs=copy.deepcopy(request["inputs"]),
                 constants={}, nodes=[], outputs=[])
    descriptors = {x["name"]: _shape(x["shape"]) for x in request["inputs"]}
    values = {name:name for name in descriptors}
    used = set(values); entries=[]; call_ids=set()
    for index, call in enumerate(request["calls"]):
        _module_state(call, request["public_state"])
        require(call["id"] not in call_ids, "source_call_unique");call_ids.add(call["id"])
        for role, item in call["operands"].items():
            require(item["name"] in descriptors and tuple(item["shape"]) == descriptors[item["name"]],
                    "source_operand_binding")
        mini = legacy.plan(call["module"],call["operands"],call["options"],request["approximation"],
                           resource_profile=request["resource_profile"])
        remap = {x["name"]: values[x["name"]] for x in mini["mathematical_graph"]["inputs"]}
        def fresh(local):
            base = "mha" + str(index) + "__" + local; result=base;extra=0
            while result in used:
                extra+=1;result=base+"__"+str(extra)
            used.add(result);return result
        for name, value in mini["mathematical_graph"]["constants"].items():
            remap[name] = fresh(name);graph["constants"][remap[name]] = copy.deepcopy(value)
        for node in mini["mathematical_graph"]["nodes"]:
            out = fresh(node["outputs"][0]);remap[node["outputs"][0]]=out
            graph["nodes"].append(dict(id=fresh(node["id"]),op=node["op"],
                inputs=[remap[x] for x in node["inputs"]],attrs=copy.deepcopy(node["attrs"]),outputs=[out]))
        shapes = _output_shapes(call);refs={}
        for output in mini["mathematical_graph"]["outputs"]:
            symbolic=call["id"]+"__"+output["name"]
            require(symbolic not in descriptors, "source_output_collision")
            descriptors[symbolic]=shapes[output["name"]]
            values[symbolic]=remap[output["value"]];refs[output["name"]]=symbolic
        entries.append(dict(source_node=call["id"],target=call["target"],
            source_request_sha256=mini["source_request_sha256"],
            projections=_slice_origins(call,mini,remap,request["public_state"]),
            mask_direction="MHA True blocks; effective_allow=not blocked",
            mask_sources=copy.deepcopy(call["mask_sources"]),mask_dispatch=mini["mask_dispatch"],
            effective_allow_sha256=digest(mini["effective_allow"]),
            effective_additive_sha256=digest(mini["effective_additive"]),
            outputs=refs,output_shapes={k:list(v) for k,v in shapes.items()}))
    result_value = _decode(request["output_values"], descriptors)
    leaves, tree = flatten(result_value,lambda x:isinstance(x,TensorRef),request["outputs"],
                           source_format=SOURCE_FORMAT)
    graph["outputs"]=[dict(name=name,value=values[leaf.name]) for name,leaf in zip(request["outputs"],leaves)]
    lowered, decomposition = decompose(graph,resource_profile=request["resource_profile"])
    result=dict(format=FORMAT,source_format=SOURCE_FORMAT,source_request=request,
        source_request_sha256=digest(request),implementation=source_identity(),
        mathematical_graph=graph,lowered_graph=lowered,decomposition=decomposition,
        output_record=tree,normalization=dict(calls=entries,source_fx_sha256=hashlib.sha256(
            request["source_fx_text"].encode()).hexdigest()),resource_profile=identity(request["resource_profile"]),
        production_intake_integrated=False,compiler_validated=False,encrypted_execution_validated=False,
        agent_generated=False,original_function_accuracy_certified=False)
    result["binding"]=digest(result)
    return result

def verify(plan, *, expected_source_request_sha256):
    require(type(plan) is dict and plan.get("format") == FORMAT and
            plan.get("source_request_sha256") == expected_source_request_sha256,
            "external_source_binding")
    require(plan.get("binding") == digest({k:v for k,v in plan.items() if k!="binding"}), "plan_binding")
    require(build_request(plan["source_request"]) == plan, "plan_semantics_or_source_identity")
    return plan

def capture(module, inputs, outputs, *, approximation, trusted=False):
    import torch
    require(trusted is True and isinstance(module,torch.nn.Module) and module.training is False,
            "explicit_trusted_eval_source")
    for child in module.modules():
        require(not child._forward_hooks and not child._forward_pre_hooks, "source_hooks_rejected")
        if isinstance(child,torch.nn.MultiheadAttention):
            require(type(child) is torch.nn.MultiheadAttention and child.training is False,
                    "native_eval_mha_only")
    legacy = _load_legacy("mha_source_plan.py")
    source = copy.deepcopy(module)
    def state():
        result={}
        for name, value in source.state_dict().items():
            require(value.device.type=="cpu" and value.dtype in (torch.float64,torch.bool),
                    "strict_cpu_float64_or_boolean_state")
            result[name]=_state_record(value.detach().numpy())
        return result
    before=state();rng=torch.get_rng_state().clone()
    try:
        fx=torch.fx.symbolic_trace(source)
    except (torch.fx.proxy.TraceError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("dynamic_or_unregistered_source") from error
    require(state()==before and torch.equal(rng,torch.get_rng_state()), "source_trace_mutation")
    ports=[n for n in fx.graph.nodes if n.op=="placeholder"]
    require(len(ports)==len(inputs),"source_input_count")
    env={node:TensorRef(item["name"],_shape(item["shape"])) for node,item in zip(ports,inputs)}
    attrs={};calls=[];result=None
    def bound(value):
        if isinstance(value,torch.fx.Node):
            require(value in env,"source_operand_available");return env[value]
        if type(value) in (tuple,list):return type(value)(bound(x) for x in value)
        if type(value) is dict:return {k:bound(x) for k,x in value.items()}
        if value is None or type(value) in (bool,int):return value
        raise ValueError("dynamic_or_unregistered_argument")
    for node in fx.graph.nodes:
        if node.op=="placeholder":continue
        if node.op=="get_attr":
            value=source
            for part in str(node.target).split("."):value=getattr(value,part)
            require(str(node.target) in before and isinstance(value,torch.Tensor),"public_mask_attribute")
            env[node]=legacy.public_mask(value);attrs[node]=str(node.target)
        elif node.op=="call_module":
            child=fx.get_submodule(node.target)
            require(type(child) is torch.nn.MultiheadAttention,"registered_mha_calls_only")
            names=("query","key","value","key_padding_mask","need_weights","attn_mask","average_attn_weights","is_causal")
            require(len(node.args)<=8 and not(set(node.kwargs)-set(names)) and
                    not(set(names[:len(node.args)])&set(node.kwargs)), "mha_call_signature")
            args=dict(zip(names[3:],(None,True,None,True,False)))
            args.update(zip(names,node.args));args.update(node.kwargs)
            require(all(k in args for k in names),"mha_call_signature")
            operands={}
            for role in names[:3]:
                item=bound(args[role]);require(isinstance(item,TensorRef),"mha_cipher_operand")
                operands[role]=dict(kind="cipher",name=item.name,shape=list(item.shape),dtype="float64")
            options={name:bound(args[name]) for name in names[3:]}
            mask_sources={name:attrs.get(args[name]) if isinstance(args[name],torch.fx.Node) else None
                          for name in ("attn_mask","key_padding_mask")}
            require(all(args[name] is None or mask_sources[name] is not None
                        for name in mask_sources),"public_mask_state_only")
            call=dict(id=node.name,target=str(node.target),module=legacy.snapshot_module(child),
                      operands=operands,options=options,mask_sources=mask_sources)
            legacy.plan(call["module"],operands,options,approximation)
            shapes=_output_shapes(call)
            env[node]=(TensorRef(node.name+"__attn_output",shapes["attn_output"]),
                       TensorRef(node.name+"__attn_weights",shapes["attn_weights"])
                       if options["need_weights"] else None)
            calls.append(call)
        elif node.op=="call_function" and node.target is operator.getitem:
            require(len(node.args)==2 and not node.kwargs and type(node.args[1]) is int,
                    "static_tuple_selection")
            value=bound(node.args[0]);require(type(value) in (tuple,list) and
                                             0<=node.args[1]<len(value),"static_tuple_selection")
            env[node]=value[node.args[1]]
        elif node.op=="output":
            result=bound(node.args[0])
        else:
            raise ValueError("dynamic_or_unregistered_source_operation")
    require(result is not None and state()==before and torch.equal(rng,torch.get_rng_state()),
            "source_capture_purity")
    request=dict(format=REQUEST_FORMAT,source_format=SOURCE_FORMAT,inputs=copy.deepcopy(inputs),
        outputs=copy.deepcopy(outputs),approximation=copy.deepcopy(approximation),
        resource_profile="general-tensors-v1",source_fx_text=str(fx.graph),source_fx_code=fx.code,
        public_state=before,calls=calls,output_values=_encode(result))
    plan=build_request(request)
    return plan,dict(original_module=source,source_fx=fx)

def source_reference(plan, inputs, *, approximate):
    """Independent sequential native-MHA formula; does not evaluate the graph."""
    reference=_load_legacy("mha_source_reference.py")
    request=plan["source_request"];values=dict(inputs)
    for call in request["calls"]:
        sub=dict(module=call["module"],operands=call["operands"],options=call["options"],
                 approximation=request["approximation"],resource_profile=request["resource_profile"])
        result=reference.evaluate(sub,values,approximate=approximate)
        for name,value in result.items():values[call["id"]+"__"+name]=value
    shapes={name:tuple(value.shape) for name,value in values.items()}
    structure=_decode(request["output_values"],shapes)
    leaves,record=flatten(structure,lambda x:isinstance(x,TensorRef),request["outputs"],
                          source_format=SOURCE_FORMAT)
    require(record==plan["output_record"],"reference_output_structure")
    return {name:values[leaf.name] for name,leaf in zip(request["outputs"],leaves)}
