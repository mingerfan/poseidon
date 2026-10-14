"""Versioned metadata-only Tensor broadcast spellings before ordered capture."""
import copy, hashlib, math
from pathlib import Path
from benchmark_graph import require, digest, canonical
from model_resources import resolve

POLICY = "poseidon-pytorch-static-broadcast-v1"
SOURCE_FORMAT = "poseidon-trusted-pytorch-v28"
PACKAGE_FORMAT = "poseidon-trusted-pytorch-provenance-v28"
REFERENCE_FORMAT = "poseidon-trusted-pytorch-reference-v28"
CAPTURE_FORMAT = "poseidon-public-expression-source-capture-v5"
METHODS = frozenset(("expand_as", "broadcast_to"))

def arguments(method, args, kwargs):
    require(method in METHODS and 1 <= len(args) <= 2, "static_broadcast_arguments")
    key = "other" if method == "expand_as" else "size"
    require(type(kwargs) is dict and not (set(kwargs) - {key}), "static_broadcast_keywords")
    require(not (len(args) == 2 and key in kwargs), "static_broadcast_duplicate_argument")
    require(len(args) == 2 or key in kwargs, "static_broadcast_missing_argument")
    return args[0], args[1] if len(args) == 2 else kwargs[key]

def shape(value, *, resource_profile):
    limits = resolve(resource_profile)
    require(type(value) in (tuple, list) and len(value) <= limits.rank and
            all(type(n) is int and 0 < n <= limits.axis_size for n in value) and
            math.prod(value) <= limits.tensor_elements, "static_broadcast_shape_capacity")
    return list(value)

def plan(source, requested, *, resource_profile):
    source = shape(source, resource_profile=resource_profile)
    limits = resolve(resource_profile)
    require(type(requested) in (tuple, list) and len(source) <= len(requested) <= limits.rank and
            all(type(n) is int and n != 0 and n >= -1 for n in requested),
            "static_broadcast_static_size")
    offset = len(requested) - len(source); result = []
    for index, dimension in enumerate(requested):
        original = source[index - offset] if index >= offset else None
        if dimension == -1:
            require(original is not None, "static_broadcast_leading_minus_one")
            dimension = original
        require(original is None or original == 1 or original == dimension,
                "static_broadcast_dimension_mismatch")
        result.append(dimension)
    return shape(result, resource_profile=resource_profile)

def metadata(method, args, kwargs, *, resolve_value, resource_profile):
    import torch
    source, target = arguments(method, args, kwargs)
    source = resolve_value(source)
    require(isinstance(source, torch.Tensor), "static_broadcast_source_metadata")
    if method == "expand_as":
        target = resolve_value(target)
        require(isinstance(target, torch.Tensor), "static_broadcast_target_metadata")
        requested = list(target.shape)
    else:
        from torch.fx.immutable_collections import immutable_list
        require(type(target) in (tuple, list, torch.Size, immutable_list), "static_broadcast_static_size")
        requested = list(target)
    return source.expand(tuple(plan(list(source.shape), requested, resource_profile=resource_profile)))

def normalized_inventory(source, *, resource_profile):
    from pytorch_public_expression_flow import _decode, Ref
    limits = resolve(resource_profile)
    capacity = limits.expanded_nodes + limits.constants + limits.inputs + limits.outputs
    require(type(source) is list and 1 <= len(source) <= capacity, "static_broadcast_inventory_capacity")
    seen = {}; result = []; entries = []
    for item in source:
        require(type(item) is dict and set(item) == {"name","op","target","args","kwargs","tensor"} and
                type(item["name"]) is str and item["name"] not in seen, "static_broadcast_inventory_identity")
        names = {name: Ref(name) for name in seen}
        args = _decode(item["args"], names); kwargs = _decode(item["kwargs"], names)
        require(type(args) is tuple and type(kwargs) is dict, "static_broadcast_inventory_call")
        node = copy.deepcopy(item)
        if item["op"] == "call_method" and item["target"] in METHODS:
            left, right = arguments(item["target"], args, kwargs)
            require(type(left) is Ref, "static_broadcast_tensor_source")
            tensor = seen[left.name]["tensor"]
            require(type(tensor) is dict and set(tensor) == {"policy","shape","dtype"} and
                    tensor["policy"] == "signature-static-ordered-alias-staging-v1" and
                    tensor["dtype"] in ("float64","float32","int64","int32","bool"),
                    "static_broadcast_tensor_metadata")
            if item["target"] == "expand_as":
                require(type(right) is Ref and type(seen[right.name]["tensor"]) is dict,
                        "static_broadcast_target_metadata")
                requested = seen[right.name]["tensor"]["shape"]
            else:
                require(type(right) in (tuple, list), "static_broadcast_static_size")
                requested = list(right)
            selected = plan(tensor["shape"], requested, resource_profile=resource_profile)
            require(item["tensor"] == dict(policy=tensor["policy"],shape=selected,dtype=tensor["dtype"]),
                    "static_broadcast_result_metadata")
            node.update(target="expand",
                        args=dict(kind="tuple",items=[dict(kind="fx_value",name=left.name),
                             dict(kind="tuple",items=[dict(kind="int",value=n) for n in selected])]),
                        kwargs=dict(kind="dict",items=[]))
            entries.append(dict(source_node=item["name"],source_method=item["target"],input=left.name,
                                target=right.name if type(right) is Ref else None,source_shape=tensor["shape"],
                                requested_shape=list(requested),result_shape=selected,dtype=tensor["dtype"]))
        result.append(node); seen[item["name"]] = item
    return result, entries

def check(record, inputs, *, resource_profile, expected_source_inventory=None):
    fields = {"format","input_signature","resource_profile","source_inventory","normalized_inventory","entries",
              "source_fx_text","source_fx_sha256","normalized_fx_text","normalized_fx_sha256","public_state",
              "source_order_preserved","public_state_preserved","implementation_sha256","binding"}
    require(type(record) is dict and set(record) == fields and record["format"] == POLICY and
            record["binding"] == digest({k:v for k,v in record.items() if k != "binding"}),
            "static_broadcast_transport_binding")
    require(record["input_signature"] == inputs and record["resource_profile"] == resource_profile and
            record["source_order_preserved"] is True and record["public_state_preserved"] is True and
            record["implementation_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "static_broadcast_source_identity")
    require(len(canonical(record)) <= resolve(resource_profile).json_bytes, "static_broadcast_record_capacity")
    if expected_source_inventory is not None:
        require(record["source_inventory"] == expected_source_inventory, "static_broadcast_source_bridge")
    for name in ("source", "normalized"):
        text = record[name + "_fx_text"]
        require(type(text) is str and hashlib.sha256(text.encode()).hexdigest() == record[name + "_fx_sha256"],
                "static_broadcast_fx_text_binding")
    expected, entries = normalized_inventory(record["source_inventory"], resource_profile=resource_profile)
    require(record["normalized_inventory"] == expected and record["entries"] == entries,
            "static_broadcast_normalization_changed")
    return expected

def normalize(traced, inputs, *, resource_profile):
    import torch
    from pytorch_fused_matrix_normalization import inventory, _state
    from pytorch_arithmetic_normalization import clone_public_state
    require(isinstance(traced, torch.fx.GraphModule) and traced.meta.get("broadcast_source_policy") == POLICY,
            "static_broadcast_explicit_policy")
    source = inventory(traced); expected, entries = normalized_inventory(source, resource_profile=resource_profile)
    source_text = str(traced.graph); state = _state(traced); result = clone_public_state(traced)
    by_name = {node.name:node for node in result.graph.nodes}
    for entry in entries:
        node = by_name[entry["source_node"]]
        node.target = "expand"; node.args = (by_name[entry["input"]], tuple(entry["result_shape"])); node.kwargs = {}
    result.graph.lint(); result.recompile()
    require(inventory(result) == expected and inventory(traced) == source and _state(result) == _state(traced) == state,
            "static_broadcast_pure_normalization")
    normalized_text = str(result.graph)
    record = dict(format=POLICY,input_signature=copy.deepcopy(inputs),resource_profile=resource_profile,
                  source_inventory=source,normalized_inventory=inventory(result),entries=entries,
                  source_fx_text=source_text,source_fx_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
                  normalized_fx_text=normalized_text,normalized_fx_sha256=hashlib.sha256(normalized_text.encode()).hexdigest(),
                  public_state=state,source_order_preserved=True,public_state_preserved=True,
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record["binding"] = digest(record)
    check(record,inputs,resource_profile=resource_profile,expected_source_inventory=source)
    return result, record


def supports_source(spec):
    from pytorch_source_package_policy import SOURCE_FORMAT as package_format
    return type(spec) is dict and spec.get("format") in (SOURCE_FORMAT, package_format)
