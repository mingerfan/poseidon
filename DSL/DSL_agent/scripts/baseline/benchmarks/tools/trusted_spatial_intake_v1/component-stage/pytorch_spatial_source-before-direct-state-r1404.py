"""Agent-owned static spatial source rules for explicitly trusted FX capture.

This draft selects mathematical graph operations, before DSL generation.
Public source attributes and metadata determine shapes; tensor values never
determine control flow or output size.
"""
import math
from benchmark_graph import require
from model_resources import resolve
from operator_spatial_graph import resize_spec, transpose_spec

from pytorch_spatial_policy import POLICY

def arguments(args, kwargs, names, defaults, label):
    require(len(args) <= len(names) and not (set(kwargs) - set(names)), label + "_arguments")
    result = dict(zip(names, defaults))
    for name, value in zip(names, args):
        require(name not in kwargs, label + "_duplicate_argument")
        result[name] = value
    result.update(kwargs)
    return result

def static(value, resolve_value):
    import torch
    from torch.fx import Node, Proxy
    if isinstance(value, (Node, Proxy)):
        value = resolve_value(value)
    if isinstance(value, (tuple, list, torch.Size)):
        return [static(v, resolve_value) for v in value]
    require(value is None or type(value) in (int, float, bool, str), "spatial_static_attribute")
    require(type(value) is not float or math.isfinite(value), "spatial_finite_attribute")
    return value

def vector(value, rank, resolve_value, label):
    value = static(value, resolve_value)
    if type(value) in (int, float):
        value = [value] * rank
    require(type(value) is list and len(value) == rank, label)
    return value

def select(kind, target, args, kwargs, *, child=None, resolve_value=lambda value: value):
    import torch
    from torch.nn import functional as F
    if kind == "call_module" and type(child) in (
            torch.nn.Upsample, torch.nn.UpsamplingNearest2d, torch.nn.UpsamplingBilinear2d):
        values = arguments(args, kwargs, ("input",), (None,), "spatial_upsample")
        attrs = dict(size=child.size, scale_factor=child.scale_factor, mode=child.mode,
                     align_corners=child.align_corners,
                     recompute_scale_factor=child.recompute_scale_factor, antialias=False)
        return dict(operator="resize", operands=[values["input"]], raw_attrs=attrs,
                    source="module:" + type(child).__name__)
    if kind == "call_function" and target is F.interpolate:
        names = ("input", "size", "scale_factor", "mode", "align_corners", "recompute_scale_factor", "antialias")
        values = arguments(args, kwargs, names, (None, None, None, "nearest", None, None, False), "spatial_interpolate")
        return dict(operator="resize", operands=[values.pop("input")], raw_attrs=values,
                    source="function:torch.nn.functional.interpolate")
    if kind == "call_module" and type(child) in (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d):
        values = arguments(args, kwargs, ("input", "output_size"), (None, None), "spatial_transpose")
        rank = 1 if type(child) is torch.nn.ConvTranspose1d else 2
        return dict(operator="conv_transpose" + str(rank) + "d",
            operands=[values["input"], child.weight] + ([] if child.bias is None else [child.bias]),
            raw_attrs=dict(stride=child.stride, padding=child.padding, dilation=child.dilation,
                           output_padding=child.output_padding, groups=child.groups, output_size=values["output_size"]),
            source="module:" + type(child).__name__,
            state_sources=[None, str(target) + ".weight"] + ([] if child.bias is None else [str(target) + ".bias"]))
    if kind == "call_function" and target in (F.conv_transpose1d, F.conv_transpose2d):
        rank = 1 if target is F.conv_transpose1d else 2
        names = ("input", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation")
        values = arguments(args, kwargs, names, (None, None, None, 1, 0, 0, 1, 1), "spatial_transpose")
        return dict(operator="conv_transpose" + str(rank) + "d",
            operands=[values["input"], values["weight"]] + ([] if values["bias"] is None else [values["bias"]]),
            raw_attrs={key: values[key] for key in ("stride", "padding", "dilation", "output_padding", "groups")},
            source="function:torch.nn.functional.conv_transpose" + str(rank) + "d")
    return None

def selected_spec(selected, shape, *, resolve_value=lambda value: value, weight_shape=None, resource_profile):
    rank = len(shape) - 2 if selected["operator"] == "resize" else (1 if selected["operator"].endswith("1d") else 2)
    raw = selected["raw_attrs"]
    if selected["operator"] == "resize":
        attrs = {key: static(value, resolve_value) for key, value in raw.items()}
        for key in ("size", "scale_factor"):
            if attrs[key] is not None:
                attrs[key] = vector(attrs[key], rank, resolve_value, "spatial_resize_vector")
        target, _ = resize_spec(tuple(shape), attrs)
        output = tuple(shape[:2]) + tuple(target)
    else:
        attrs = {key: vector(raw[key], rank, resolve_value, "spatial_transpose_vector")
                 for key in ("stride", "padding", "dilation", "output_padding")}
        attrs["groups"] = static(raw["groups"], resolve_value)
        attrs["output_size"] = static(raw.get("output_size"), resolve_value)
        spec = transpose_spec(tuple(shape), tuple(weight_shape), attrs)
        output = (spec["source"][0], weight_shape[1] * spec["groups"], *spec["output_spatial"])
        if not spec["batched"]:
            output = output[1:]
    from benchmark_graph import dimensions
    dimensions(list(output), resource_profile=resource_profile)
    return attrs, tuple(output)

def metadata(kind, target, args, kwargs, *, child, resolve_value, resource_profile):
    import torch
    selected = select(kind, target, args, kwargs, child=child, resolve_value=resolve_value)
    if selected is None:
        return None
    tensor = resolve_value(selected["operands"][0])
    require(isinstance(tensor, torch.Tensor) and tensor.dtype is torch.float64
            and tensor.is_contiguous(), "spatial_cpu_float64_contiguous_metadata")
    weight_shape = None
    if selected["operator"] != "resize":
        weight = resolve_value(selected["operands"][1])
        require(isinstance(weight, torch.Tensor) and weight.dtype is torch.float64, "spatial_public_weight_metadata")
        weight_shape = tuple(weight.shape)
    attrs, output = selected_spec(selected, tuple(tensor.shape), resolve_value=resolve_value,
                                  weight_shape=weight_shape, resource_profile=resource_profile)
    return torch.empty(output, dtype=torch.float64, device="meta")

def emit_source(node, traced, *, emit, ref, constant, public_values, static_values, resource_profile):
    import torch
    child = traced.get_submodule(node.target) if node.op == "call_module" else None
    def resolve_value(value):
        if isinstance(value, torch.fx.Node):
            require(value in static_values, "spatial_data_dependent_attribute")
            return static_values[value]
        return value
    selected = select(node.op, node.target, node.args, node.kwargs, child=child, resolve_value=resolve_value)
    if selected is None:
        return None
    argument = selected["operands"][0]
    require(isinstance(argument, torch.fx.Node) and argument not in public_values, "spatial_cipher_input")
    info = argument.meta.get("tensor_meta")
    require(info is not None and info.dtype is torch.float64, "spatial_float64_input_metadata")
    expected = 1
    for size, stride in zip(reversed(info.shape), reversed(info.stride)):
        require(size == 1 or stride == expected, "spatial_contiguous_input")
        expected *= size
    operands = [ref(argument)]
    shapes = []
    for position, value in enumerate(selected["operands"][1:], 1):
        if isinstance(value, torch.fx.Node):
            require(value in public_values and isinstance(public_values[value], torch.Tensor),
                    "spatial_public_weight_required")
            tensor = public_values[value]
            operands.append(ref(value))
        else:
            require(isinstance(value, torch.Tensor) and value.dtype is torch.float64, "spatial_public_tensor")
            tensor = value
            source = selected.get("state_sources", [None] * len(selected["operands"]))[position]
            operands.append(constant(value, source))
        shapes.append(tuple(tensor.shape))
    attrs, output = selected_spec(selected, tuple(info.shape), resolve_value=resolve_value,
        weight_shape=shapes[0] if shapes else None, resource_profile=resource_profile)
    if len(shapes) > 1:
        co = output[1] if len(output) == (3 if selected["operator"].endswith("1d") else 4) else output[0]
        require(shapes[1] == (co,), "spatial_bias_shape")
    emit(node, selected["operator"], operands, attrs)
    return dict(fx_node=node.name, source=selected["source"], operator=selected["operator"],
                input_shape=list(info.shape), output_shape=list(output), attributes=attrs,
                fixed_semantics="Torch 2.0.1 CPU Float64 contiguous")

def check_record(record, emissions, descriptions, *, source_origins, resource_profile):
    """Bind every spatial source record to the typed, regenerated emission graph."""
    require(type(record)is dict and set(record)==
        {"version","entries","graph_format","original_function_accuracy_certified"},
        "spatial_source_record_fields")
    require(record["version"]==POLICY and record["graph_format"]=="poseidon-operator-graph-v4"
        and record["original_function_accuracy_certified"]is False,"spatial_source_record_identity")
    limits=resolve(resource_profile)
    entries=record["entries"]
    require(type(entries)is list and len(entries)<=limits.original_nodes,"spatial_source_record_budget")
    require(type(emissions)is dict and type(emissions.get("nodes"))is list
        and type(descriptions)is dict,"spatial_source_emission_inventory")
    operations={"resize","conv_transpose1d","conv_transpose2d"}
    selected={n["id"]:n for n in emissions["nodes"]if n["op"]in operations}
    require(len(selected)==sum(n["op"]in operations for n in emissions["nodes"])
        and len(entries)==len(selected),"spatial_source_record_coverage")
    sources={
        "resize":{"module:Upsample","module:UpsamplingNearest2d","module:UpsamplingBilinear2d",
                  "function:torch.nn.functional.interpolate"},
        "conv_transpose1d":{"module:ConvTranspose1d","function:torch.nn.functional.conv_transpose1d"},
        "conv_transpose2d":{"module:ConvTranspose2d","function:torch.nn.functional.conv_transpose2d"},
    }
    require(type(source_origins)is dict,"spatial_source_origin_inventory")
    seen=set()
    for entry in entries:
        require(type(entry)is dict and set(entry)==
            {"fx_node","source","operator","input_shape","output_shape","attributes","fixed_semantics"},
            "spatial_source_entry_fields")
        fx_name=entry["fx_node"]
        require(type(fx_name)is str,"spatial_source_fx_name")
        name="fx_"+fx_name
        require(name in selected and name not in seen,"spatial_source_entry_node")
        seen.add(name);node=selected[name];op=node["op"]
        require(entry["operator"]==op and entry["source"]in sources[op]
            and entry["fixed_semantics"]=="Torch 2.0.1 CPU Float64 contiguous",
            "spatial_source_entry_identity")
        require(source_origins.get(name)==dict(fx_node=fx_name,operation=op,action=None,
            fx_kind="call_module"if entry["source"].startswith("module:")else "call_function"),
            "spatial_source_entry_origin")
        require(entry["attributes"]==node["attrs"] and len(node["outputs"])==1
            and node["inputs"]and node["inputs"][0]in descriptions
            and node["outputs"][0]in descriptions,"spatial_source_entry_attributes")
        incoming=descriptions[node["inputs"][0]];outgoing=descriptions[node["outputs"][0]]
        require(incoming["kind"]=="cipher"and outgoing["kind"]=="cipher",
            "spatial_source_encrypted_dependency")
        require(entry["input_shape"]==list(incoming["shape"])
            and entry["output_shape"]==list(outgoing["shape"]),"spatial_source_entry_shapes")
    return record
