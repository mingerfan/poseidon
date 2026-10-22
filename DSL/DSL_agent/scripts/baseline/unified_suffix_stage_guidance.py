"""Public staged semantics for registered suffix reductions; no DSL answer."""
import copy, math
VERSION = "explicit-v20"
PREVIOUS_VERSION = "explicit-v19"
MAX_EXAMPLES = 12
WINDOW = (
    "A suffix_gather consumes a window aggregate, not the raw source tensor. "
    "For width W, its source-frame lane s represents the sum of the W source "
    "lanes starting at s. Only complete logical row starts are observable; "
    "other rolling-window lanes can mix rows and are not tensor outputs. "
    "The registered step and mask frame describe the subsequent gather of "
    "these aggregates. Applying a suffix_gather mask directly to the raw "
    "tensor omits W-1 members. No window crosses a ciphertext boundary."
)
NORMALIZATION = (
    "The suffix_gather mask already contains 1/W for mean and 1 for sum. "
    "Window aggregation is a sum; apply the registered normalization once. "
    "Gathered rows use compact output coordinates, not their original row-start "
    "positions. The vector key does not identify an input row."
)
BROADCAST = (
    "A compact reduction result has one logical value per preserved coordinate. "
    "Broadcasting back expands each value into its own row; shifting and adding "
    "the whole compact vector can mix different rows. A nonlinear or affine "
    "operation can make padding nonzero even when input padding was zero. "
    "Exclude padding before later gathers or broadcasts."
)
def require(ok, message):
    if not ok:
        raise ValueError(message)
def suffix_context(request):
    from benchmark_graph import validate
    graph = request["model"]; shapes = validate(graph)["shapes"]
    packed = request.get("constant_origins", {}).get("capacity_tensor_packing", {})
    windows = packed.get("suffix_windows", {})
    require(type(windows) is dict, "Suffix window inventory")
    if not windows:
        return None
    nodes = {n["id"]:n for n in graph["nodes"]}
    period = request["layout"]["input_slot_period"]
    require(type(period) is int and period > 0, "Suffix period")
    examples = []
    for identifier, window in sorted(windows.items()):
        require(identifier in nodes and type(window) is dict and set(window)=={"rows","width"},
                "Suffix node geometry")
        node = nodes[identifier]
        require(node["op"] in ("sum","mean") and len(node["inputs"])==len(node["outputs"])==1,
                "Suffix reduction operation")
        shape = list(shapes[node["inputs"][0]])
        output_shape = list(shapes[node["outputs"][0]])
        axes = sorted({a % len(shape) for a in node["attrs"]["axes"]})
        require(axes and axes == list(range(axes[0],len(shape))), "Suffix axes")
        width = math.prod(shape[a] for a in axes)
        rows = math.prod(shape)//width
        require(all(type(window[k]) is int for k in ("rows","width")) and
                window==dict(rows=rows,width=width) and width<=period and
                math.prod(output_shape)==rows and
                all((row*width)//period==(row*width+width-1)//period for row in range(rows)),
                "Complete suffix rows")
        bindings = [v for v in packed.get("bindings",[])
                    if v.get("role")=="suffix_gather" and v.get("node_id")==identifier]
        require(bindings and all(v.get("source")==node["inputs"][0] for v in bindings),
                "Suffix gather source")
        examples.append(dict(node=identifier,operator=node["op"],source=node["inputs"][0],
            output=node["outputs"][0],input_shape=shape,output_shape=output_shape,
            reduced_axes=axes,row_width=width,rows=rows,period=period,
            gather_source="complete_window_sum",gather_output="compact_rows",
            mask_normalization=1.0/width if node["op"]=="mean" else 1.0,
            divisor_already_in_masks=node["op"]=="mean",binding_count=len(bindings)))
    return dict(version="bound-suffix-stages-v1",count=len(examples),
                examples=examples[:MAX_EXAMPLES],window=WINDOW,normalization=NORMALIZATION,
                downstream_coordinates=BROADCAST)
def specification(request,inherited):
    require(type(inherited) is dict and inherited.get("version")==PREVIOUS_VERSION,
            "Suffix stages require explicit-v19")
    result=copy.deepcopy(inherited);result["version"]=VERSION
    context=suffix_context(request)
    if context is not None:
        result["bound_suffix_reduction_stages"]=context
    return result
def repair_hint(request,source,diagnostic="",stage="static_check",*,witness=None):
    if request.get("generation_guidance",{}).get("version")!=VERSION:
        return ""
    from unified_graph_contract import validate_request,explicit_generation_guidance
    from public_weight_context import request_digest
    from unified_binding_coordinate_guidance import repair_hint as previous
    try:
        validate_request(request)
        legacy=dict(request,generation_guidance=explicit_generation_guidance(request,PREVIOUS_VERSION))
        legacy["request_id"]=request_digest(legacy)
        prior=previous(legacy,source,diagnostic,stage,witness=witness)
        parts=[]
        if stage=="numerical_comparison" and suffix_context(request) is not None:
            parts.extend([WINDOW,NORMALIZATION,BROADCAST])
        if prior:
            parts.append(prior)
        return "\n".join(parts)[:6500]
    except (ValueError,TypeError,KeyError,IndexError,RecursionError,MemoryError):
        return ""
