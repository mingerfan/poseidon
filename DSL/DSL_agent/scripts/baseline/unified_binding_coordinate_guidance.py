"""Public coordinate frames selected from validated packing and graph semantics."""
import copy

VERSION = "explicit-v19"
PREVIOUS_VERSION = "explicit-v18"
ROLES = ("tensor_route", "pool_gather", "pool_stream_gather", "suffix_gather")
DIRECT = (
    "R_k(c)[j] = c[(j+k) mod P]. A direct registered gather, including G=0, "
    "uses destination-frame coefficients: output[j] receives vector[j] times "
    "source[(j+step) mod P]. Rotate the source to those coordinates before "
    "applying this destination mask. Rotating vector*source instead moves the "
    "mask support and changes which output slots are present. Apply every "
    "registered contribution to its own output chunk; addition combines them."
)
BSGS = (
    "For a registered baby_step=B and giant_step=G with G!=0, step=(B+G) mod P "
    "and the vector is in the baby frame. Destination j receives "
    "vector[(j+G) mod P]*source[(j+B+G) mod P]. Thus apply the final giant "
    "rotation to the whole masked baby contribution. The unchanged pre-rotated "
    "vector is not a destination mask."
)
FRAMES = (
    "The binding role determines the mask frame. A direct gather mask selects "
    "destination slots. A nonzero-G BSGS mask selects the baby frame. Linear "
    "tile fixed/moving masks select the input of their particular phase; its "
    "moving selection happens before the phase rotation. These rules refer to "
    "different bindings and are not interchangeable. Intermediate slot layouts "
    "are a program choice; final outputs obey the frozen selectors."
)
BROADCAST = (
    "For add, subtract and multiply, right-align logical axes and expand only "
    "singleton dimensions. At output coordinate q, each operand uses its own "
    "coordinate with singleton axes set to zero. Non-singleton row, batch and "
    "head coordinates remain separate. Padding and repeated periodic copies "
    "are not additional tensor elements. A common scalar replicated across "
    "all rows is not a general broadcast."
)
NAMES = (
    "Choose local names distinct from all public_constants keys, declared "
    "function names, enabled trusted helper names, hc, np, object, range and "
    "zero_ct. These names are reserved bindings, including short names such "
    "as p0 or p1 when present in public_constants. Assigning a ciphertext "
    "temporary to such a name does not turn it into a local variable."
)

def require(ok, message):
    if not ok:
        raise ValueError(message)

def route_frames(request):
    """Summarize actual registered roles; never infer masks from variable names."""
    period = request["layout"]["input_slot_period"]
    require(type(period) is int and period > 0, "Invalid slot period")
    groups = {}
    packing = request.get("constant_origins", {}).get("capacity_tensor_packing", {})
    for binding in packing.get("bindings", []):
        role = binding.get("role")
        if role not in ROLES:
            continue
        step = binding.get("step")
        require(type(step) is int and 0 <= step < period, "Invalid route step")
        has_baby = "baby_step" in binding
        has_giant = "giant_step" in binding
        require(has_baby == has_giant, "Incomplete route frame")
        frame = "destination"
        if has_baby:
            baby, giant = binding["baby_step"], binding["giant_step"]
            require(all(type(x) is int and 0 <= x < period for x in (baby, giant))
                    and step == (baby + giant) % period, "Invalid route frame")
            if giant:
                frame = "baby"
        key = (role, frame)
        groups[key] = groups.get(key, 0) + 1
    return [dict(role=role, mask_frame=frame, bindings=count)
            for (role, frame), count in sorted(groups.items())]

def broadcast_context(request):
    from benchmark_graph import validate
    shapes = validate(request["model"])["shapes"]
    rows = []
    for node in request["model"]["nodes"]:
        if node["op"] not in ("add", "subtract", "multiply"):
            continue
        output = list(shapes[node["outputs"][0]])
        inputs = [list(shapes[name]) for name in node["inputs"]]
        if any(shape != output for shape in inputs):
            rows.append(dict(node=node["id"], op=node["op"],
                             operand_shapes=inputs, output_shape=output))
    return dict(count=len(rows), examples=rows[:6], rule=BROADCAST) if rows else None

def specification(request, inherited):
    require(type(inherited) is dict and inherited.get("version") == PREVIOUS_VERSION,
            "Binding frames require explicit-v18")
    result = copy.deepcopy(inherited)
    result["version"] = VERSION
    frames = route_frames(request)
    if frames:
        rules = dict(version="registered-binding-frames-v1", roles=frames, distinction=FRAMES)
        if any(row["mask_frame"] == "destination" for row in frames):
            rules["destination"] = DIRECT
        if any(row["mask_frame"] == "baby" for row in frames):
            rules["baby"] = BSGS
        result["bound_route_coordinate_frames"] = rules
        # The new ledger supplies this same relation for every selected role.
        result.pop("bound_bsgs_coordinates", None)
    broadcasts = broadcast_context(request)
    if broadcasts:
        result["logical_arithmetic_broadcast"] = broadcasts
    result["reserved_binding_rule"] = NAMES
    return result

def coordinate_hint(request):
    frames = route_frames(request)
    parts = []
    if frames:
        if any(row["mask_frame"] == "destination" for row in frames):
            parts.append(DIRECT)
        if any(row["mask_frame"] == "baby" for row in frames):
            parts.append(BSGS)
        parts.append(FRAMES)
    if broadcast_context(request):
        parts.append(BROADCAST)
    return "\n".join(parts)

def repair_hint(request, source, diagnostic="", stage="static_check", *, witness=None):
    """Reuse bounded feedback; add only fixed public frame and binding rules."""
    if request.get("generation_guidance", {}).get("version") != VERSION:
        return ""
    from unified_graph_contract import validate_request, explicit_generation_guidance
    from public_weight_context import request_digest
    from unified_contract_scope_guidance import repair_hint as previous
    try:
        validate_request(request)
        legacy = dict(request, generation_guidance=explicit_generation_guidance(request, PREVIOUS_VERSION))
        legacy["request_id"] = request_digest(legacy)
        prior = previous(legacy, source, diagnostic, stage, witness=witness)
        parts = []
        if stage == "static_check":
            parts.append(NAMES)
        elif stage == "numerical_comparison":
            parts.append(coordinate_hint(request))
        if prior:
            parts.append(prior)
        return "\n".join(part for part in parts if part)[:6500]
    except (ValueError, TypeError, KeyError, IndexError, RecursionError, MemoryError):
        return ""
