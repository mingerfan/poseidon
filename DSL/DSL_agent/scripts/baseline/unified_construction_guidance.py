"""Versioned public construction costs and controlled repair facts; no DSL lowering."""
import ast
from collections import Counter
import json

VERSION="explicit-v13"
BINDINGS=(
    "Keep supplied public constants, zero_ct, runtime names and declared function/helper names unchanged. "
    "Use fresh local names for intermediate expressions and distinct public loop indices.")
WORK=(
    "Native static work starts each declared function at the number of supplied public constant bindings, "
    "including unused bindings. Each native helper invocation adds that helper's full expanded work plus one. "
    "Public range loops are expanded before this count; reusing a local result avoids repeating the call. "
    "Share existing intermediates or compose fewer native calls while preserving the complete logical graph. "
    "These conservative work units are not ciphertext operation counts, multiplicative depth or runtime memory.")
ROTATION=(
    "Call rotate only with one of the supplied admitted_steps. A zero displacement reuses the value. "
    "Compose admitted steps for other displacements while preserving the actual slot and chunk mapping; "
    "a chunk rotation alone does not implement a rotation of the complete logical tensor.")

def resources(request):
    from unified_graph_contract import validate_layout,GOLDEN_WORK_LIMIT
    from packed_input_abi import rotations
    from decorated_functions import LIMIT
    from native_public_loops import MAX_ITERATIONS,MAX_RANGE,MAX_NODES
    from unified_capacity_layout import EXTENDED_ABIS
    period=validate_layout(request["layout"]);layout=request["layout"]
    count=len(request["public_constants"])
    result=dict(version="public-construction-accounting-v1",
        contract="public" if "construction_profile" in request else "native",
        source_bytes=65536,source_ast_nodes=LIMIT,golden_work_limit=GOLDEN_WORK_LIMIT,
        supplied_constant_bindings=count,immutable_binding_rule=BINDINGS,
        rotation=dict(admitted_steps=list(rotations(period,capacity_layout=layout
            if layout["execution_abi"] in EXTENDED_ABIS else None)),rule=ROTATION))
    if "construction_profile" not in request:
        result["native"]=dict(function_initial_work=count,module_work_limit=LIMIT,
            helper_call_surcharge=1,range_trip_limit=MAX_RANGE,expanded_iteration_limit=MAX_ITERATIONS,
            expanded_ast_node_limit=MAX_NODES,rule=WORK)
    else:
        result["public"]=dict(interpretation_step_limit=4096,emitted_cipher_operation_limit=256,
            rule="Public helpers use the selected bounded AST normalizer rather than native Func calls. "
                 "The native per-helper registry formula does not describe those public helper calls.")
    return result

def specification(request):
    from unified_tensor_axis_guidance import specification as previous,MAX_BYTES
    context=previous(request);context["version"]=VERSION
    context["construction_resources"]=resources(request)
    lists=[context["tensor_axis_annotations"][k] for k in
        ("transforms","linears","broadcasts","input_chunks","reductions")]
    lists.extend(context["shape_examples"][k] for k in ("reductions","broadcasts"))
    lists.extend((context["logical_constant_scalar_bindings"]["bindings"],context["constant_role_examples"]["bindings"]))
    while len(json.dumps(context,sort_keys=True).encode())>MAX_BYTES:
        candidates=[row for row in lists if row]
        if not candidates:raise ValueError("Construction required guidance byte bound")
        max(candidates,key=lambda row:len(json.dumps(row,sort_keys=True))).pop()
    return context

def cost_facts(request,source,witness=None):
    """Recompute trusted static counts; never execute or modify candidate code."""
    if "construction_profile" in request:return None
    if type(source) is not str or len(source.encode())>65536:return None
    try:
        tree=ast.parse(source)
        if sum(1 for _ in ast.walk(tree))>4096:return None
        from unified_graph_contract import validate_request,_native_candidate_plan,GOLDEN_WORK_LIMIT
        period=validate_request(request)
        plan=_native_candidate_plan(source,request,period)
        functions=plan["functions"];golden=functions["golden"];observed=golden["expanded_cost"]
        if observed<=GOLDEN_WORK_LIMIT:return None
        declarations={n.name:n for n in tree.body if type(n) is ast.FunctionDef}
        entry=declarations["golden"];location=dict(kind="source",line=entry.lineno,column=entry.col_offset)
        if witness is not None:
            from application_diagnostics import checked
            diagnostic=checked(witness)
            if (diagnostic["reason_code"]!="expanded_work_budget" or diagnostic["location"]!=location
                or diagnostic["context"]!=dict(limit=GOLDEN_WORK_LIMIT,observed=observed)):return None
        counts=Counter(golden["calls"]);costs=[]
        for name,count in counts.items():
            node=declarations[name];per_call=functions[name]["expanded_cost"]+1
            costs.append(dict(location=dict(kind="source",line=node.lineno,column=node.col_offset),
                calls=count,per_call_work=per_call,total_work=count*per_call))
        costs.sort(key=lambda row:(-row["total_work"],row["location"]["line"],row["location"]["column"]))
        bindings=len(request["public_constants"])
        return dict(format="native-static-work-facts-v1",observed=observed,limit=GOLDEN_WORK_LIMIT,
            supplied_constant_bindings=bindings,direct_native_helper_calls=sum(counts.values()),
            direct_native_call_work=sum(x["total_work"] for x in costs),
            direct_registry_work=bindings*sum(counts.values()),largest_calls=costs[:3],
            acceptance_granted=False,candidate_executed=False)
    except (SyntaxError,ValueError,TypeError,KeyError,IndexError,RecursionError,MemoryError):
        return None

def cost_text(value):
    if value is None:return ""
    text=("Static work accounting: "+str(value["observed"])+" > "+str(value["limit"])+
        "; "+str(value["supplied_constant_bindings"])+" supplied constant bindings; "+
        str(value["direct_native_helper_calls"])+" direct native helper invocations add "+
        str(value["direct_native_call_work"])+" units, including "+
        str(value["direct_registry_work"])+" units of repeated registry work.")
    for row in value["largest_calls"]:
        text+=(" Helper at line "+str(row["location"]["line"])+": "+str(row["calls"])+" calls x "+
            str(row["per_call_work"])+" units = "+str(row["total_work"])+".")
    return text+" "+WORK

def repair_hint(request,source,diagnostic="",stage="static_check",*,witness=None):
    if request.get("generation_guidance",{}).get("version")!=VERSION:return ""
    if type(source) is not str or len(source.encode())>65536:return ""
    try:
        tree=ast.parse(source)
        if sum(1 for _ in ast.walk(tree))>4096:return ""
        from unified_graph_contract import validate_request
        validate_request(request);constraints=resources(request)
    except (SyntaxError,ValueError,TypeError,KeyError,IndexError,RecursionError,MemoryError):
        return ""
    hints=[]
    if stage=="static_check":
        facts=cost_facts(request,source,witness)
        if facts is not None:hints.append(cost_text(facts))
        hints.extend((BINDINGS,"Admitted rotation steps: "+json.dumps(constraints["rotation"]["admitted_steps"])+". "+ROTATION))
    from unified_tensor_axis_guidance import repair_hint as previous
    view=dict(request,generation_guidance=dict(request["generation_guidance"],version="explicit-v12"))
    prior=previous(view,source,stage=stage)
    if prior:hints.append(prior)
    return "\n".join(dict.fromkeys(hints))[:6500]
