"""Public binding/shape guidance draft, staged separately from frozen requests."""
import ast
import copy
import json
import math

VERSION="explicit-v11"
BINDINGS=(
    "Only names in public_constants or the explicitly admitted public-weight interface "
    "are bound DSL globals. Names under model.constants are logical graph references, "
    "not implicit Python variables. Respect the actual bound scalar/vector contents and "
    "the exact logical-to-public scalar matches; p-number order has no semantic role. "
    "A single-lane packing mask or a masked reduction coefficient is not an all-lane "
    "gain, zero bias or replicated scalar. A logical public tensor still needs its "
    "declared coordinate mapping; a uniform tensor does not turn matmul into scalar multiplication.")
SHAPES=(
    "A reduction produces exactly the declared reduced shape. Intermediate slot representation "
    "is your choice, but every later broadcast must use that same representation. A prefix scalar "
    "and a replicated scalar differ. Broadcast every declared dimension, including row/channel "
    "axes and across ciphertext chunks; exclude padding. Sum and mean differ by the full "
    "logical reduction count. Final outputs follow the frozen named C-order bindings.")

def logical_scalar_bindings(request,checked):
    """Exact public data matches, never constant-index guesses or DSL lowering."""
    public={name:value for name,value in request["public_constants"].items()
            if type(value) in (int,float)}
    examples=[];inspected=unclassified=0
    for name,value in sorted(request["model"]["constants"].items())[:96]:
        inspected+=1
        pending=[value];scalars=[];visits=0;complete=True
        while pending:
            item=pending.pop();visits+=1
            if visits>256:
                complete=False;break
            if type(item) in (int,float) and math.isfinite(item):
                scalars.append(item)
            elif type(item) is list:
                if len(item)+len(pending)+visits>256:
                    complete=False;break
                pending.extend(item)
            else:
                complete=False;break
        if not complete or not scalars:
            unclassified+=1;continue
        if all(v==scalars[0] for v in scalars):
            names=sorted(n for n,v in public.items() if v==scalars[0])
            examples.append(dict(logical_name=name,logical_shape=checked["shapes"][name],
                kind="scalar" if type(value) in (int,float) else "uniform_tensor",
                bound_scalar_names=names[:8],matching_scalar_count=len(names)))
    return dict(bindings=examples[:32],matching_logical_count=len(examples),
        inspected_logical_count=inspected,logical_count=len(request["model"]["constants"]),
        unclassified_count=unclassified,
        scope="Exact equality to already supplied public scalars. Uniform tensor matches "
              "apply to its elementwise scalar value only; preserve graph shape and "
              "matrix/linear operator semantics. This is not a ciphertext lowering plan.")

def specification(request):
    from benchmark_graph import validate, require
    from unified_capacity_layout import ABIS
    from unified_poseidon_guidance import specification as previous, REDUCTION, BROADCAST
    require(request.get("compiler_configuration",{}).get("backend")=="poseidon_cpu",
            "Poseidon binding guidance requires an explicit Poseidon CPU configuration")
    layout=request["layout"]
    require(layout["execution_abi"] in ("unified-periodic-inputs-v1",*ABIS),
            "Binding guidance requires a configured packing ABI")
    model=request["model"]
    checked=validate(model)
    context=copy.deepcopy(previous(request))
    groups=context["rules"]
    reductions=[]
    broadcasts=[]
    for node in model["nodes"]:
        op=node["op"]
        if op in ("mean","sum"):
            reductions.append(dict(node=node["id"],operator=op,input=node["inputs"][0],
                input_shape=checked["shapes"][node["inputs"][0]],
                output=node["outputs"][0],output_shape=checked["shapes"][node["outputs"][0]],
                axes=list(node["attrs"]["axes"]),keepdims=node["attrs"]["keepdims"]))
            groups["packed_reduction"]=REDUCTION
        if op in ("add","subtract","multiply"):
            shape=checked["shapes"][node["outputs"][0]]
            expanded=[dict(value=name,kind=checked["kinds"][name],shape=checked["shapes"][name])
                      for name in node["inputs"] if checked["shapes"][name]!=shape]
            if expanded:
                broadcasts.append(dict(node=node["id"],operator=op,expanded_operands=expanded,
                    output=node["outputs"][0],output_shape=shape))
                groups["tensor_broadcast"]=BROADCAST
    constants=[]
    for name,value in sorted(request["public_constants"].items()):
        if type(value) in (int,float):
            require(math.isfinite(value),"Finite public constant")
            role=dict(name=name,kind="scalar")
        elif type(value) is list and len(value)==layout["input_slot_period"]:
            require(all(type(v) in (int,float) and math.isfinite(v) for v in value),
                    "Public constant vector")
            nonzero=sum(v!=0 for v in value)
            kind=("uniform_vector" if all(v==value[0] for v in value)
                  else "single_lane_vector" if nonzero==1 else "coordinate_vector")
            role=dict(name=name,kind=kind,elements=len(value),nonzero_elements=nonzero)
        else:
            raise ValueError("Public binding kind is not described")
        constants.append(role)
    if constants:
        groups["public_binding_roles"]=BINDINGS
    if reductions or broadcasts:
        groups["reduction_broadcast_shapes"]=SHAPES
    context.update(version=VERSION,selection=sorted(groups),rules={k:groups[k] for k in sorted(groups)},
        shape_examples=dict(reductions=reductions[:32],reduction_count=len(reductions),
                            broadcasts=broadcasts[:32],broadcast_count=len(broadcasts),
                            scope="Public logical shape annotations, not a ciphertext lowering plan"),
        logical_constant_scalar_bindings=logical_scalar_bindings(request,checked),
        constant_role_examples=dict(bindings=constants[:64],binding_count=len(constants),
                                   scope="Roles of already supplied public values, not replacement weights"))
    require(len(json.dumps(context,sort_keys=True).encode())<=32768,"Binding guidance byte bound")
    return context

def relevant(request):
    context=specification(request)
    return any(key in context["selection"] for key in
               ("packed_reduction","tensor_broadcast","public_binding_roles"))

def repair_hint(request,source,diagnostic="",stage="static_check"):
    if request.get("generation_guidance",{}).get("version")!=VERSION:
        return ""
    if type(source) is not str or len(source.encode())>65536:
        return ""
    try:
        tree=ast.parse(source)
    except (SyntaxError,ValueError,RecursionError):
        return ""
    if sum(1 for _ in ast.walk(tree))>4096:
        return ""
    hints=[]
    if stage in ("static_check","numerical_comparison","compiler","artifact_gate"):
        selected=specification(request)
        for name in ("public_binding_roles","packed_reduction","tensor_broadcast","reduction_broadcast_shapes"):
            if name in selected["rules"]:
                hints.append(selected["rules"][name])
    from unified_poseidon_guidance import repair_hint as previous
    old=dict(request,generation_guidance=dict(request["generation_guidance"],version="explicit-v10"))
    hint=previous(old,source,diagnostic,stage)
    if hint:
        hints.append(hint)
    return "\n".join(dict.fromkeys(hints))[:5000]
