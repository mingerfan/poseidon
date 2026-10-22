"""Versioned logical profiles and canonical physical chunk bindings.

V2 retains four input/output chunks. V3 permits sixteen for explicitly configured profiles.
V4 explicitly binds larger periods to the same profile and immutable physical layout.
CKKS parameters, construction work and sandbox limits remain unchanged.
"""
import math
from benchmark_graph import require,dimensions,NAME,canonical,validate,graph_resource_profile
from model_resources import identity,resolve
from unified_chunk_layout import PERIODS

ABI="unified-capacity-chunked-inputs-v2"
PHYSICAL_INPUTS=4
PHYSICAL_OUTPUTS=4
EXTENDED_ABI="unified-capacity-chunked-inputs-v3"
WIDE_ABI="unified-capacity-chunked-inputs-v4"
WIDE_PERIODS=(512,1024,2048,4096)
EXTENDED_ABIS=(EXTENDED_ABI,WIDE_ABI)
ABIS=(ABI,*EXTENDED_ABIS)
EXTENDED_INPUTS=16
EXTENDED_OUTPUTS=16
EXTENDED_PROFILES=("resnet20-v1","general-tensors-v1")

def physical_limits(profile,execution_abi=None):
    resolve(profile)
    chosen=execution_abi if execution_abi is not None else (EXTENDED_ABI if profile in EXTENDED_PROFILES else ABI)
    require(chosen in ABIS,"Unknown capacity execution ABI")
    require(chosen not in EXTENDED_ABIS or profile in EXTENDED_PROFILES,"Extended capacity requires a configured sixteen-chunk profile")
    return (chosen,EXTENDED_INPUTS,EXTENDED_OUTPUTS) if chosen in EXTENDED_ABIS else (chosen,PHYSICAL_INPUTS,PHYSICAL_OUTPUTS)

def entry_parameter_limit(value,input_names,expected_outputs,slot_period):
    """Grant wider entry arity only from a fully validated immutable V3 layout."""
    validate_layout(value)
    require(value["execution_abi"] in EXTENDED_ABIS,"Extended entry requires capacity V3/V4")
    names=tuple(b["dsl_name"] for b in value["inputs"])+("zero_ct",)
    require(type(input_names) in (list,tuple) and tuple(input_names)==names and
            type(expected_outputs) is int and expected_outputs==value["output_ciphertexts"] and
            type(slot_period) is int and slot_period==value["input_slot_period"],
            "Extended entry/layout binding mismatch")
    return len(names)

def declarations(inputs,outputs,period,profile,execution_abi=None):
    limits=resolve(profile)
    if execution_abi is None and type(period) is int and period in WIDE_PERIODS:
        execution_abi=WIDE_ABI
    abi,max_inputs,max_outputs=physical_limits(profile,execution_abi)
    require(type(inputs) is list and 1<=len(inputs)<=limits.inputs and
            type(outputs) is list and 1<=len(outputs)<=limits.outputs,"Capacity logical I/O count")
    sizes=[]
    for entries,maximum in ((inputs,limits.input_elements),(outputs,limits.output_elements)):
        require(all(type(s) is dict and set(s)=={"name","shape"} and type(s["name"]) is str
                    and NAME.fullmatch(s["name"]) for s in entries),"Capacity tensor declaration")
        require(len({s["name"] for s in entries})==len(entries),"Duplicate capacity tensor name")
        current=[math.prod(dimensions(s["shape"],resource_profile=profile)) for s in entries]
        require(sum(current)<=maximum,"Capacity logical element budget")
        sizes.append(current)
    if period is None:
        period=next((p for p in PERIODS if sum(math.ceil(n/p) for n in sizes[0])<=max_inputs
                     and sum(math.ceil(n/p) for n in sizes[1])<=max_outputs),None)
    allowed=WIDE_PERIODS if abi==WIDE_ABI else PERIODS
    require(type(period) is int and period in allowed,"Capacity needs a configured period and bounded physical layout")
    require(sum(math.ceil(n/period) for n in sizes[0])<=max_inputs,f"Capacity input ciphertext budget (maximum {max_inputs})")
    require(sum(math.ceil(n/period) for n in sizes[1])<=max_outputs,f"Capacity output ciphertext budget (maximum {max_outputs})")
    physical=[];results=[];selectors=[];ciphertext=0
    for spec,n in zip(inputs,sizes[0]):
        for start in range(0,n,period):
            physical.append(dict(name=spec["name"],dsl_name=("x","y","z","t")[len(physical)] if abi==ABI else "x"+str(len(physical)),
                shape=spec["shape"],offset=start,elements=min(period,n-start)))
    for spec,n in zip(outputs,sizes[1]):
        chunks=[]
        for start in range(0,n,period):
            count=min(period,n-start)
            chunks.append(dict(ciphertext=ciphertext,offset=start,elements=count))
            selectors.extend([[ciphertext,i] for i in range(count)]);ciphertext+=1
        results.append(dict(name=spec["name"],shape=spec["shape"],chunks=chunks))
    return dict(execution_abi=abi,resource_profile=identity(profile),input_slot_period=period,
        logical_inputs=inputs,inputs=physical,outputs=results,output_ciphertexts=ciphertext,
        output_selectors=selectors,output_shape=[len(selectors)],output_representation="named_chunked_prefixes",
        auxiliary_ciphertexts=[dict(dsl_name="zero_ct",kind="encrypted_zero",source="trusted_client",slot_period=period)])

def layout(model,period=None,*,execution_abi=None,output_binding_policy=None):
    if output_binding_policy is not None:
        from unified_output_aliases import POLICY,layout as alias_layout
        require(output_binding_policy==POLICY,"Unknown output binding policy")
        return alias_layout(model,period,execution_abi=execution_abi)
    checked=validate(model)
    return declarations(model["inputs"],[dict(name=o["name"],shape=checked["shapes"][o["value"]])
                                       for o in model["outputs"]],period,graph_resource_profile(model),execution_abi)

def validate_layout(value):
    require(type(value) is dict and value.get("execution_abi") in ABIS,"Capacity layout identity")
    profile=value.get("resource_profile",{})
    require(type(profile) is dict and type(profile.get("name")) is str and
            profile==identity(profile["name"]),"Changed capacity resource profile")
    require(type(value.get("outputs")) is list and all(type(o) is dict and {"name","shape"}<=set(o)
            for o in value["outputs"]),"Capacity output declarations")
    if "output_binding" in value:
        from unified_output_aliases import validate_layout as validate_alias_layout
        return validate_alias_layout(value)
    expected=declarations(value.get("logical_inputs"),[dict(name=o["name"],shape=o["shape"])
        for o in value["outputs"]],value.get("input_slot_period"),profile["name"],value["execution_abi"])
    require(canonical(value)==canonical(expected),"Changed capacity physical layout")
    return expected["input_slot_period"]

def rules(value):
    extra=""
    if "output_binding" in value:
        from unified_output_aliases import rules as alias_rules
        extra=alias_rules(value)
    p=value["resource_profile"]
    if value["execution_abi"] in EXTENDED_ABIS:
        validate_layout(value)
        text="""
Capacity ABI unified-capacity-chunked-inputs-v3: resource_profile and physical bindings are immutable.
Only resnet20-v1 uses this ABI. At most sixteen input chunks and sixteen output chunks;
append exactly one trusted encrypted zero_ct to the ordered x0,x1,... entry parameters.
Logical tensors retain their declared shapes and names. Flatten each separately in C order,
partition into P-element chunks, pad each final chunk, and repeat each chunk across 16384 slots.
Return all physical ciphertexts in the frozen output order; selectors decode only logical elements.
Every logical operation must include all chunks, including tails. A chunk rotation is not a
whole-tensor rotation. Reductions exclude padded slots, including padding changed by public bias.
Native helpers retain sixteen parameters and object storage retains sixteen cells.
Source, AST, work, sandbox, depth, rotations and security limits remain unchanged.
"""+"Logical input/output element limits: "+str(p["input_elements"])+"/"+str(p["output_elements"])+".\n"
        if p["name"]=="general-tensors-v1":
            text=text.replace("Only resnet20-v1 uses this ABI.", "This request selects the general-tensors-v1 profile (logical rank 1..32).")
            text+="Individual operators retain their registered rank and axis restrictions. Object-array construction bounds remain unchanged.\n"
            text+="Static transpose/permute use the tensor_route masks from the frozen packing descriptor; they route logical C-order elements across chunks.\n"
        if value["execution_abi"]==WIDE_ABI:
            text=text.replace(EXTENDED_ABI,WIDE_ABI)
            text+="V4 uses an explicitly bound P in 512/1024/2048/4096; no other ABI permits these periods.\n"
            text+="For slice, tensor_route masks map static C-order source elements to the declared output, including cross-chunk boundaries.\n"
        return text+extra
    return """
Capacity ABI unified-capacity-chunked-inputs-v2: resource_profile is immutable.
The logical graph uses its declared resource profile; physical chunk bindings,
periods, rotations, four input/output ciphertexts and encrypted-zero rules remain fixed.
Every logical operation includes all chunks and C-order elements, including tail chunks.
A rotation of one chunk is not a rotation of the whole tensor. Reductions must exclude
padded tail slots, including padding made nonzero by public additions.
A larger logical graph does not relax source, work, sandbox, depth or security bounds.
"""+"Logical input/output element limits: "+str(p["input_elements"])+"/"+str(p["output_elements"])+".\n"+extra

def tail_masks(model,period):
    checked=validate(model)
    lengths={math.prod(shape)%period for name,shape in checked["shapes"].items()
             if checked["kinds"][name]=="cipher" and math.prod(shape)%period}
    return {"tail"+str(n):[float(i<n) for i in range(period)] for n in sorted(lengths)}

def period_rotations(period,value):
    """Authorize a wide rotation set only after checking every physical binding."""
    validate_layout(value)
    require(value["execution_abi"]==WIDE_ABI and type(period) is int and
            period==value["input_slot_period"],"Wide period/layout binding mismatch")
    return tuple(1<<i for i in range(period.bit_length()-1))
