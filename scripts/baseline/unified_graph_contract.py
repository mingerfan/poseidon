"""Independent logical model/layout preparation for the opt-in unified graph ABI."""
import math
from benchmark_graph import FORMAT, canonical, digest, require, validate

ABI="unified-periodic-inputs-v1"
TASK="hecate-unified-graph-synthesis-v1"
CONTRACT="hecate-unified-native-v1"
COMPACT="compact-periodic-v1"
LEGACY_ORIGINS={"kind":"verified public model and layout"}
COMPACT_ORIGINS={**LEGACY_ORIGINS,"policy":COMPACT,
    "packing_masks":"mask0 selects slot zero. A replicated ciphertext value can be multiplied by mask0 and then rotated to the desired output slot using permitted rotations."}
GUIDANCE={"schema":1,"model":"logical tensor graph","packing":"common periodic inputs, named packed outputs",
          "linear":"cross-element weighted reduction","array_cells":"whole Exprs, never slots",
          "scope":"bounded native construction, no unrestricted Python"}
RULES="""Implement the supplied logical graph exactly, including every named output.
Only @hc.func("c,p,...") decorated functions: golden plus at most 16 scalar-Expr helpers.
Golden parameters must match layout.inputs in order, followed by zero_ct; all are c.
Every logical input is encrypted separately: C-order flatten, pad to P, repeat across
16384 slots. P=layout.input_slot_period. Never concatenate logical inputs before encryption.
Return one ciphertext per named output in layout.outputs order, with logical values in
its C-order prefix, repeated every P slots. Other slots are unobserved.
Arithmetic +,-,*, unary minus requires a ciphertext operand. Public constants are immutable
scalar or length-P values in public_constants; finite literals with magnitude<=1024 allowed.
Positive rotate(k) reads slot j+k; only positive powers of two below P are callable.
Compose rotations for other displacements. Linear is a reduction, not elementwise multiply.
zero_ct is client-encrypted zero, not an extra model input or a bootstrap substitute.
Compiler owns security, scale, level, rescale, modswitch and relinearization.
Helpers have typed scalar c/p positional parameters, no recursion, defaults, kwargs or imports.
Public bounded range loops, assignments, tuple/list unpacking and final returns are permitted.
np.array(data,dtype=object), literal indexing/slicing, reshape/flatten/copy/transpose/T/item
operate on whole Expr storage: rank<=4, cells<=16, never ciphertext slots.
Array-left +,-,* uses outer-storage broadcasting. Scalar +=,-=,*= rebind; array name-target
augmentation mutates shared storage, copies remain independent. No subscript writes.
Starred calls expand the first axis into scalar Expr arguments. No encrypted conditions,
while, arbitrary calls, I/O, network, bootstrap, raw level changes or hidden data access.
Expanded work<=4096, golden work<=1024, source<=65536 bytes, AST and sandbox bounds apply.
Do not change weights, layout, reference, numerical thresholds or test inputs.
Return only response-schema JSON. Numerical correctness requires real encrypted validation.
"""


def layout(model,chunk_period=None):
    if chunk_period is not None:
        from unified_chunk_layout import layout as chunk_layout
        return chunk_layout(model,chunk_period)
    check=validate(model)
    sizes=[math.prod(s["shape"]) for s in model["inputs"]]+[
        math.prod(s) for s in check["output_shapes"].values()]
    period=next(p for p in (4,8,16,32,64,128,256) if p>=max(sizes))
    inputs=[dict(name=s["name"],dsl_name=("x","y","z","t")[i],shape=s["shape"])
            for i,s in enumerate(model["inputs"])]
    outputs=[dict(name=o["name"],shape=check["shapes"][o["value"]],ciphertext=i)
             for i,o in enumerate(model["outputs"])]
    selectors=[[i,j] for i,o in enumerate(outputs) for j in range(math.prod(o["shape"]))]
    return dict(execution_abi=ABI,input_slot_period=period,inputs=inputs,outputs=outputs,
                output_ciphertexts=len(outputs),output_selectors=selectors,
                output_shape=[len(selectors)],output_representation="named_packed_prefixes",
                auxiliary_ciphertexts=[dict(dsl_name="zero_ct",kind="encrypted_zero",source="trusted_client",
                                             slot_period=period)])


def validate_layout(value):
    from unified_chunk_layout import ABI as CHUNK_ABI,validate_layout as validate_chunk
    if type(value) is dict and value.get("execution_abi")==CHUNK_ABI:return validate_chunk(value)
    require(type(value) is dict and value.get("execution_abi")==ABI,"Unified layout identity")
    require(set(value)=={"execution_abi","input_slot_period","inputs","outputs","output_ciphertexts",
                         "output_selectors","output_shape","output_representation","auxiliary_ciphertexts"},
            "Unified layout fields")
    # Reconstruct only logical declarations, then compare every derived field.
    specs=value["inputs"];outputs=value["outputs"]
    require(type(specs) is list and 1<=len(specs)<=4 and type(outputs) is list and 1<=len(outputs)<=4,
            "Unified ABI arity")
    from benchmark_graph import dimensions,NAME
    require(all(type(s) is dict and set(s)=={"name","dsl_name","shape"} and
                type(s["name"]) is str and NAME.fullmatch(s["name"]) and
                s["dsl_name"]==("x","y","z","t")[i] for i,s in enumerate(specs)),"Unified input binding")
    require(len({s["name"] for s in specs})==len(specs),"Repeated input")
    shapes=[dimensions(s["shape"]) for s in specs]
    require(sum(math.prod(s) for s in shapes)<=256,"Unified input budget")
    require(all(type(o) is dict and set(o)=={"name","shape","ciphertext"} and
                type(o["name"]) is str and NAME.fullmatch(o["name"]) and type(o["ciphertext"]) is int and
                o["ciphertext"]==i for i,o in enumerate(outputs)),"Unified output binding")
    require(len({o["name"] for o in outputs})==len(outputs),"Repeated output")
    outs=[dimensions(o["shape"]) for o in outputs]
    require(sum(math.prod(s) for s in outs)<=256,"Unified output budget")
    p=next(p for p in (4,8,16,32,64,128,256) if p>=max(math.prod(s) for s in shapes+outs))
    selectors=[[i,j] for i,s in enumerate(outs) for j in range(math.prod(s))]
    require(canonical(value)==canonical(dict(execution_abi=ABI,input_slot_period=p,inputs=specs,outputs=outputs,
        output_ciphertexts=len(outs),output_selectors=selectors,output_shape=[len(selectors)],
        output_representation="named_packed_prefixes",
        auxiliary_ciphertexts=[dict(dsl_name="zero_ct",kind="encrypted_zero",source="trusted_client",slot_period=p)])),
        "Changed unified physical layout")
    return p


def scalars(value):
    if type(value) is list:
        return [v for child in value for v in scalars(child)]
    return [float(value)]


def registry(model,plan,policy=None):
    """Public constant preparation is independent of rule-emitted DSL."""
    require(policy in (None,COMPACT),"Unknown unified constant policy")
    check=validate(model)
    numbers={0.0,1.0,-1.0,2.0,-2.0,0.5}
    for value in model["constants"].values():numbers.update(scalars(value))
    for node in model["nodes"]:
        op=node["op"];refs=node["inputs"];a=node["attrs"]
        if op=="mean":
            shape=check["shapes"][refs[0]]
            numbers.add(1.0/math.prod(shape[i%len(shape)] for i in a["axes"]))
        if op in ("avg_pool1d","avg_pool2d"):
            import itertools
            rank=1 if op.endswith("1d") else 2
            spatial=check["shapes"][refs[0]][-rank:]
            output=check["shapes"][node["outputs"][0]][-rank:]
            if a["count_include_pad"]:numbers.add(1.0/math.prod(a["kernel"]))
            else:
                for pos in itertools.product(*(range(n) for n in output)):
                    valid=math.prod(max(0,min(n,p*st-pad+k)-max(0,p*st-pad))
                                    for n,p,st,pad,k in zip(spatial,pos,a["stride"],a["padding"],a["kernel"]))
                    require(valid>0,"Empty average window")
                    numbers.add(1.0/valid)
        if op=="batch_norm":
            mean,var,gamma,beta=[model["constants"][r] for r in refs[1:]]
            for m,v,g,b in zip(mean,var,gamma,beta):
                gain=g/math.sqrt(v+a["eps"]);numbers.update((gain,b-m*gain))
    require(all(math.isfinite(v) and abs(v)<=1024 for v in numbers),"Derived public constant magnitude")
    values={"p"+str(i):v for i,v in enumerate(sorted(numbers))}
    max_output=max(math.prod(o["shape"]) for o in plan["outputs"])
    from unified_chunk_layout import ABI as CHUNK_ABI
    if plan["execution_abi"]==CHUNK_ABI:max_output=min(max_output,plan["input_slot_period"])
    p=plan["input_slot_period"]
    for i in range(1 if policy==COMPACT else max(1,max_output)):values["mask"+str(i)]=[float(j==i) for j in range(p)]
    require(len(values)<=256,"Unified constant registry budget")
    return values


def prepare(model,profile_hash,configuration=None,construction=None,*,construction_profile=None,constant_policy=None,helper_profile=None,helper_exercise=None,chunk_period=None,generation_guidance=None,capability_composition=None):
    from candidate_contract import RESPONSE_SCHEMA
    require(constant_policy in (None,COMPACT),"Unknown unified constant policy")
    if capability_composition is not None:
        from capability_combinations import validate_selection
        validate_selection(capability_composition,construction_profile,construction,helper_profile,helper_exercise,chunk_period)
    p=layout(model,chunk_period)
    from upstream_candidate_helpers import CHUNK_PROFILE
    require((helper_profile!=CHUNK_PROFILE and (chunk_period is None or helper_profile is None))
            or (helper_profile==CHUNK_PROFILE and chunk_period is not None),
            "Chunk layout requires separately verified upstream helper bindings")
    try:
        constants=registry(model,p,constant_policy)
    except ValueError as error:
        if constant_policy is not None or str(error)!="Unified constant registry budget":raise
        constant_policy=COMPACT
        constants=registry(model,p,constant_policy)
    body=dict(schema=1,task=TASK,model=model,fx_graph="independent typed logical DAG; no rule DSL included",
              public_constants=constants,constant_origins=dict(COMPACT_ORIGINS if constant_policy==COMPACT else LEGACY_ORIGINS),
              layout=p,rules=RULES,response_schema=RESPONSE_SCHEMA,semantic_guidance=GUIDANCE,
              compiler_profile_sha256=profile_hash,privacy={"input":"encrypted","weights":"public"})
    if helper_profile is not None:
        from upstream_candidate_helpers import PROFILE,BN_PROFILE,CONCAT_PROFILE,SPATIAL_PROFILE,MAPPED_PROFILE,FUSED_PROFILE,DS_PROFILE,VR_PROFILE,CHUNK_PROFILE,POLYNOMIAL_PROFILE,POLYNOMIAL_RULES,manifest,RULES as HELPER_RULES,BN_RULES,CONCAT_RULES,SPATIAL_RULES,MAPPED_RULES,FUSED_RULES,DS_RULES,VR_RULES,CHUNK_RULES
        require(helper_profile in (PROFILE,BN_PROFILE,CONCAT_PROFILE,SPATIAL_PROFILE,MAPPED_PROFILE,FUSED_PROFILE,DS_PROFILE,VR_PROFILE,CHUNK_PROFILE,POLYNOMIAL_PROFILE) and construction_profile is None and (construction is None or capability_composition is not None),
                "Upstream helpers currently require free native construction")
        body["upstream_helpers"]=manifest(model,helper_profile,chunk_period)
        body["rules"]+=POLYNOMIAL_RULES if helper_profile==POLYNOMIAL_PROFILE else CHUNK_RULES if helper_profile==CHUNK_PROFILE else VR_RULES if helper_profile==VR_PROFILE else DS_RULES if helper_profile==DS_PROFILE else FUSED_RULES if helper_profile==FUSED_PROFILE else MAPPED_RULES if helper_profile==MAPPED_PROFILE else SPATIAL_RULES if helper_profile==SPATIAL_PROFILE else HELPER_RULES if helper_profile==PROFILE else CONCAT_RULES if helper_profile==CONCAT_PROFILE else BN_RULES
    if helper_exercise is not None:
        from upstream_helper_coverage import exercise_spec,RULES as WITNESS_RULES
        require(helper_profile is not None,'Directed helper requires capability')
        body['upstream_exercise']=exercise_spec(helper_exercise,body['upstream_helpers'])
        body['rules']+=WITNESS_RULES
    if construction_profile is not None:
        from unified_public_contract import CONTRACT as PUBLIC,RULES as PUBLIC_RULES,GUIDANCE as PUBLIC_GUIDANCE
        require(construction_profile==PUBLIC,"Unknown unified construction profile")
        body.update(construction_profile=PUBLIC,rules=PUBLIC_RULES,semantic_guidance=PUBLIC_GUIDANCE)
    if chunk_period is not None:
        from unified_chunk_layout import chunk_rules
        body["rules"]=chunk_rules(body["rules"],public=construction_profile is not None)
    if construction is not None:
        if construction_profile is None:
            from unified_graph_exercises import spec
        else:
            from unified_public_exercises import spec
        body["construction_exercise"]=spec(construction)
    if capability_composition is not None:
        from capability_combinations import RULES as COMPOSITION_RULES
        body["capability_composition"]=capability_composition
        body["rules"]+=COMPOSITION_RULES
    if configuration is not None:body["compiler_configuration"]=configuration
    if generation_guidance is not None:
        require(generation_guidance in (EXPLICIT_GUIDANCE,COMPOSITE_GUIDANCE,MATHEMATICAL_GUIDANCE,DIRECTED_GUIDANCE,TYPED_GUIDANCE,BINDING_GUIDANCE,PRECISE_GUIDANCE,GATE_GUIDANCE,EXPRESSION_GUIDANCE),"Unknown unified generation guidance")
        body["generation_guidance"]=explicit_generation_guidance(body,generation_guidance)
    request=dict(body,request_id=digest(body))
    if len(canonical(request))>131072 and constant_policy is None:
        return prepare(model,profile_hash,configuration,construction,
                       construction_profile=construction_profile,constant_policy=COMPACT,helper_profile=helper_profile,helper_exercise=helper_exercise,chunk_period=chunk_period,generation_guidance=generation_guidance,capability_composition=capability_composition)
    require(len(canonical(request))<=131072,"Unified request size budget")
    return request


def validate_request(request):
    from unified_public_contract import CONTRACT as PUBLIC,RULES as PUBLIC_RULES,GUIDANCE as PUBLIC_GUIDANCE
    from candidate_contract import RESPONSE_SCHEMA
    required={"schema","task","model","fx_graph","public_constants","constant_origins","layout",
              "rules","response_schema","semantic_guidance","compiler_profile_sha256","privacy","request_id"}
    require(type(request) is dict and required<=set(request)<=required|{"construction_profile","construction_exercise","compiler_configuration","upstream_helpers","upstream_exercise","generation_guidance","capability_composition"},
            "Unified request fields")
    require(type(request["schema"]) is int and request["schema"]==1 and request["response_schema"]==RESPONSE_SCHEMA and
            request["privacy"]=={"input":"encrypted","weights":"public"} and
            request["fx_graph"]=="independent typed logical DAG; no rule DSL included","Unified request metadata")
    origins=request["constant_origins"]
    require(origins==LEGACY_ORIGINS or origins==COMPACT_ORIGINS,"Unified constant policy metadata")
    policy=COMPACT if origins==COMPACT_ORIGINS else None
    public="construction_profile" in request
    if public:
        require(request["construction_profile"]==PUBLIC,"Unknown unified construction profile")
    expected_rules=PUBLIC_RULES if public else RULES
    from unified_chunk_layout import ABI as CHUNK_ABI,chunk_rules
    chunk=request["layout"].get("execution_abi")==CHUNK_ABI
    from upstream_candidate_helpers import CHUNK_PROFILE
    helper=request.get("upstream_helpers",{}).get("profile")
    composition=request.get("capability_composition")
    if composition is not None:
        from capability_combinations import validate_selection
        validate_selection(composition,request.get("construction_profile"),
            request.get("construction_exercise",{}).get("id"),helper,
            request.get("upstream_exercise",{}).get("required_helpers"),
            request["layout"]["input_slot_period"] if chunk else None)
    require((helper!=CHUNK_PROFILE and (not chunk or helper is None))
            or (helper==CHUNK_PROFILE and chunk),
            "Chunk layout requires separately verified upstream helper bindings")
    if chunk and helper!=CHUNK_PROFILE:expected_rules=chunk_rules(expected_rules,public=public)
    if "upstream_helpers" in request:
        from upstream_candidate_helpers import validate_manifest,PROFILE,CONCAT_PROFILE,SPATIAL_PROFILE,MAPPED_PROFILE,FUSED_PROFILE,DS_PROFILE,VR_PROFILE,CHUNK_PROFILE,POLYNOMIAL_PROFILE,POLYNOMIAL_RULES,RULES as HELPER_RULES,BN_RULES,CONCAT_RULES,SPATIAL_RULES,MAPPED_RULES,FUSED_RULES,DS_RULES,VR_RULES,CHUNK_RULES
        require(not public and ("construction_exercise" not in request or composition is not None),
                "Upstream helpers currently require free native construction")
        validate_manifest(request["upstream_helpers"],request["model"],request["layout"])
        expected_rules+=POLYNOMIAL_RULES if request["upstream_helpers"]["profile"]==POLYNOMIAL_PROFILE else CHUNK_RULES if request["upstream_helpers"]["profile"]==CHUNK_PROFILE else VR_RULES if request["upstream_helpers"]["profile"]==VR_PROFILE else DS_RULES if request["upstream_helpers"]["profile"]==DS_PROFILE else FUSED_RULES if request["upstream_helpers"]["profile"]==FUSED_PROFILE else MAPPED_RULES if request["upstream_helpers"]["profile"]==MAPPED_PROFILE else SPATIAL_RULES if request["upstream_helpers"]["profile"]==SPATIAL_PROFILE else HELPER_RULES if request["upstream_helpers"]["profile"]==PROFILE else CONCAT_RULES if request["upstream_helpers"]["profile"]==CONCAT_PROFILE else BN_RULES
    if 'upstream_exercise' in request:
        from upstream_helper_coverage import exercise_spec,RULES as WITNESS_RULES
        require('upstream_helpers' in request,'Directed helper requires capability')
        value=request['upstream_exercise']
        require(type(value) is dict and canonical(value)==canonical(exercise_spec(value.get('required_helpers'),request['upstream_helpers'])),
                'Changed directed helper exercise')
        expected_rules+=WITNESS_RULES
    if chunk and helper==CHUNK_PROFILE:expected_rules=chunk_rules(expected_rules,public=public)
    if composition is not None:
        from capability_combinations import RULES as COMPOSITION_RULES
        expected_rules+=COMPOSITION_RULES
    require(request.get("task")==TASK and request.get("rules")==expected_rules and
            request.get("semantic_guidance")==(PUBLIC_GUIDANCE if public else GUIDANCE),"Unified request contract")
    from compiler_configuration import request_configuration
    request_configuration(request)
    if "construction_exercise" in request:
        if public:
            from unified_public_exercises import validate_request as check_exercise
        else:
            from unified_graph_exercises import validate_request as check_exercise
        check_exercise(request)
    p=validate_layout(request["layout"])
    require(canonical(request["layout"])==canonical(layout(request["model"],p if chunk else None)),"Model/layout mismatch")
    require(canonical(request["public_constants"])==canonical(registry(request["model"],request["layout"],policy)),
            "Public registry changed")
    if "generation_guidance" in request:
        value=request["generation_guidance"]
        require(type(value) is dict and value.get("version") in (EXPLICIT_GUIDANCE,COMPOSITE_GUIDANCE,MATHEMATICAL_GUIDANCE,DIRECTED_GUIDANCE,TYPED_GUIDANCE,BINDING_GUIDANCE,PRECISE_GUIDANCE,GATE_GUIDANCE,EXPRESSION_GUIDANCE),
                "Changed unified generation guidance")
        require(canonical(value)==canonical(explicit_generation_guidance(request,value["version"])),
                "Changed unified generation guidance")
    require(request["request_id"]==digest({k:v for k,v in request.items() if k!="request_id"}),
            "Unified request hash")
    require(len(canonical(request))<=131072,"Unified request bytes")
    return p


def validate_candidate(candidate,request,*,check_construction=True):
    p=validate_request(request)
    require(type(candidate) is dict and set(candidate)=={"schema","request_id","hecate_source"} and
            type(candidate["schema"]) is int and candidate["schema"]==1 and
            candidate["request_id"]==request["request_id"],"Unified candidate identity")
    if "construction_profile" in request:
        from unified_public_contract import validate
        return validate(candidate["hecate_source"],request,check_construction=check_construction)
    from decorated_functions import validate as native_validate
    result=native_validate(candidate["hecate_source"],request["public_constants"],
        request["layout"]["output_ciphertexts"],input_names=tuple(s["dsl_name"] for s in request["layout"]["inputs"])+("zero_ct",),
        array_mutation=True,slot_period=p,upstream_helpers="upstream_helpers" in request,upstream_request=request if "upstream_helpers" in request else None)
    require(result["functions"]["golden"]["expanded_cost"]<=1024,"Unified candidate work budget")
    if "capability_composition" in request:
        from capability_combinations import validate_calls
        validate_calls(candidate["hecate_source"],request)
    if check_construction and "construction_exercise" in request:
        from unified_native_coverage import check_exercise
        result["construction_exercise"]=check_exercise(candidate["hecate_source"],request)
    if check_construction and 'upstream_exercise' in request:
        from upstream_helper_coverage import check_exercise
        result['upstream_exercise']=check_exercise(candidate['hecate_source'],request)
    return result


# Optional request metadata: omission retains every pre-existing request byte/hash.
EXPLICIT_GUIDANCE="explicit-v1"
COMPOSITE_GUIDANCE="explicit-v2"
MATHEMATICAL_GUIDANCE="explicit-v3"
DIRECTED_GUIDANCE="explicit-v4"
TYPED_GUIDANCE="explicit-v5"
BINDING_GUIDANCE="explicit-v6"
PRECISE_GUIDANCE="explicit-v7"
GATE_GUIDANCE="explicit-v8"
EXPRESSION_GUIDANCE="explicit-v9"


def explicit_generation_guidance(request,version=EXPLICIT_GUIDANCE):
    """Expose only the public ABI and contract; never a model implementation."""
    require(version in (EXPLICIT_GUIDANCE,COMPOSITE_GUIDANCE,MATHEMATICAL_GUIDANCE,DIRECTED_GUIDANCE,TYPED_GUIDANCE,BINDING_GUIDANCE,PRECISE_GUIDANCE,GATE_GUIDANCE,EXPRESSION_GUIDANCE),"Unknown unified generation guidance")
    names=[s["dsl_name"] for s in request["layout"]["inputs"]]+["zero_ct"]
    signature=",".join("c" for _ in names)
    public="construction_profile" in request
    result={
        "version":version,
        "entry_name":"golden",
        "ordered_parameters":names,
        "entry_header":'@hc.func("'+signature+'")\ndef golden('+", ".join(names)+'):',
        "entry_rule":"Define exactly one top-level golden. The header has no body; implement the supplied graph yourself. Use the literal decorator call with parentheses and the exact c signature, never bare @hc.func.",
        "parameter_rule":"Use exactly these unannotated positional names, without defaults, varargs or keyword-only parameters. zero_ct is the final trusted ciphertext argument.",
        "constant_rule":"Use public_constants keys directly as prebound names. Do not define replacement weights or load a dictionary/file. hc and np are provided by the trusted runner; do not import them.",
        "helper_rule":("Additional construction helpers are undecorated and obey the public construction rules."
                       if public else "Each additional native helper needs its own literal @hc.func c/p signature matching its unannotated positional parameters. Helper c/p types need not match golden."),
        "rotation_rule":"Call ciphertext.rotate(k), never hc.rotate(...). The request rules determine permitted displacements.",
        "output_rule":"Return exactly layout.output_ciphertexts whole ciphertext expressions in the frozen output order; object-array cells are whole expressions, not slots.",
        "authority":"This guidance clarifies the existing rules; it adds no allowed syntax, helper, resource budget or backend capability.",
    }
    if version in (COMPOSITE_GUIDANCE,MATHEMATICAL_GUIDANCE,DIRECTED_GUIDANCE,TYPED_GUIDANCE,BINDING_GUIDANCE,PRECISE_GUIDANCE,GATE_GUIDANCE,EXPRESSION_GUIDANCE):
        # Reuse authoritative component instructions, never golden recipes or
        # model-specific implementations. Legacy and v1 request bytes stay fixed.
        requirements=[]
        exercise=request.get("construction_exercise",{})
        if public and exercise.get("aggregate_requirement"):
            from unified_public_exercises import spec
            for feature in exercise["required_features"]:
                component=spec("unified-public-"+feature.replace(".","-"))
                require(component["required_features"]==[feature],"Composite component mismatch")
                requirements.append(dict(feature=feature,instruction=component["instruction"]))
        result["construction_requirements"]=requirements
    if version in (MATHEMATICAL_GUIDANCE,DIRECTED_GUIDANCE,TYPED_GUIDANCE,BINDING_GUIDANCE,PRECISE_GUIDANCE,GATE_GUIDANCE,EXPRESSION_GUIDANCE):
        from unified_logical_semantics import specification
        result["logical_model_semantics"]=specification(request["model"])
        result["encrypted_zero_rule"]=(
            "The SEAL backend can reject transparent ciphertexts produced by exact ciphertext self-cancellation "
            "or multiplication by plaintext zero. The supplied zero_ct is a genuine client-encrypted zero. "
            "Preserve the mathematical graph semantics without relying on a transparent intermediate; "
            "do not disable backend checks or decrypt/re-encrypt.")
    if version in (DIRECTED_GUIDANCE,TYPED_GUIDANCE,BINDING_GUIDANCE,PRECISE_GUIDANCE,GATE_GUIDANCE,EXPRESSION_GUIDANCE):
        from unified_directed_semantics import specification
        result["directed_semantics"]=specification(request)
    if version in (TYPED_GUIDANCE,BINDING_GUIDANCE,PRECISE_GUIDANCE,GATE_GUIDANCE,EXPRESSION_GUIDANCE):
        from unified_failure_semantics import specification
        result["typed_construction_semantics"]=specification()
    if version in (BINDING_GUIDANCE,PRECISE_GUIDANCE,GATE_GUIDANCE,EXPRESSION_GUIDANCE):
        from unified_typed_guidance import specification
        result["typed_request_bindings"]=specification(request)
    if version in (PRECISE_GUIDANCE,GATE_GUIDANCE,EXPRESSION_GUIDANCE):
        from unified_precise_guidance import specification
        result["precise_construction_rules"]=specification(request)
    if version in (GATE_GUIDANCE,EXPRESSION_GUIDANCE):
        from unified_gate_guidance import specification
        result["value_and_scale_discipline"]=specification()
    if version==EXPRESSION_GUIDANCE:
        from unified_expression_guidance import specification
        result["expression_construction_rules"]=specification(request)
    return result


def static_repair_hint(source,request,diagnostic=""):
    """Bounded AST observations with fixed messages, never execution or source echo."""
    import ast
    if "generation_guidance" not in request:
        return ""
    validate_request(request)
    if type(source) is not str or len(source.encode())>65536:
        return ""
    try:
        tree=ast.parse(source)
    except (SyntaxError,ValueError,RecursionError):
        return ""
    if sum(1 for _ in ast.walk(tree))>4096:
        return ""
    guidance=explicit_generation_guidance(request,request["generation_guidance"]["version"])
    entry=[n for n in tree.body if type(n) is ast.FunctionDef and n.name=="golden"]
    hints=[]
    if len(entry)!=1:
        hints.append("Define exactly one top-level function named golden; main is not the entry point. Required header: "+guidance["entry_header"])
    else:
        fn=entry[0]
        expected=ast.parse(guidance["entry_header"]+"\n    pass").body[0]
        if ([ast.dump(d) for d in fn.decorator_list]!=
                [ast.dump(d) for d in expected.decorator_list]):
            hints.append("The golden decorator must be a literal hc.func call, not a bare attribute. Required header: "+guidance["entry_header"])
        if (ast.dump(fn.args)!=ast.dump(expected.args) or fn.returns is not None):
            hints.append("Golden parameters must be the exact unannotated positional ABI. Required header: "+guidance["entry_header"])
    if any(type(n) is ast.Call and type(n.func) is ast.Attribute and
           type(n.func.value) is ast.Name and n.func.value.id=="hc" and n.func.attr=="rotate"
           for n in ast.walk(tree)):
        hints.append(guidance["rotation_rule"])
    if "construction_profile" in request and any(
            type(n) is ast.FunctionDef and n.name!="golden" and n.decorator_list
            for n in ast.walk(tree)):
        hints.append(guidance["helper_rule"])
    for item in guidance.get("construction_requirements",[]):
        hints.append(item["feature"]+": "+item["instruction"])
    if guidance["version"]==TYPED_GUIDANCE:
        from unified_failure_semantics import repair_hint
        hint=repair_hint(diagnostic)
        if hint:hints.append(hint)
    if guidance["version"] in (BINDING_GUIDANCE,PRECISE_GUIDANCE,GATE_GUIDANCE,EXPRESSION_GUIDANCE):
        from unified_typed_guidance import repair_hint
        hint=repair_hint(request,diagnostic)
        if hint:hints.append(hint)
    if guidance["version"] in (PRECISE_GUIDANCE,GATE_GUIDANCE,EXPRESSION_GUIDANCE):
        from unified_precise_guidance import repair_hint
        hint=repair_hint(request,diagnostic)
        if hint:hints.append(hint)
    if guidance["version"]==EXPRESSION_GUIDANCE:
        from unified_expression_guidance import repair_hint
        hint=repair_hint(request,source,diagnostic)
        if hint:hints.append(hint)
    return "\n".join(hints)
