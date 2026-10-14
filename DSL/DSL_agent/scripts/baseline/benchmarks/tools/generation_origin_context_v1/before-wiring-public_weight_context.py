"""Opt-in public weight descriptors and bounded trusted request storage.

The generation digest commits to every original value through canonical content
hashes. Execution retains the checked numeric registry; no candidate file access.
"""
import copy
import json
import math
import re
from benchmark_graph import canonical, digest, require
from model_resources import resolve

POLICY = "descriptors-v1"
FORMAT = "poseidon-public-weight-descriptors-v1"
MATH_POLICY = "descriptors-math-roles-v2"
MATH_FORMAT = "poseidon-public-weight-descriptors-v2"
from public_origin_columns import POLICY as COMPACT_POLICY, WEIGHT_FORMAT as COMPACT_FORMAT
POLICIES = (POLICY,MATH_POLICY,COMPACT_POLICY)
CONTEXT_BYTES = 131072
TRUSTED_BYTES = 4*1024**2  # Existing application task-file bound, not a candidate bound.
RULES = """\nPublic weight descriptor contract v1:
Model constants name the logical weights. Only names listed in public_constants
are prebound DSL values; constant_origins describes their model origin.
The generation context gives shapes and content hashes instead of numeric data.
Use public_constants names directly; never substitute weights, open a file or load
a mapping. The trusted frontend supplies their fixed values during tracing.
\n"""

def rules_for(policy):
    require(policy in POLICIES,"Public weight policy")
    if policy in (MATH_POLICY,COMPACT_POLICY):
        from public_math_context import RULES as mathematical_rules
        if policy==COMPACT_POLICY:
            from public_origin_columns import RULES as origin_rules
            return mathematical_rules+origin_rules
        return mathematical_rules
    return RULES

def selected(request):
    return type(request) is dict and "public_weight_binding" in request

def storage_limit(request):
    return TRUSTED_BYTES if selected(request) else CONTEXT_BYTES

def _shape(value):
    if type(value) in (int,float):
        require(math.isfinite(value) and abs(value)<=1024, "Public weight numeric value")
        return []
    require(type(value) is list and value, "Public weight tensor")
    children=iter(value)
    child = _shape(next(children))
    require(all(_shape(item)==child for item in children), "Public weight rectangular shape")
    return [len(value),*child]

def descriptor(value):
    dims = _shape(value)
    return dict(kind="tensor" if dims else "scalar",shape=dims,elements=math.prod(dims),
                content_sha256=digest(value))

def binding(request,policy=POLICY):
    require(policy in POLICIES,"Public weight policy")
    model = request["model"]
    profile = model.get("resource_profile")
    require(profile in ("medium-v1","resnet20-v1","general-tensors-v1"),
            "Weight descriptors require a capacity resource profile")
    require(request.get("compiler_configuration",{}).get("backend")=="poseidon_cpu",
            "Weight descriptors require Poseidon CPU")
    value=dict(format=FORMAT,policy=POLICY,resource_profile=profile,
                model_constants={name:descriptor(value) for name,value in model["constants"].items()},
                public_constants={name:descriptor(value) for name,value in request["public_constants"].items()})
    if policy in (MATH_POLICY,COMPACT_POLICY):
        from public_math_context import build
        value.update(format=MATH_FORMAT if policy==MATH_POLICY else COMPACT_FORMAT,policy=policy)
        value["math_roles"]=build(request,value)
    return value

def project(request):
    info = request["public_weight_binding"]
    value = copy.deepcopy(request)
    if is_compact_projected(request):
        from public_origin_columns import registries, restore, verify_guidance
        registries(request);restore(request["constant_origins"]);verify_guidance(request)
        return value
    value["model"]["constants"] = copy.deepcopy(info["model_constants"])
    value["public_constants"] = copy.deepcopy(info["public_constants"])
    if info.get("policy")==COMPACT_POLICY:
        from public_origin_columns import project as project_origins, descriptor_references
        value["constant_origins"]=project_origins(request["constant_origins"])
        value["public_weight_binding"].update(descriptor_references(info["model_constants"],info["public_constants"]))
        if "generation_guidance" in value:
            from public_origin_columns import compact_guidance
            value["generation_guidance"]=compact_guidance(value["generation_guidance"],info["public_constants"])
    return value

def is_compact_projected(request):
    from public_origin_columns import FORMAT as ORIGIN_FORMAT
    return (type(request) is dict and request.get("public_weight_binding",{}).get("policy")==COMPACT_POLICY
            and type(request.get("constant_origins")) is dict and request["constant_origins"].get("format")==ORIGIN_FORMAT)

def request_digest(request):
    value = project(request) if selected(request) else request
    return digest({k:v for k,v in value.items() if k!="request_id"})

def verify_binding(request):
    require(canonical(request["public_weight_binding"]) == canonical(binding(request,policy=request["public_weight_binding"].get("policy"))), "Public weight binding changed")
    require(request["rules"].endswith(rules_for(request["public_weight_binding"].get("policy"))), "Public weight rules changed")
    context = project(request)
    require(len(canonical(context))<=CONTEXT_BYTES, "Weight descriptor generation context budget")
    require(request["request_id"]==request_digest(request), "Public weight request hash")
    return context

def verify_context(value):
    """Transport checks the approved projected hash; full semantics stay host-owned."""
    from candidate_contract import RESPONSE_SCHEMA
    from unified_graph_contract import TASK
    from compiler_configuration import request_configuration
    required={"schema","task","model","fx_graph","public_constants","constant_origins","layout",
              "rules","response_schema","semantic_guidance","compiler_profile_sha256","privacy","request_id",
              "public_weight_binding"}
    optional={"construction_profile","construction_exercise","compiler_configuration","upstream_helpers",
              "upstream_exercise","generation_guidance","capability_composition"}
    require(type(value) is dict and required<=set(value)<=required|optional and
            len(canonical(value))<=CONTEXT_BYTES, "Public weight context fields/budget")
    require(type(value["schema"]) is int and value["schema"]==1 and value["task"]==TASK and
            value["response_schema"]==RESPONSE_SCHEMA and value["privacy"]=={"input":"encrypted","weights":"public"} and
            value["fx_graph"]=="independent typed logical DAG; no rule DSL included" and
            type(value["rules"]) is str and value["rules"].endswith(rules_for(value.get("public_weight_binding",{}).get("policy"))), "Public weight context contract")
    request_configuration(value)
    require(value.get("compiler_configuration",{}).get("backend")=="poseidon_cpu","Weight descriptor backend")
    info = value["public_weight_binding"]
    require(type(info) is dict and set(info)==({"format","policy","resource_profile","model_constants","public_constants","math_roles"}
            if info.get("policy") in (MATH_POLICY,COMPACT_POLICY) else {"format","policy","resource_profile","model_constants","public_constants"}) and
            ((info["format"],info["policy"]) in ((FORMAT,POLICY),(MATH_FORMAT,MATH_POLICY),(COMPACT_FORMAT,COMPACT_POLICY))) and
            info["resource_profile"] in ("medium-v1","resnet20-v1","general-tensors-v1"),
            "Public weight descriptor contract")
    if info["policy"]==COMPACT_POLICY:
        from public_origin_columns import registries, verify_guidance
        info=dict(info,**registries(value));verify_guidance(value)
    limits = resolve(info["resource_profile"])
    require(type(value["model"]) is dict and value["model"].get("resource_profile")==limits.name and
            value["model"].get("constants")==info["model_constants"] and value["public_constants"]==info["public_constants"],
            "Public weight projected registry")
    require(type(info["model_constants"]) is dict and len(info["model_constants"])<=limits.constants and
            type(info["public_constants"]) is dict and len(info["public_constants"])<=256, "Public weight count")
    for group in ("model_constants","public_constants"):
        total = 0
        for name,item in info[group].items():
            require(type(name) is str and re.fullmatch("[A-Za-z_][A-Za-z0-9_]*",name), "Public weight name")
            require(type(item) is dict and set(item)=={"kind","shape","elements","content_sha256"} and
                    type(item["shape"]) is list and len(item["shape"])<=limits.rank and
                    all(type(n) is int and 1<=n<=limits.axis_size for n in item["shape"]) and
                    item["kind"]==("tensor" if item["shape"] else "scalar") and type(item["elements"]) is int and
                    item["elements"]==math.prod(item["shape"]) and
                    type(item["content_sha256"]) is str and re.fullmatch("[0-9a-f]{64}",item["content_sha256"]),
                    "Public weight descriptor shape/hash")
            require(item["elements"]<=limits.constant_elements, "Public weight descriptor elements")
            total += item["elements"]
        require(total <= (limits.total_constant_elements if group=="model_constants" else 256*4096),
                "Public weight aggregate elements")
    if info["policy"] in (MATH_POLICY,COMPACT_POLICY):
        from public_math_context import verify_projected
        verify_projected(value)
    require(value["request_id"]==request_digest(value), "Public weight context hash")
    return copy.deepcopy(value)

def parse_data(raw):
    """Strict parser for the trusted host data envelope."""
    require(type(raw) is str and len(raw.encode())<=TRUSTED_BYTES, "Trusted trace payload budget")
    def pairs(items):
        value={}
        for k,v in items:
            require(k not in value,"Duplicate trace payload key")
            value[k]=v
        return value
    def nonfinite(_):
        raise ValueError("Nonfinite trace payload")
    def real(token):
        value=float(token);require(math.isfinite(value),"Nonfinite trace payload");return value
    try:
        value=json.loads(raw,object_pairs_hook=pairs,parse_constant=nonfinite,parse_float=real)
    except (RecursionError,MemoryError) as error:
        raise ValueError("Trace payload parsing resources") from error
    return value

def read_payload(raw):
    """Separate trusted payload parsing; the generated candidate stays at 128 KiB."""
    value=parse_data(raw)
    require(type(value) is dict and set(value)=={"request","candidate"}, "Trace payload fields")
    require(len(raw.encode())<=storage_limit(value["request"]), "Trace payload policy budget")
    require(len(canonical(value["candidate"]))<=CONTEXT_BYTES, "Generated candidate response budget")
    return value


def read_request(raw):
    value=parse_data(raw)
    require(type(value) is dict and len(raw.encode())<=storage_limit(value),"Request storage policy budget")
    return value
