"""Application policy over existing contracts. No provider or environment imports."""
import copy
from benchmark_graph import OPS,BOUNDS,validate,digest,require
from component_contract import capabilities,prepare_task,failure
BACKEND="upstream_SEAL_HEVM_CPU"
LEVELS=("compiled","numerical")
DEFAULT_BUDGET=dict(generations=4,http_attempts=16,wall_seconds=3600,
                    api_timeout_seconds=1200,max_tokens=384000)
def budget(value):
    require(type(value) is dict and set(value)==set(DEFAULT_BUDGET),"Budget fields")
    for k,maximum in DEFAULT_BUDGET.items():
        limit=43200 if k=="wall_seconds" else maximum
        require(type(value[k]) is int and 1<=value[k]<=limit,"Budget bound: "+k)
    return copy.deepcopy(value)
def prepare(model,backend,level,options):
    require(backend==BACKEND,"Backend not supported")
    require(level in LEVELS,"Explicit validation level required")
    validate(model)
    options=copy.deepcopy(options or {})
    # Select only an existing, hash-versioned context. Never strip mandatory rules.
    if "generation_guidance" not in options:
        if any(n["op"]=="polynomial" and n["attrs"]["basis"]=="chebyshev" for n in model["nodes"]):
            options["generation_guidance"]="explicit-v9"
        elif options.get("construction") is not None:
            options["generation_guidance"]="explicit-v8"
    request=prepare_task(model,options)
    return request,dict(operators=sorted({n["op"] for n in model["nodes"]}),
                        guidance=options.get("generation_guidance"),
                        source="validated graph features; no model-name dispatch")
def describe():
    c=capabilities()
    return dict(**c,application_contract="poseidon-application-v1",validation_levels=list(LEVELS),
                operators={k:sorted(v) for k,v in OPS.items()},
                acceptance="Contract acceptance does not establish compiler or numerical feasibility")
def structured_failure(result):
    raw=result.get("failure") or {}
    layer=raw.get("layer",result.get("public_feedback",{}).get("layer","pipeline"))
    known={"response_parse","source_parse","static_check","dsl_trace","compiler","artifact_gate",
           "seal_runtime","numerical_comparison","integrity","provider","component","pipeline"}
    if layer not in known:layer="pipeline"
    f=failure(layer,"")  # Fixed codes only; no paths, reference or arbitrary messages.
    if layer in ("response_parse","source_parse"):
        f.update(code="candidate_response_rejected",owner="candidate",retry_policy="repair_within_budget")
    codes={
        "required_construct_not_executed":("candidate","repair_within_budget"),
        "construction_evidence_unresolved":("coverage_checker_or_candidate","review"),
        "candidate_contract_rejected":("candidate","repair_within_budget"),
        "numerical_mismatch":("candidate_or_backend","diagnose"),
        "backend_rejected":("candidate_or_backend","diagnose"),
        "integrity_rejected":("host","never_automatic"),
        "candidate_response_rejected":("candidate","repair_within_budget")}
    if raw.get("code") in codes:
        f["code"]=raw["code"];f["owner"],f["retry_policy"]=codes[raw["code"]]
    f.pop("diagnostic",None)
    f["location"]=None
    f["context"]={}
    earth=result.get("compiler_diagnosis")
    if type(earth) is dict and earth.get("format")=="fixed-earth-mul-diagnosis-v1":
        f["context"]={k:earth[k] for k in ("compiler_accumulated_scale","compiler_budget")
                      if type(earth.get(k)) is int}
    return f
def decision(f):
    if f["retry_policy"] in ("never_automatic","review"):
        return "service_error" if f["owner"]=="host" else "validation_failed"
    if f["retry_policy"] in ("repair_within_budget","diagnose"):return "repair"
    return "validation_failed"
