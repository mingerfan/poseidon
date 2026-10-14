"""Application policy over existing contracts. No provider or environment imports."""
import copy
from compiler_configuration import CPU_CONFIGURATIONS
from benchmark_graph import OPS,BOUNDS,validate,digest,require
from component_contract import capabilities,prepare_task,failure
BACKEND="poseidon_cpu"
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
    require(options.get("compiler_configuration","poseidon-cpu-eva-w45-v1") in CPU_CONFIGURATIONS,"New tasks require Poseidon CPU")
    # Select only an existing, hash-versioned context. Never strip mandatory rules.
    if "generation_guidance" not in options:
        from unified_poseidon_guidance import relevant
        if relevant(prepare_task(model,options)):
            options["generation_guidance"]="explicit-v10"
        elif options.get("construction") is not None:
            options["generation_guidance"]="explicit-v8"
    request=prepare_task(model,options)
    from application_capabilities import assess_request
    assessment=assess_request(request)
    return request,dict(operators=sorted({n["op"] for n in model["nodes"]}),
                        guidance=options.get("generation_guidance"),
                        capability_report=assessment["summary"],
                        source="validated graph features; no model-name dispatch")
def describe():
    from application_capabilities import describe as capability_matrix
    from application_diagnostics import POLICIES, PRECISE
    c=capabilities()
    return dict(**c,application_contract="poseidon-application-v1",validation_levels=list(LEVELS),
                operators={k:sorted(v) for k,v in OPS.items()},
                capability_matrix=capability_matrix(),
                feedback_policies=list(POLICIES),default_feedback_policy=PRECISE,
                admission_policies=["legacy-v1","structured-v1"],
                operational_limits=dict(format="poseidon-operational-limits-v1",
                    enforcement="monitored-v1",hard_filesystem_quota=False,
                    fields=["max_new_artifact_bytes","min_free_disk_bytes"]),
                acceptance="Contract acceptance does not establish compiler or numerical feasibility")
def structured_failure(result, policy="legacy-v1"):
    from application_diagnostics import POLICIES, PRECISE, from_result
    require(policy in POLICIES, "Feedback policy")
    if policy == PRECISE:
        return from_result(result)
    raw=result.get("failure") or {}
    supplied=result.get("structured_diagnostic")
    if (raw.get("layer")=="resource_gate" or
            type(supplied) is dict and supplied.get("reason_code")=="compiled_storage_budget"):
        # Storage witnesses keep their checked integer context under both policies.
        # Existing legacy reasons retain their original coarse reconstruction.
        return from_result(result)
    layer=raw.get("layer",result.get("public_feedback",{}).get("layer","pipeline"))
    known={"response_parse","source_parse","static_check","dsl_trace","compiler","artifact_gate",
           "poseidon_runtime","numerical_comparison","integrity","provider","component","pipeline"}
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
