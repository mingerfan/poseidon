"""Bound resource facts for application preflight; no generation or execution."""
import copy,hashlib,json,math
from collections import Counter
from pathlib import Path

FORMAT="poseidon-request-resource-analysis-v1"
POLICY="poseidon-input-u64-payload-v1"
ROOT=(Path(__file__).resolve().parents[5] if Path(__file__).resolve().parents[3].name == "DSL_agent" and Path(__file__).resolve().parents[4].name == "DSL" else Path(__file__).resolve().parents[3])

def require(ok,code):
    if not ok:raise ValueError(code)

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def sha(raw):return hashlib.sha256(raw).hexdigest()

def source_authority():
    from application_identity import source_closure,POLICY as authority_policy
    closure=source_closure(ROOT/"scripts/baseline",("application_planning/resources.py",))
    return dict(policy=authority_policy,sha256=sha(canonical(closure)),sources=closure["sources"])

from application_planning.backend_parameters import _integer, native_address_limit, parameters

def logical_graph(model,checked):
    kinds=checked["kinds"];shapes=checked["shapes"]
    elements=lambda name:math.prod(shapes[name])
    depth={name:0 for name in shapes if name in model["constants"] or
           name in {v["name"] for v in model["inputs"]}}
    opaque=[]
    for node in model["nodes"]:
        cipher=[depth[r] for r in node["inputs"] if kinds[r]=="cipher"]
        value=None if any(v is None for v in cipher) else max(cipher,default=0)
        if node["op"]=="polynomial":
            value=None;opaque.append(node["id"])
        elif value is not None:
            if node["op"]=="square":value+=1
            elif node["op"]=="power":value+=1 if node["attrs"]["exponent"]==2 else 2
            elif node["op"]=="multiply" and kinds[node["inputs"][1]]=="cipher":value+=1
        for name in node["outputs"]:depth[name]=value
    outputs=[depth[item["value"]] for item in model["outputs"]]
    path=None if any(v is None for v in outputs) else max(outputs)
    return dict(node_count=len(model["nodes"]),operator_counts=dict(sorted(Counter(
        n["op"] for n in model["nodes"]).items())),
        tensor_definition_elements=sum(elements(n) for n in shapes if kinds[n]=="cipher"),
        tensor_definition_float64_payload_bytes=8*sum(elements(n) for n in shapes if kinds[n]=="cipher"),
        largest_cipher_tensor_elements=max(elements(n) for n in shapes if kinds[n]=="cipher"),
        cipher_multiply_path=dict(value=path,certainty="unknown" if path is None else "exact",
            scope="syntactic core graph: cipher/cipher products only; power4 uses two squares",
            opaque_polynomial_nodes=len(opaque),
            claim="not CKKS level consumption or a lower bound on an optimized candidate"),
        storage_scope="logical definitions including aliases; not allocated or live runtime memory")

def analyze_request(request,*,memory_budget_bytes=None,sample_count=4):
    from component_contract import reconstruct_request
    from benchmark_graph import validate
    from public_weight_context import descriptor
    require(type(sample_count) is int and 1<=sample_count<=16,"resource_sample_count")
    reconstruct_request(copy.deepcopy(request))
    ceiling=native_address_limit()
    require(memory_budget_bytes is None or
        type(memory_budget_bytes) is int and 1<=memory_budget_bytes<=ceiling,
        "resource_memory_budget")
    budget=ceiling if memory_budget_bytes is None else memory_budget_bytes
    params=parameters(request)
    layout=request["layout"];model=request["model"];checked=validate(model)
    inputs=len(layout["inputs"]);aux=len(layout["auxiliary_ciphertexts"])
    count=inputs+aux
    per=params["poly_degree"]*params["data_modulus_count"]*params["input_components"]*params["coefficient_bytes"]
    floor=count*per
    public=sum(math.prod(descriptor(value)["shape"]) for value in request["public_constants"].values())
    resources=dict(
        input_ciphertexts=dict(value=inputs,certainty="exact",basis="canonical input bindings"),
        auxiliary_ciphertexts=dict(value=aux,certainty="exact",basis="separately encrypted trusted zero"),
        output_declarations=dict(value=layout["output_ciphertexts"],certainty="exact",
            basis="canonical selectors; may alias existing objects"),
        input_ciphertext_coefficient_payload_bytes=dict(value=floor,certainty="conservative",bound="lower",
            basis="all input ciphertexts coexist before runtime; excludes keys, overhead and intermediates"),
        packed_sample_float64_payload_bytes=dict(value=count*layout["input_slot_period"]*8,certainty="exact",
            basis="one sample of physical periods; excludes array objects, packing temporaries and JSON"),
        packed_dataset_float64_payload_bytes=dict(value=sample_count*count*layout["input_slot_period"]*8,
            certainty="exact",basis="logical packed dataset payload; not peak resident memory"),
        public_registry_float64_payload_bytes=dict(value=8*public,certainty="exact",
            basis="bound logical arrays; excludes encoded plaintexts and unused-registry elimination"),
        actual_intermediate_ciphertexts=dict(value=None,certainty="unknown",basis="requires candidate"),
        ckks_level_consumption=dict(value=None,certainty="unknown",basis="requires compiler schedule"),
        evaluation_key_bytes=dict(value=None,certainty="unknown",basis="requires compiled operations"),
        peak_memory_bytes=dict(value=None,certainty="unknown",basis="requires runtime allocations"),
        execution_seconds=dict(value=None,certainty="unknown",basis="requires measured execution"))
    authority=source_authority()
    value=dict(format=FORMAT,policy=POLICY,status="assessed" if floor<=budget else "blocked",
        request_id=request["request_id"],request_sha256=sha(canonical(request)),
        layout_sha256=sha(canonical(layout)),compiler_configuration_sha256=sha(canonical(request["compiler_configuration"])),
        parameters=params,logical=logical_graph(model,checked),resources=resources,
        limits=dict(memory_budget_bytes=budget,native_address_limit=ceiling,sample_count=sample_count),
        requested_memory_budget_bytes=memory_budget_bytes,source_authority=authority,
        execution_readiness="unknown",compiled=False,encrypted_execution=False,paid_calls=0,
        failure=None if floor<=budget else dict(layer="resource",code="input_ciphertext_storage_exceeds_budget",
            location=None,context=dict(required_lower_bytes=floor,budget_bytes=budget,
                                      scope="selected layout input ciphertext coefficients")),
        claim="lower bound can reject an insufficient budget; a fitting lower bound does not certify execution")
    value["binding"]=sha(canonical(value))
    require(authority==source_authority(),"resource_analysis_source_drift")
    return value

def check_analysis(value,request):
    require(type(value) is dict and value.get("format")==FORMAT and value.get("policy")==POLICY,
            "resource_analysis_format")
    require(value.get("binding")==sha(canonical({k:v for k,v in value.items() if k!="binding"})),
            "resource_analysis_binding")
    expected=analyze_request(request,memory_budget_bytes=value["requested_memory_budget_bytes"],
                             sample_count=value["limits"]["sample_count"])
    require(canonical(value)==canonical(expected),"resource_analysis_changed")
    return value

def assess_plan(planned,model,*,memory_budget_bytes=None,sample_count=4):
    from application_planning.layout import check_plan
    request=check_plan(planned,model)
    return analyze_request(request,memory_budget_bytes=memory_budget_bytes,sample_count=sample_count)

def submit_planned(app,model,planned,analysis,**requirements):
    """Budget gate over the same existing Application; no extra model call."""
    from application_planning.layout import check_plan,submit_planned as submit
    request=check_plan(planned,model)
    check_analysis(analysis,request)
    require(analysis["status"]=="assessed","resource_submission_blocked")
    result=submit(app,model,planned,**requirements)
    result["resource_analysis_binding"]=analysis["binding"]
    return result
