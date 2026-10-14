"""Bound resource facts for application preflight; no generation or execution."""
import ast,copy,hashlib,inspect,json,math
from collections import Counter
from pathlib import Path

FORMAT="poseidon-request-resource-analysis-v1"
POLICY="poseidon-input-u64-payload-v1"
ROOT=Path(__file__).resolve().parents[3]

def require(ok,code):
    if not ok:raise ValueError(code)

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def sha(raw):return hashlib.sha256(raw).hexdigest()

def source_authority():
    from application_identity import source_closure,POLICY as authority_policy
    closure=source_closure(ROOT/"scripts/baseline",("application_planning/resources.py",))
    return dict(policy=authority_policy,sha256=sha(canonical(closure)),sources=closure["sources"])

def _integer(node):
    # Only evaluate the literal arithmetic used by the trusted sandbox policy.
    if isinstance(node,ast.Constant):
        require(type(node.value) is int and 0<=node.value<=2**48,"resource_limit_literal")
        return node.value
    require(isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Mult,ast.Pow)),
            "resource_limit_expression")
    a,b=_integer(node.left),_integer(node.right)
    require(not isinstance(node.op,ast.Pow) or b<=16,"resource_limit_exponent")
    value=a*b if isinstance(node.op,ast.Mult) else a**b
    require(value<=2**48,"resource_limit_overflow")
    return value

def native_address_limit():
    """Read the existing policy without applying its limits to the host process."""
    from sandbox_policy import limits
    tree=ast.parse(inspect.getsource(limits))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and
           isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and
           n.func.value.id=="resource" and n.func.attr=="setrlimit" and
           len(n.args)==2 and isinstance(n.args[0],ast.Attribute) and
           n.args[0].attr=="RLIMIT_AS"]
    require(len(calls)==1 and isinstance(calls[0].args[1],ast.Tuple),"resource_address_policy")
    values=tuple(_integer(n) for n in calls[0].args[1].elts)
    require(len(values)==2 and 0<values[0]==values[1],"resource_address_policy")
    return values[0]

def parameters(request):
    from compiler_configuration import request_configuration
    from poseidon_runtime_paths import SPEC
    from model_weight_package import regular_bytes
    from candidate_contract import strict_json
    config=request_configuration(request)
    require(config is not None and config["backend"]=="poseidon_cpu","resource_backend")
    raw=regular_bytes(SPEC,65536)
    require(sha(raw)==config["operator_spec_sha256"],"resource_spec_identity")
    spec=strict_json(raw.decode())
    context=spec["context"]
    degree=context["poly_degree"];count=len(context["rns_moduli_log2"])
    require(spec["spec_id"]=="poseidon-agent-cpu-v1" and
        context["context_id"]==config["parameter_identity"] and
        spec["levels"]==dict(lower_bound=0,upper_bound=count-1) and
        degree==32768 and count==13,"resource_parameter_contract")
    # Ciphertext::resize_internal allocates size * degree * data-modulus count
    # uint64_t coefficients. Input encryption produces two components.
    basis=["src/poseidon/ciphertext.cpp","src/poseidon/ciphertext.h",
           "src/poseidon/tools/agent_runtime/cpu_runner.cpp"]
    basis_hashes={name:sha(regular_bytes(ROOT/name,1024**2)) for name in basis}
    return dict(context_id=context["context_id"],operator_spec_sha256=sha(raw),
        poly_degree=degree,data_modulus_count=count,input_level=count-1,
        input_components=2,coefficient_bytes=8,source_basis_sha256=basis_hashes,
        source_basis=["src/poseidon/ciphertext.cpp:122",
                      "src/poseidon/ciphertext.h:57",
                      "src/poseidon/tools/agent_runtime/cpu_runner.cpp:91"])

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
