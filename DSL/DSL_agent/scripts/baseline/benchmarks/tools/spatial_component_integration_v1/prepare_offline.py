"""Prepare bounded spatial rules and requests; never compile or encrypt."""
import pathlib,hashlib,json,sys,os,shlex,copy
HERE=pathlib.Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
from hecate_python_env import ROOT,WORK,VENV,enter_nix
if "--inside"not in sys.argv:
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=240,keep_env=(),environment="poseidon-cpu"))
import numpy as np
from benchmark_graph import digest,require,samples
from benchmark_math import evaluate as core
from benchmark_torch import evaluate as torch_core
from operator_arithmetic_graph import decompose,evaluate_reference
from application_policy import prepare
from application_identity import current_identity_manifest
from component_contract import reconstruct_request
from candidate_contract import validate_candidate
from unified_graph_lowering import candidate_source
from candidate_bundle import check_provenance
require(os.environ.get("IN_NIX_SHELL")=="pure"and pathlib.Path(sys.prefix)==VENV,"Pinned preparation")
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,value):
    with p.open("x")as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
sdk=current_identity_manifest();historical=read(ROOT/"docs/baseline/agent-input-binding-offline-proposal-r1390.json")
require(sdk["backend"]==historical["backend_identity"],"Backend remains exactly unchanged")
options=copy.deepcopy(historical["options"])
ready=HERE/"ready-r1396";ready.mkdir()
models=[read(p)for p in sorted((HERE/"models").glob("*.json"))]
first=next(m["id"]for m in models if m["nodes"][0]["op"]=="resize"and m["nodes"][0]["attrs"]["mode"]=="linear")
second=next(m["id"]for m in models if m["nodes"][0]["op"]=="resize"and m["nodes"][0]["attrs"]["mode"]=="bilinear")
third=next(m["id"]for m in models if m["nodes"][0]["op"]=="conv_transpose2d"and m["nodes"][0]["attrs"]["groups"]>1)
reuse={first,second,third}
models.extend([
dict(format="poseidon-operator-graph-v1",id="compat_v1_linear",inputs=[dict(name="x",shape=[4])],
    constants=dict(w=[[.3,-.2,.1,.4],[-.1,.2,.5,-.25]],b=[.01,-.03]),
    nodes=[dict(id="linear",op="linear",inputs=["x","w","b"],attrs={},outputs=["y"])],
    outputs=[dict(name="output",value="y")]),
dict(format="poseidon-operator-graph-v2",id="compat_v2_dot",inputs=[dict(name="x",shape=[4]),dict(name="z",shape=[4])],
    constants={},nodes=[dict(id="dot",op="dot",inputs=["x","z"],attrs={},outputs=["y"])],outputs=[dict(name="output",value="y")]),
dict(format="poseidon-operator-graph-v3",id="compat_v3_public_subtract",inputs=[dict(name="x",shape=[4])],
    constants=dict(left=.125),nodes=[dict(id="subtract",op="subtract",inputs=["left","x"],attrs={},outputs=["y"])],
    outputs=[dict(name="output",value="y")])])
cases=[]
for model in models:
    name=model["id"];folder=ready/name;folder.mkdir()
    lowered,bound=decompose(model,resource_profile="general-tensors-v1")
    maximum=0.
    for batch in samples(model,16):
        reference=evaluate_reference(model,batch)
        for actual in(core(lowered,batch),torch_core(lowered,batch)):
            for output,value in reference.items():
                np.testing.assert_allclose(actual[output],value,atol=2e-12,rtol=2e-12)
                maximum=max(maximum,float(np.max(np.abs(actual[output]-value))))
    report=dict(schema=1,mode="offline_model_import",original_sha256=digest(model),model_sha256=digest(lowered),
        probes=16,reference_status="passed",max_decomposition_error=maximum,
        approximation_accuracy_certified=False,encrypted_execution=False,agent_generated=False,paid_calls=0)
    provenance=dict(original=model,decomposition=bound,reference_report=report)
    check_provenance(provenance,lowered)
    for file,item in(("original.json",model),("model.json",lowered),("provenance.json",provenance)):write(folder/file,item)
    case=dict(case=name,source_contract=model["format"],resource_profile="general-tensors-v1",
        original=str(folder/"original.json"),model=str(folder/"model.json"),provenance=str(folder/"provenance.json"),
        repeat_package=name in reuse,reference_probes=16,maximum_reference_difference=maximum)
    try:
        request,_=prepare(lowered,"poseidon_cpu","numerical",options)
        require(reconstruct_request(request)==request,"Request unchanged under reconstruction")
        source,lowering=candidate_source(request)
        static=validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
        (folder/"candidate.py").write_text(source)
        for file,item in(("request.json",request),("lowering.json",lowering),("static.json",static)):write(folder/file,item)
        case.update(preflight_status="passed",request=str(folder/"request.json"),candidate=str(folder/"candidate.py"),request_id=request["request_id"])
    except (ValueError,TypeError,KeyError)as error:
        # Preserve every model and failure; no substitution or unapproved execution.
        case.update(preflight_status="blocked",preflight_failure=dict(layer="rule_preflight",type=type(error).__name__,message=str(error)))
        write(folder/"preflight-failure.json",case["preflight_failure"])
    case["files"]={str(p):sha(p)for p in folder.iterdir()if p.is_file()}
    cases.append(case)
require(len(cases)==45,"Frozen count")
write(HERE/"selected-offline-cases-r1396.json",cases)
write(HERE/"sdk-r1396.json",sdk)
scope=copy.deepcopy(historical["scope"]);scope.update(maximum_compiler_calls=45,maximum_encrypted_calls=51)
plan=dict(format="poseidon-spatial-graph-offline-ready-v1",approved=False,backend="poseidon_cpu",
    candidate_origin="frozen_deterministic_rule",provider_calls=0,http_attempts=0,
    output=str(WORK/"results/agent-spatial-graph-offline-r1396"),sdk_identity=sdk["manifest_sha256"],
    backend_identity=sdk["backend"],options=options,generation_limits=historical["generation_limits"],
    request_operational_limits=historical["request_operational_limits"],
    execution_operational_limits=historical["execution_operational_limits"],
    scope=scope,cases=cases,tools={str(p):sha(p)for p in(HERE/"offline_cpu.py",pathlib.Path(__file__))},
    plaintext_checks=dict(new_contexts=42,legacy_contexts=3,probes_per_model=16),
    result_claim="rule candidates; actual compilation and Poseidon numeric validation pending approval")
plan["binding"]=digest(plan)
write(ROOT/"docs/baseline/agent-spatial-offline-proposal-r1396.json",plan)
require(current_identity_manifest()["manifest_sha256"]==sdk["manifest_sha256"],"SDK unchanged during preparation")
print(json.dumps(dict(cases=len(cases),eligible=sum(c["preflight_status"]=="passed"for c in cases),
    blocked=sum(c["preflight_status"]!="passed"for c in cases),package_reuse_cases=sorted(reuse),
    sdk=sdk["manifest_sha256"],proposal_binding=plan["binding"],compiler_calls=0,encrypted_calls=0,provider_calls=0)))
