"""Freeze spatial source requests through the existing application; no compilation, FHE or provider."""
import copy,hashlib,json,os,shlex,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
from hecate_python_env import ROOT,WORK,VENV,enter_nix
if "--inside"not in sys.argv:
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=600,environment="poseidon-cpu"))
from application_identity import current_identity_manifest
from application_execution.context import WorkerContext
from application_execution.backend import application
from benchmark_graph import digest,require
from component_contract import reconstruct_request
from unified_graph_lowering import candidate_source
from candidate_contract import validate_candidate
from candidate_bundle import check_provenance
def read(p):return json.loads(Path(p).read_text())
def write(p,data):
    with Path(p).open("x")as f:json.dump(data,f,indent=2,allow_nan=False);f.write("\n")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned source preparation")
start=time.monotonic();sdk=current_identity_manifest()
prior=read(ROOT/"docs/baseline/agent-spatial-offline-proposal-r1400.json")
require(sdk["backend"]==prior["backend_identity"],"Backend unchanged")
ready=HERE/"ready-r1405";ready.mkdir()
receipts=WORK/"results"/("agent-spatial-source-prepare-r1405-"+str(time.time_ns()));receipts.mkdir()
stage=WORK/"results/agent-spatial-source-component-stage-1791728068527392953"
sandbox=WORK/"results/agent-spatial-source-sandbox-stage-1791728344527028739"
public=BASE/"cases/pytorch-spatial-v27"
selected=[
 ("resize_1_nearest_0",stage/"resize_1_nearest_0/source/manifest.json"),
 ("resize_1_nearest_exact_1",stage/"resize_1_nearest_exact_1/source/manifest.json"),
 ("resize_1_linear_0",public/"linear_resize/manifest.json"),
 ("resize_2_nearest_1",stage/"resize_2_nearest_1/source/manifest.json"),
 ("resize_2_nearest_exact_0",stage/"resize_2_nearest_exact_0/source/manifest.json"),
 ("resize_2_bilinear_2",stage/"resize_2_bilinear_2/source/manifest.json"),
 ("transpose_1_0",public/"grouped_conv_transpose/manifest.json"),
 ("transpose_1_1",stage/"transpose_1_1/source/manifest.json"),
 ("transpose_2_0",stage/"transpose_2_0/source/manifest.json"),
 ("transpose_2_1",stage/"transpose_2_1/source/manifest.json"),
 ("composition_0",stage/"composition_0/source/manifest.json"),
 ("composition_2",stage/"composition_2/source/manifest.json"),
 ("structured_dict",public/"masked_resize/manifest.json"),
 ("nested_keywords",sandbox/"nested_keywords-source/manifest.json"),
 ("float32_public_weight",sandbox/"float32_public_weight-source/manifest.json"),
 ("compat_v26_keywords0",sandbox/"compat_v26_keywords0-source/manifest.json"),
 ("compat_v26_nested0",sandbox/"compat_v26_nested0-source/manifest.json"),
]
legacy=next(c for c in prior["cases"]if c["case"]=="compat_v1_linear")
selected.append(("compat_v1_linear",None))
repeat={"structured_dict","nested_keywords","float32_public_weight"}
cases=[];all_source=[]
for name,manifest in selected:
    folder=ready/name;folder.mkdir();case=dict(case=name,resource_profile="general-tensors-v1",repeat_package=name in repeat)
    if manifest:
        original_spec=read(manifest);source=folder/"source";source.mkdir()
        for fn in original_spec["files"]:
            content=(manifest.parent/fn).read_text()
            content=content.replace("from pytorch_input_binding import CALL_FORMAT",'CALL_FORMAT="poseidon-pytorch-call-v1"')
            (source/fn).write_text(content)
        spec=copy.deepcopy(original_spec)
        spec["files"]={fn:sha(source/fn)for fn in spec["files"]}
        for info in spec["weights"].values():
            if not(source/info["file"]).exists():(source/info["file"]).write_bytes((manifest.parent/info["file"]).read_bytes())
        write(source/"manifest.json",spec);case.update(source_manifest=str(source/"manifest.json"),source_contract=spec["format"],
            historical_source_manifest=str(manifest),historical_source_manifest_sha256=sha(manifest))
        request_source=dict(format="poseidon-model-source-request-v1",kind="trusted_pytorch",
            manifest=case["source_manifest"],trust_local_code=True)
    else:
        original=read(legacy["original"]);write(folder/"original.json",original)
        case.update(original=str(folder/"original.json"),source_contract=original["format"])
        request_source=dict(format="poseidon-model-source-request-v1",kind="graph",model=original)
    attempts=[]
    for period in(4,8,16,32,64,128,256):
        options=copy.deepcopy(prior["options"]);options["chunk_period"]=period
        app=application(WorkerContext.capture(receipts/name/("p"+str(period))/"tasks"))
        task=app.submit_source(request_source,resource_profile="general-tensors-v1",
            options=options,admission_policy="structured-v1",limits=prior["generation_limits"],
            operational_limits=prior["request_operational_limits"])
        try:
            result={}
            result=app.prepare_source(task)
            require(result["status"]=="queued"and result["intake"]["reference_verified"]is True,"Source admission")
            require(result["generations"]==result["http_attempts"]==0,"No provider calls")
            meta=app._checked_task_metadata(task);request=meta["request"];provenance=meta["model_provenance"]
            check_provenance(provenance,request["model"]);require(reconstruct_request(request)==request,"Request reconstruction")
            candidate,lowering=candidate_source(request)
            static=validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=candidate),request)
            for file,value in(("model.json",request["model"]),("provenance.json",provenance),("request.json",request),
                               ("lowering.json",lowering),("static.json",static)):
                write(folder/file,value)
            (folder/"candidate.py").write_text(candidate)
            case.update(preflight_status="passed",options=options,model=str(folder/"model.json"),
                provenance=str(folder/"provenance.json"),request=str(folder/"request.json"),
                request_id=request["request_id"],candidate=str(folder/"candidate.py"),source_task=task,
                source_task_directory=str(app._dir(task)))
            attempts.append(dict(period=period,status="static_passed",task=task))
            break
        except Exception as error:
            attempts.append(dict(period=period,status="blocked",layer="source_or_rule_preflight",
                type=type(error).__name__,message=str(error)[:1600],task=task))
            (receipts/(name+"-p"+str(period)+"-failure.txt")).write_text(traceback.format_exc())
            if result.get("status") not in("queued","unsupported"):
                break
    case["layout_attempts"]=attempts
    if case.get("preflight_status")!="passed":
        case.update(preflight_status="blocked",preflight_failure=attempts[-1])
        write(folder/"preflight-failure.json",case["preflight_failure"])
    case["files"]={str(p):sha(p)for p in folder.rglob("*")if p.is_file()}
    cases.append(case)
    print(json.dumps(dict(case=name,preflight_status=case["preflight_status"],period=case.get("options",{}).get("chunk_period"))),flush=True)
write(HERE/"selected-offline-cases-r1405.json",cases)
write(HERE/"sdk-r1405.json",sdk)
scope=copy.deepcopy(prior["scope"]);scope.update(maximum_compiler_calls=len(cases),maximum_encrypted_calls=len(cases)+3*len(repeat))
plan=dict(format="poseidon-spatial-source-offline-ready-v1",approved=False,backend="poseidon_cpu",
    candidate_origin="frozen_deterministic_rule",provider_calls=0,http_attempts=0,
    output=str(WORK/"results/agent-spatial-source-offline-r1405"),sdk_identity=sdk["manifest_sha256"],
    backend_identity=sdk["backend"],options=prior["options"],generation_limits=prior["generation_limits"],
    request_operational_limits=prior["request_operational_limits"],execution_operational_limits=prior["execution_operational_limits"],
    scope=scope,cases=cases,tools={str(p):sha(p)for p in(HERE/"offline_source_r1405.py",Path(__file__).resolve())},
    preparation=dict(result=str(receipts),source_capture="existing isolated application intake",
                     actual_compilations=0,encrypted_calls=0,provider_calls=0),
    result_claim="new PyTorch source route; deterministic rule compilation and Poseidon numerical execution pending approval")
plan["binding"]=digest(plan);write(ROOT/"docs/baseline/agent-spatial-source-offline-proposal-r1405.json",plan)
require(current_identity_manifest()==sdk,"SDK unchanged during preparation")
print(json.dumps(dict(cases=len(cases),preflight_passed=sum(c["preflight_status"]=="passed"for c in cases),
    blocked=sum(c["preflight_status"]!="passed"for c in cases),repeat_packages=sorted(repeat),
    sdk=sdk["manifest_sha256"],binding=plan["binding"],seconds=time.monotonic()-start,
    compiler_calls=0,encrypted_calls=0,provider_calls=0)),flush=True)
raise SystemExit(1 if any(c["preflight_status"]!="passed"for c in cases)else 0)
