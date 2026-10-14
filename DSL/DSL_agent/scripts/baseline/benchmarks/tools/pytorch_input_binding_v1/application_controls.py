"""Real source admission over the application interface; no generation/compile/FHE."""
import json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[2]))
from benchmark_graph import require,digest
from hecate_python_env import VENV,WORK,ROOT
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_identity import current_identity_manifest
from component_contract import reconstruct_request
require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned application controls")
def read(p):return json.loads(Path(p).read_text())
def write(path,value):
    with path.open("x")as stream:json.dump(value,stream,sort_keys=False,indent=2,allow_nan=False);stream.write("\n")
plan=read(ROOT/"docs/baseline/agent-application-control-offline-proposal-r1382.json")
selected=[dict(case=n,source_manifest=str(HERE/"cases-final"/n/"manifest.json"),contract="poseidon-trusted-pytorch-v26",
             expected_model=str(WORK/"results/agent-pytorch-input-binding-source-r1389-final"/n/"model.json"))
          for n in("dictionary0","nested1","defaults2")]
selected +=[dict(case="compat_"+r["case"],source_manifest=r["source_manifest"],contract=r["source_contract"],
             expected_request=r["request"])for r in plan["cases"]]
root=WORK/"results"/"agent-input-binding-application-r1389";root.mkdir()
sdk=current_identity_manifest();start=time.monotonic();rows=[]
for case in selected:
    directory=root/case["case"];directory.mkdir()
    app=application(WorkerContext.capture(directory/"tasks"))
    task=app.submit_source(dict(format="poseidon-model-source-request-v1",kind="trusted_pytorch",
           manifest=case["source_manifest"],trust_local_code=True),resource_profile="general-tensors-v1",
           options=plan["options"],limits=plan["generation_limits"],
           admission_policy="structured-v1",operational_limits=plan["request_operational_limits"])
    initial=app.status(task);require(initial["status"]=="queued"and initial["phase"]=="intake_queued","Queued source intake")
    prepared=app.prepare_source(task)
    write(directory/"prepared-status.json",prepared)
    require(prepared["status"]=="queued"and prepared["intake"]["reference_verified"],"Application source/reference ready")
    require(prepared["generations"]==prepared["http_attempts"]==0 and prepared["acceptance"]is None,"No execution acceptance")
    metadata=app._checked_task_metadata(task);request=metadata["request"]
    require(request==reconstruct_request(request),"Request reconstruction equality")
    if "expected_model"in case:
        require(request["model"]==read(case["expected_model"]),"Application/direct model equality")
    else:
        require(request==read(case["expected_request"]),"Old request exact equality")
    require(metadata["model_provenance"]["trusted_pytorch"]["manifest"]["format"]==case["contract"],"Explicit source version")
    write(directory/"request.json",request)
    write(directory/"metadata.json",metadata)
    cancelled=app.cancel(task);require(cancelled["status"]=="cancelled","No generation queued after check")
    row=dict(case=case["case"],source_manifest=case["source_manifest"],source_contract=case["contract"],
             task=task,metadata=str(directory/"metadata.json"),request=str(directory/"request.json"),
             request_id=request["request_id"],status="passed",cancelled=True,source_reference_verified=True,
             compiled=False,encrypted_execution=False,agent_generated=False,provider_calls=0,http_attempts=0)
    rows.append(row);write(directory/"checked.json",row);print(case["case"],"application prepared+cancelled",flush=True)
require(current_identity_manifest()["manifest_sha256"]==sdk["manifest_sha256"],"Application SDK unchanged")
data=dict(format="poseidon-input-binding-application-controls-v1",sdk_identity=sdk["manifest_sha256"],rows=rows,
          passed=len(rows),failed=0,old_requests_equal=3,seconds=time.monotonic()-start,
          provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_calls=0)
data["binding"]=digest(data);write(HERE/"application-controls.json",data);write(root/"application-controls.json",data)
print(json.dumps({k:v for k,v in data.items()if k!="rows"}),flush=True)
