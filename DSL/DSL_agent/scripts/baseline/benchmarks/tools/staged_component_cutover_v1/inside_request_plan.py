"""Prepare exact offline options and statically check three fixed rule programs."""
import hashlib,json,sys
from pathlib import Path
sys.path[:0]=["/tests","/app"]
import numpy as np,torch
from platform_config import require_python_packages
require_python_packages(torch,np)
assert sys.version_info[:3]==(3,10,14)
from application_component import Application
from component_contract import validate_program
from unified_graph_lowering import candidate_source
from deepseek_provider import public_request
from test_joint_component import folder,InertBackend
from test_intake import PROFILE,MEASUREMENTS
from pytorch_ordered_contract import FORMAT
torch.set_num_threads(1)
rows=[]
for name in ("linear","norm","split_weight"):
    imported,provenance,model=folder(name,"offline-plan")
    app=Application(Path("/out")/("request-plan-tasks-"+name),InertBackend())
    options=dict(chunk_period=4,generation_guidance="explicit-v11",
        capacity_packing_version="capacity-public-tensors-v9",public_weight_policy="descriptors-v1")
    task=app.submit_import(imported,options=options)
    assert app.status(task)["status"]=="queued"
    meta=app._checked_task_metadata(task);request=meta["request"]
    assert request["model"]==model and request["generation_guidance"]["version"]=="explicit-v11"
    code=candidate_source(request)[0]
    candidate=dict(schema=1,request_id=request["request_id"],hecate_source=code)
    static=validate_program(candidate,request)
    assert static["status"]=="static_accepted_not_executed",static
    outbound=json.dumps(public_request(request))
    assert code not in outbound
    for private in ("weight_files","trusted_pytorch","reference_report","capture_metadata","capture_emissions"):
        assert private not in outbound,private
    state=app.status(task)
    assert state["generations"]==state["http_attempts"]==0 and state["acceptance"] is None
    (Path("/out")/(name+"-request.json")).write_text(json.dumps(request,sort_keys=True))
    (Path("/out")/(name+"-rule.py")).write_text(code)
    rows.append(dict(case=name,request_id=request["request_id"],resource_profile=PROFILE,
        source_format=FORMAT,generation_guidance="explicit-v11",public_weight_policy="descriptors-v1",
        rule_source_sha256=hashlib.sha256(code.encode()).hexdigest(),static_status=static["status"],
        public_context_checked=True,compiled=False,encrypted_execution=False,
        generations=0,http_attempts=0))
report=dict(format="poseidon-joint-post-cutover-request-preflight-v1",status="passed",models=len(rows),
    rows=rows,source_models=MEASUREMENTS,resource_profile=PROFILE,
    runtime=dict(python=list(sys.version_info[:3]),torch=torch.__version__,numpy=np.__version__),
    provider_calls=0,compilations=0,encrypted_executions=0,production_applied=False,
    scope="Real isolated source/reference and fixed rule static check; no backend or new Agent generation.")
Path("/out/report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({k:report[k] for k in ("status","models","runtime","provider_calls","compilations","encrypted_executions")}))
