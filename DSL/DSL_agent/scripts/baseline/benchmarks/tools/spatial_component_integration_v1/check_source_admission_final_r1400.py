"""Pinned component checks only; no compiler, encryption or model API."""
import sys,json,hashlib,os,shlex,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,WORK,enter_nix
if "--inside"not in sys.argv:
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=180,keep_env=(),environment="poseidon-cpu"))
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_identity import current_identity_manifest
from benchmark_graph import digest,require
OUTPUT=WORK/"results"/"agent-spatial-source-admission-r1400";OUTPUT.mkdir()
plan=json.loads(((Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])/"docs/baseline/agent-spatial-offline-proposal-r1400.json").read_text())
require(current_identity_manifest()["manifest_sha256"]==plan["sdk_identity"],"Source SDK binding")
selected=[c for c in plan["cases"]if c["repeat_package"]];rows=[]
for case in selected:
    app=application(WorkerContext.capture(OUTPUT/case["case"]/"tasks"))
    original=json.loads(Path(case["original"]).read_text())
    task=app.submit_source(dict(format="poseidon-model-source-request-v1",kind="graph",model=original),
        resource_profile=case["resource_profile"],options=case["options"],admission_policy="structured-v1",
        operational_limits=plan["request_operational_limits"],limits=plan["generation_limits"])
    status=app.prepare_source(task);meta=app._checked_task_metadata(task)
    require(status["status"]=="queued"and status["intake"]["reference_verified"]is True,"Source reference")
    require(meta["request"]==json.loads(Path(case["request"]).read_text()),"Exact prepared request")
    require(meta["model_provenance"]["original"]==original,"Exact original model")
    rows.append(dict(case=case["case"],status="source_admission_passed",request_id=case["request_id"],
                     task_id=task,reference_verified=True,encrypted_execution=False,agent_generated=False))
require(current_identity_manifest()["manifest_sha256"]==plan["sdk_identity"],"Unchanged source SDK")
report=dict(format="poseidon-spatial-source-admission-v1",status="passed",sdk_identity=plan["sdk_identity"],
    proposal_binding=plan["binding"],rows=rows,api_calls=0,compilations=0,encrypted_calls=0)
report["binding"]=digest(report)
with(OUTPUT/"source-admission.json").open("x")as f:json.dump(report,f,sort_keys=True,indent=2)
print(json.dumps(report))
