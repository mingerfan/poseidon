"""Freeze live requests only after current suffix-stage Poseidon acceptance."""
from pathlib import Path
import sys,shlex,json,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=300,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from application_execution.context import WorkerContext
from application_execution.backend import application
from component_contract import request_options
from deepseek_provider import public_request
prior=q.read(T/"suffix_stage_guidance_v1/offline-plan-r2.json");q.bound(prior)
acceptance=q.read(T/"suffix_stage_guidance_v1/offline-audit-r1652.json");q.bound(acceptance)
current=q.read(T/"suffix_stage_guidance_v1/controls-current-legacy.json");q.bound(current)
q.require(acceptance["passed"]==3and current["passed"]==52and current["errors"]==current["failed"]==0,"current_offline_acceptance")
sdk=PoseidonBackend().identity_manifest()
q.require(sdk["manifest_sha256"]==acceptance["sdk"],"current_sdk")
OUT=q.RESULTS/"agent-suffix-stage-live-r1653";q.require(not OUT.exists(),"new_campaign")
q.require(live.measure(q.RESULTS)+512*1024**2<=48*1024**3and shutil.disk_usage(q.RESULTS).free>=4*1024**3+512*1024**2,"storage_reservation")
OUT.mkdir(mode=0o700);rows=[]
for original in prior["requests"]:
 name=original["case"];req=original["request"];folder=OUT/name;folder.mkdir(mode=0o700)
 app=application(WorkerContext.capture(folder/"tasks"))
 task=app.submit(req["model"],options=request_options(req),limits=dict(generations=4,http_attempts=16,
   wall_seconds=14400,api_timeout_seconds=1200,max_tokens=384000))
 q.write(folder/"task-id.json",dict(task_id=task))
 metadata=app._checked_task_metadata(task);q.require(metadata["request"]==req,"same_model_weights_and_layout")
 q.write(folder/"request.json",req);q.write(folder/"public-request.json",public_request(req))
 rows.append(dict(id=name,task_id=task,request_id=req["request_id"],generation_guidance="explicit-v20",
  model_sha256=q.digest(req["model"]),request_file=str(folder/"request.json"),request_sha256=q.sha(folder/"request.json"),
  public_request_file=str(folder/"public-request.json"),public_request_sha256=q.sha(folder/"public-request.json"),
  task_metadata_binding=metadata["binding"]))
 q.require(PoseidonBackend().identity_manifest()==sdk,"frozen_sdk")
files={str(p):q.sha(p)for p in HERE.glob("*.py")}
for row in rows:
 for name in("request.json","public-request.json","task-id.json"):
  p=OUT/row["id"]/name;files[str(p)]=q.sha(p)
plan=dict(format="poseidon-suffix-stage-live-plan-v1",status="authorized_prepared_not_started",
 authorization="human_approved_all_current_plan_offline_and_live_tests",campaign=str(OUT),cases=rows,
 sdk=sdk["manifest_sha256"],offline_audit_binding=acceptance["binding"],current_controls_binding=current["binding"],
 provider=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,
   api_timeout_seconds=1200,max_tokens=384000,max_repairs=3,provider_retries=3),
 budget=dict(api_concurrency=1,encrypted_concurrency=1,compiler_jobs=2,link_jobs=1,
  wall_seconds=43200,case_seconds=14400,max_generations=12,max_http_attempts=48,
  max_compiler_calls=12,max_encrypted_calls=18,new_artifact_bytes=512*1024**2,
  global_result_bytes=48*1024**3,per_case_bytes=4*1024**3,reserve_bytes=4*1024**3),
 outward_content=["public graph","public weights/descriptors","layout","DSL rules","controlled feedback"],
 excluded_content=["manual or rule DSL","private samples","reference answers","credentials","private keys"],
 monetary_cap=None,retries_may_duplicate_billing=True,prior_unknown_calls_retried=False,
 source_files=files,stream_limits_changed=False,network_changed=False,backend_changed=False,parameters_changed=False)
plan["binding"]=q.digest(plan);q.write(HERE/"live-plan.json",plan);q.write(OUT/"live-plan.json",plan)
q.write((Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])/"docs/baseline/agent-suffix-stage-live-scope-r1653.json",plan)
print(json.dumps(dict(binding=plan["binding"],prepared=len(rows),generations=12,http_attempts=48,api_calls=0,sdk=sdk["manifest_sha256"])))
