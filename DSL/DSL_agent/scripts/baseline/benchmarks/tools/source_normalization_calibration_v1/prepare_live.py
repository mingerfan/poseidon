"""Freeze normal source application requests for origin-aware Agent evaluation."""
from pathlib import Path
import sys,shlex,json,time,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=600,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_model_source import FORMAT
from application_policy import DEFAULT_BUDGET
from deepseek_provider import public_request
from source_preparation_service import identity as entry_identity
from generation_model_origin import verify
from component_contract import reconstruct_request
controls=q.read(HERE/"controls.json");q.bound(controls)
admission=q.read(HERE/"acceptance.json");q.bound(admission)
offline=q.read(HERE/"manual-report.json");q.bound(offline)
audit=q.read(HERE/"audit-r1692-r2.json");q.bound(audit)
q.require(controls["passed"]==118 and controls["failed"]==controls["errors"]==controls["skipped"]==0,"control_acceptance")
q.require(admission["passed"]==offline["passed"]==audit["manual_poseidon_passed"]==6,"source_and_backend_acceptance")
sdk=PoseidonBackend().identity_manifest()
q.require(sdk["manifest_sha256"]==offline["sdk"]==admission["sdk"]==audit["sdk"],"same_checked_sdk")
OUT=q.RESULTS/"agent-normalization-source-live-r1693"
q.require(not OUT.exists(),"new_campaign")
q.require(live.measure(q.RESULTS)+512*1024**2<=48*1024**3and shutil.disk_usage(q.RESULTS).free>=4*1024**3+512*1024**2,"storage_reservation")
OUT.mkdir(mode=0o700);rows=[]
# Explicit test metadata selects cases, never dispatches a model implementation.
order=[r["case"]for r in offline["cases"]]
for name in order:
 previous=next(r for r in offline["cases"]if r["case"]==name)
 source=q.RESULTS/"agent-normalization-source-calibration-r1689"/name/"freeze/source/manifest.json"
 folder=OUT/name;folder.mkdir(mode=0o700)
 app=application(WorkerContext.capture(folder/"tasks"))
 task=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",manifest=str(source),trust_local_code=True),
   resource_profile="general-tensors-v1",options=dict(generation_guidance="explicit-v21",
    public_weight_policy="descriptors-math-roles-v2",chunk_period=4,capacity_packing_version="capacity-public-tensors-v10"),
   limits=dict(DEFAULT_BUDGET,generations=4,http_attempts=16,wall_seconds=14400))
 prepared=app.prepare_source(task);q.write(folder/"prepared-state.json",prepared);q.write(folder/"task-id.json",dict(task_id=task))
 q.require(prepared["status"]=="queued","normal_source_intake")
 meta=app._checked_task_metadata(task);request=meta["request"]
 q.require(request["request_id"]==previous["request_id"],"same_prechecked_request")
 verify(request["generation_model_origin"],meta["model_provenance"],request)
 q.require(reconstruct_request(request)==request,"roundtrip")
 q.write(folder/"request.json",request);q.write(folder/"public-request.json",public_request(request))
 row=dict(id=name,task_id=task,source_manifest=str(source),source_manifest_sha256=q.sha(source),
  request_id=request["request_id"],request_file=str(folder/"request.json"),request_sha256=q.sha(folder/"request.json"),
  public_request_file=str(folder/"public-request.json"),public_request_sha256=q.sha(folder/"public-request.json"),
  model_sha256=q.digest(request["model"]),origin_binding=request["generation_model_origin"]["binding"],
  task_metadata_binding=meta["binding"],generation_guidance="explicit-v21",period=4)
 rows.append(row)
 print(json.dumps(dict(case=name,request_id=request["request_id"],status="prepared_not_generated")),flush=True)
 q.require(PoseidonBackend().identity_manifest()==sdk,"sdk_frozen")
files={str(p):q.sha(p)for p in HERE.glob("*.py")}
for row in rows:
 for p in Path(row["source_manifest"]).parent.rglob("*"):
  if p.is_file():files[str(p)]=q.sha(p)
 for name in("request.json","public-request.json","task-id.json"):
  p=OUT/row["id"]/name;files[str(p)]=q.sha(p)
plan=dict(format="poseidon-normalization-source-live-plan-v1",status="authorized_current_plan",
 authorization="human_approved_all_current_plan_offline_and_live_tests",
 campaign=str(OUT),cases=rows,sdk=sdk["manifest_sha256"],entry_identity=entry_identity(),
 controls_binding=controls["binding"],source_admission_binding=admission["binding"],
 offline_poseidon_binding=offline["binding"],source_backend_audit_binding=audit["binding"],
 provider=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,
  api_timeout_seconds=1200,max_tokens=384000,max_repairs=3,provider_retries=3),
 budget=dict(api_concurrency=1,encrypted_concurrency=1,compiler_jobs=2,link_jobs=1,
  wall_seconds=43200,case_seconds=14400,max_generations=24,max_http_attempts=96,
  max_compiler_calls=24,max_encrypted_calls=36,new_artifact_bytes=512*1024**2,
  global_result_bytes=48*1024**3,per_case_bytes=4*1024**3,reserve_bytes=4*1024**3),
 outward_content=["public mathematical model and verified high-level origin","public weight descriptors and scalar/polynomial roles","layout","legal DSL rules","controlled feedback"],
 excluded_content=["manual or rule DSL","source code","private inputs","reference answers","credentials","secret keys"],
 monetary_cap=None,retries_may_duplicate_billing=True,prior_unknown_calls_retried=False,
 source_files=files,network_changed=False,parameters_changed=False,backend_changed=False,
 original_application_accuracy_certified=False)
plan["binding"]=q.digest(plan);q.write(HERE/"live-plan.json",plan);q.write(OUT/"live-plan.json",plan)
q.write((Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])/"docs/baseline/agent-normalization-source-live-scope-r1693.json",plan)
launch=dict(format="poseidon-bounded-live-launch-v1",plan_binding=plan["binding"],
 runner_sha256=q.sha(HERE/"execute.py"),authorization="human_current_plan_all_tests_approved")
launch["binding"]=q.digest(launch);q.write(HERE/"launch.json",launch);q.write(OUT/"launch.json",launch)
print(json.dumps(dict(binding=plan["binding"],launch_binding=launch["binding"],cases=6,
 max_generations=24,max_http=96,api_calls=0)),flush=True)
