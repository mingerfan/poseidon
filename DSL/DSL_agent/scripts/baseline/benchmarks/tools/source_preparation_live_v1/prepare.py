"""Freeze explicit pre-generation source cases and live Agent requests, without API."""
from pathlib import Path
import sys,shlex,json,time,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=600,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_model_source import FORMAT
from application_policy import DEFAULT_BUDGET
from deepseek_provider import public_request
from public_weight_context import project
from source_preparation_service import identity as entry_identity
checks=q.read(T/"source_preparation_entry_v1/audit.json");q.bound(checks)
prior=q.read(T/"approximation_freeze_v1/acceptance.json");q.bound(prior)
q.require(checks["passed"]==3 and prior["passed"]==9,"offline_acceptance")
sdk=PoseidonBackend().identity_manifest()
q.require(sdk["manifest_sha256"]==prior["sdk"],"same_sdk")
OUT=q.RESULTS/"agent-source-preparation-live-r1645"
q.require(not OUT.exists(),"new_campaign")
q.require(live.measure(q.RESULTS)+4*1024**3<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+4*1024**3,"batch_storage_reservation")
OUT.mkdir(mode=0o700)
rows=[];original=q.RESULTS/"agent-pre-generation-approximation-freeze-r1641"
for case in prior["cases"]:
 name=case["case"];folder=OUT/name;folder.mkdir(mode=0o700)
 frozen=q.read(original/name/"freeze/freeze.json");q.bound(frozen)
 source=Path(frozen["configured_manifest"])
 q.require(q.sha(source)==q.sha(original/name/"capture/trusted-python/manifest.json"),"same_frozen_source")
 store=folder/"tasks";app=application(WorkerContext.capture(store))
 task=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",manifest=str(source),trust_local_code=True),
   resource_profile="general-tensors-v1",options=dict(generation_guidance="explicit-v13",
      public_weight_policy="descriptors-math-roles-v2",chunk_period=4),
   limits=dict(DEFAULT_BUDGET,generations=4,http_attempts=16,wall_seconds=14400))
 q.write(folder/"task-id.json",dict(task_id=task))
 prepared=app.prepare_source(task);q.write(folder/"prepared-state.json",prepared)
 q.require(prepared["status"]=="queued"and prepared["phase"]=="queued","source_admission")
 meta=app._checked_task_metadata(task);request=meta["request"]
 q.write(folder/"request.json",request);q.write(folder/"public-request.json",public_request(request))
 q.require(request["model"]==q.read(original/name/"capture/model.json"),"same_offline_model")
 row=dict(id=name,task_id=task,source_manifest=str(source),source_manifest_sha256=q.sha(source),
  freeze_binding=frozen["binding"],request_id=request["request_id"],model_sha256=q.digest(request["model"]),
  request_file=str(folder/"request.json"),request_sha256=q.sha(folder/"request.json"),
  public_request_file=str(folder/"public-request.json"),public_request_sha256=q.sha(folder/"public-request.json"),
  task_metadata_binding=meta["binding"],generation_guidance="explicit-v13",period=4)
 rows.append(row);print(json.dumps(dict(case=name,request_id=request["request_id"],status="prepared_not_generated")),flush=True)
 q.require(PoseidonBackend().identity_manifest()==sdk,"frozen_sdk")
files={str(p):q.sha(p)for p in HERE.glob("*.py")}
for row in rows:
 for p in Path(row["source_manifest"]).parent.rglob("*"):
  if p.is_file():files[str(p)]=q.sha(p)
 for name in ("request.json","public-request.json","task-id.json"):
  p=OUT/row["id"]/name;files[str(p)]=q.sha(p)
plan=dict(format="poseidon-source-preparation-live-plan-v1",status="authorized_current_plan",
 authorization="human_approved_all_current_plan_offline_and_live_tests",
 campaign=str(OUT),cases=rows,sdk=sdk["manifest_sha256"],entry_identity=entry_identity(),
 offline_source_acceptance_binding=prior["binding"],offline_poseidon_audit_binding=checks["binding"],
 provider=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,
  api_timeout_seconds=1200,max_tokens=384000,max_repairs=3,provider_retries=3),
 budget=dict(api_concurrency=1,encrypted_concurrency=1,compiler_jobs=2,link_jobs=1,
  wall_seconds=43200,case_seconds=14400,max_generations=36,max_http_attempts=144,
  max_compiler_calls=36,max_encrypted_calls=54,new_artifact_bytes=4*1024**3,
  global_result_bytes=48*1024**3,per_case_bytes=4*1024**3,reserve_bytes=4*1024**3),
 outward_content=["public model","public weights/descriptors","layout","DSL rules","controlled feedback"],
 excluded_content=["manual or rule DSL","private inputs","reference answers","credentials","secret keys"],
 monetary_cap=None,retries_may_duplicate_billing=True,prior_unknown_calls_retried=False,
 source_files=files,network_changed=False,parameters_changed=False,backend_changed=False,
 provider_official_verification=dict(url="https://api-docs.deepseek.com/api/create-chat-completion/",
  model="deepseek-flash",reasoning_effort="high",maximum_documented_output_tokens=393216))
plan["binding"]=q.digest(plan);q.write(HERE/"live-plan.json",plan);q.write(OUT/"live-plan.json",plan)
q.write((Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])/"docs/baseline/agent-source-preparation-live-scope-r1645.json",plan)
print(json.dumps(dict(binding=plan["binding"],cases=9,max_generations=36,max_http=144,paid_calls=0)),flush=True)
