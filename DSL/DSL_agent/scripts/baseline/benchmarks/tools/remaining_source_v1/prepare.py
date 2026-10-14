"""Prepare the six static partition and two MHA source requests only. No Provider imports."""
import hashlib,json,os,pathlib,shlex,shutil,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
GUARDS=HERE.parent/"einsum_live_v1"
sys.path.insert(0,str(GUARDS));import runner_scope as guard
from run import OwnedTree
sys.path.insert(0,str(BASE))
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
from hecate_python_env import VENV,enter_nix
WORK=guard.WORK;RESULTS=guard.RESULTS;OUT=RESULTS/"agent-remaining-source-preparation-v1"
OPTIONS=dict(generation_guidance="explicit-v13",public_weight_policy="descriptors-math-roles-v2",chunk_period=4)
PROFILE="general-tensors-v1"
def write(p,v):
 with pathlib.Path(p).open("x")as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def measure(p):return int(subprocess.check_output(["du","-sb",str(p)],text=True,timeout=20).split()[0])
def check_cases():
 cases=guard.read(HERE/"cases.json")
 guard.require(len(cases)==8 and len({c["id"]for c in cases})==8 and
  [c["source_format"]for c in cases]==["poseidon-trusted-pytorch-v17"]*6+["poseidon-trusted-pytorch-v16"]*2,
  "original_remaining_source_cases")
 for c in cases:guard.require(guard.sha(guard.regular(c["source_manifest"]))==c["source_manifest_sha256"],"original_source_changed")
 return cases
def boundary(plan):
 guard.require(time.time()<plan["deadline_wall"],"source_preparation_time_boundary")
 guard.require(measure(OUT)<=64*1024**2,"source_preparation_space_boundary")
 guard.require(measure(RESULTS)<=32*1024**3 and shutil.disk_usage(WORK).free>=4*1024**3,"global_artifact_boundary")
 guard.require(guard.sha(pathlib.Path(__file__))==plan["tool_sha256"]and
  guard.sha(HERE/"cases.json")==plan["cases_sha256"],"frozen_preparation_inputs")
 guard.verify();check_cases()
def outer():
 guard.require(not OUT.exists(),"fresh_preparation_directory");guard.verify();cases=check_cases()
 created=time.time();plan=dict(format="poseidon-remaining-source-preparation-v1",
  source_stage_identity=guard.verify()["source_stage_identity"],source_case_ids=[c["id"]for c in cases],
  cases_sha256=guard.sha(HERE/"cases.json"),created_wall=created,deadline_wall=created+300,new_bytes=64*1024**2,
  provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_calls=0,
  old_preparation_window_reused=False,old_paid_authorizations_reused=False,
  tool_sha256=guard.sha(pathlib.Path(__file__)),options=OPTIONS,resource_profile=PROFILE)
 plan["binding"]=guard.digest(plan);OUT.mkdir(mode=0o700);write(OUT/"plan.json",plan)
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
 tree=OwnedTree().start()
 try:code=enter_nix(command,seconds=max(1,int(plan["deadline_wall"]-time.time())),keep_env=(),environment="poseidon-cpu")
 finally:background=tree.finish()
 if code or background:
  write(OUT/"termination.json",dict(format="poseidon-remaining-source-preparation-termination-v1",
   exit_code=code,background_processes=background,provider_calls=0,http_attempts=0,
   compiler_calls=0,encrypted_calls=0,automatic_restart=False))
 return code if code else int(background)
def inside():
 guard.require(os.environ.get("IN_NIX_SHELL")=="pure"and pathlib.Path(sys.prefix)==VENV,"pinned_source_worker")
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_backend import PoseidonBackend
 from application_model_source import FORMAT
 from public_weight_context import project
 from candidate_contract import request_file_bytes
 plan=guard.read(OUT/"plan.json");identity=plan["source_stage_identity"];boundary(plan)
 backend=PoseidonBackend()
 guard.require(backend.identity_manifest()==identity and backend.identity()["tools"]==identity["backend"],"actual_backend_identity")
 app=application(WorkerContext.capture(OUT/"tasks"));rows=[]
 for c in check_cases():
  boundary(plan)
  task=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",manifest=c["source_manifest"],trust_local_code=True),
   resource_profile=PROFILE,options=OPTIONS,
   limits=dict(generations=4,http_attempts=16,wall_seconds=14400,api_timeout_seconds=1200,max_tokens=384000))
  state=app.prepare_source(task)
  row=dict(c,task_id=task,source_captured=state.get("intake",{}).get("reference_verified")is True,
   status="prepared"if state["status"]=="queued"and state["phase"]=="queued"else state["status"],
   provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_calls=0,new_live_task_required=True,
   current_identity=identity["manifest_sha256"],original_application_accuracy_certified=False)
  if row["status"]=="prepared":
   guard.require(row["source_captured"]is True,"independent_reference_not_verified")
   meta=app._checked_task_metadata(task);request=meta["request"]
   guard.require(meta["identity"]==identity,"source_stage_identity")
   full=HERE/(c["id"]+".request.json");public=HERE/(c["id"]+".public.json")
   with full.open("xb")as f:f.write(request_file_bytes(request))
   with public.open("xb")as f:f.write(request_file_bytes(project(request)))
   spec=guard.read(c["source_manifest"]);mp=pathlib.Path(c["source_manifest"])
   row.update(request_id=request["request_id"],full_request_file=str(full),full_request_file_sha256=guard.sha(full),
    public_request_file=str(public),public_request_file_sha256=guard.sha(public),public_context_bytes=public.stat().st_size,
    model_sha256=guard.digest(request["model"]),source_files={str((mp.parent/n).relative_to(ROOT)):h for n,h in spec["files"].items()},
    weight_files={str((mp.parent/v["file"]).relative_to(ROOT)):v["sha256"]for v in spec["weights"].values()},
    validation_level="numerical",resource_profile=PROFILE,options=OPTIONS)
   public_restored=guard.read(public)
   from unified_request_storage import restore_bindings
   restore_bindings(public_restored["constant_origins"]["capacity_tensor_packing"])
   guard.require(request==guard.request_read(full)and project(request)==public_restored,"lossless_request_storage")
   guard.require(state["generations"]==state["http_attempts"]==0,"no_generation_or_http")
  else:
   row["failure"]=state.get("failure");row["phase"]=state["phase"]
  write(OUT/(c["id"]+".receipt.json"),row);rows.append(row);boundary(plan)
  print(json.dumps(dict(case=c["id"],status=row["status"],provider_calls=0)),flush=True)
 report=dict(format="poseidon-remaining-source-preparation-report-v1",
  status="prepared"if all(c["status"]=="prepared"for c in rows)else"partially_prepared",
  planned=len(rows),prepared=sum(c["status"]=="prepared"for c in rows),rows=rows,
  plan_binding=plan["binding"],seconds=time.time()-plan["created_wall"],
  source_captures=sum(c["source_captured"]for c in rows),compiler_calls=0,encrypted_calls=0,provider_calls=0,http_attempts=0,
  approved=False,executed_live=False,new_live_tasks_required=True,automatic_restart=False)
 report["binding"]=guard.digest(report);write(OUT/"report.json",report)
 print(json.dumps(dict(status=report["status"],prepared=report["prepared"],planned=len(rows),
                      seconds=report["seconds"],provider_calls=0)),flush=True)
 return 0 if report["prepared"]==8 else 1
if __name__=="__main__":raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
