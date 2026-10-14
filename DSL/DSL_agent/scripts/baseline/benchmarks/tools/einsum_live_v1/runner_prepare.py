"""Source-only preparation checks over the current frozen requests; never constructs a Provider."""
import hashlib,json,os,pathlib,shlex,shutil,subprocess,sys,time
import runner_scope as scope
from run import OwnedTree
ROOT=scope.ROOT;BASE=scope.BASE;WORK=scope.WORK;RESULTS=scope.RESULTS
OUT=RESULTS/"agent-einsum-live-source-preparation-v1"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
def write(p,v):
 with pathlib.Path(p).open("x")as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def measure(p):return int(subprocess.check_output(["du","-sb",str(p)],text=True,timeout=20).split()[0])
def boundary(plan):
 scope.require(time.time()<plan["deadline_wall"],"source_preparation_time_boundary")
 scope.require(measure(OUT)<=64*1024**2,"source_preparation_space_boundary")
 scope.require(measure(RESULTS)<=32*1024**3 and shutil.disk_usage(WORK).free>=4*1024**3,"global_artifact_boundary")
 scope.require(scope.sha(pathlib.Path(__file__))==plan["tool_sha256"],"frozen_preparation_tool")
 scope.verify()
def outer():
 scope.require(not OUT.exists(),"fresh_source_preparation_directory")
 scope.verify();control=scope.read(scope.HERE/"runtime-controls.json")
 scope.require(control["status"]=="passed"and control["tests_run"]==61 and
  control["tools"]==scope.tools(),"runtime_controls_required")
 created=time.time();plan=dict(format="poseidon-einsum-source-preparation-v1",
  proposal_binding=scope.PROPOSAL_BINDING,proposal_sha256=scope.PROPOSAL_SHA,
  created_wall=created,deadline_wall=created+180,source_cases=list(scope.CASES),
  tool_sha256=scope.sha(pathlib.Path(__file__)),new_bytes=64*1024**2,
  provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_calls=0)
 plan["binding"]=scope.digest(plan);OUT.mkdir(mode=0o700);write(OUT/"plan.json",plan)
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
 tree=OwnedTree().start()
 try:code=enter_nix(command,seconds=max(1,int(plan["deadline_wall"]-time.time())),keep_env=(),environment="poseidon-cpu")
 finally:background=tree.finish()
 if code or background:
  write(OUT/"termination.json",dict(format="poseidon-einsum-source-preparation-termination-v1",
   exit_code=code,background_processes=background,provider_calls=0,http_attempts=0,
   compiler_calls=0,encrypted_calls=0,automatic_restart=False))
 return code if code else int(background)
def inside():
 scope.require(os.environ.get("IN_NIX_SHELL")=="pure"and pathlib.Path(sys.prefix)==VENV,"pinned_source_worker")
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_backend import PoseidonBackend
 from application_model_source import FORMAT
 from public_weight_context import project
 plan=scope.read(OUT/"plan.json");proposal=scope.verify();boundary(plan)
 backend=PoseidonBackend()
 scope.require(backend.identity_manifest()==proposal["application_stage_identity"]and
  backend.identity()["tools"]==proposal["source_stage_identity"]["backend"],"actual_backend_identity")
 app=application(WorkerContext.capture(OUT/"tasks"));rows=[]
 for case in proposal["cases"]:
  boundary(plan)
  options=dict(case["options"])
  from unified_capacity_constants import request_version
  options["capacity_packing_version"]=request_version(scope.request_read(case["full_request_file"]))
  task=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",manifest=case["source_manifest"],trust_local_code=True),
   resource_profile=case["resource_profile"],options=options,
   limits=dict(generations=4,http_attempts=16,wall_seconds=14400,api_timeout_seconds=1200,max_tokens=384000))
  state=app.prepare_source(task)
  scope.require(state["status"]=="queued"and state["phase"]=="queued"and
   state.get("intake",{}).get("reference_verified")is True,"source_reference_intake")
  meta=app._checked_task_metadata(task);request=meta["request"]
  scope.require(request==scope.request_read(case["full_request_file"]),"frozen_mathematical_request")
  scope.require(scope.digest(request["model"])==case["model_sha256"]and
   meta["identity"]==proposal["application_stage_identity"],"frozen_model_and_identity")
  public=scope.read(case["public_request_file"])
  from unified_request_storage import restore_bindings
  restore_bindings(public["constant_origins"]["capacity_tensor_packing"])
  scope.require(project(request)==public,"exact_public_context")
  scope.require(state["generations"]==state["http_attempts"]==0,"no_generation_or_http")
  row=dict(case=case["id"],status="prepared",task_id=task,request_id=request["request_id"],
   source_captured=True,independent_reference_verified=True,source_stage_identity=meta["identity"]["manifest_sha256"],
   compiler_calls=0,encrypted_calls=0,provider_calls=0,http_attempts=0,new_live_task_required=True)
  write(OUT/(case["id"]+".receipt.json"),row);rows.append(row);boundary(plan)
  print(json.dumps(dict(case=case["id"],status="source_prepared",provider_calls=0)),flush=True)
 report=dict(format="poseidon-einsum-source-preparation-report-v1",status="prepared",cases=len(rows),rows=rows,
  plan_binding=plan["binding"],proposal_binding=proposal["binding"],seconds=time.time()-plan["created_wall"],
  source_captures=len(rows),compiler_calls=0,encrypted_calls=0,provider_calls=0,http_attempts=0,
  approved=False,executed_live=False,new_live_tasks_required=True,automatic_restart=False)
 report["binding"]=scope.digest(report);write(OUT/"report.json",report)
 print(json.dumps(dict(status="prepared",cases=len(rows),seconds=report["seconds"],provider_calls=0)),flush=True)
 return 0
if __name__=="__main__":raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
